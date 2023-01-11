import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import shutil
import torchvision.transforms as transforms
import torchvision.datasets as dset
import augmentations
import logging
from torch.autograd import Variable



class AvgrageMeter(object):

  def __init__(self):
    self.reset()

  def reset(self):
    self.avg = 0
    self.sum = 0
    self.cnt = 0

  def update(self, val, n=1):
    self.sum += val * n
    self.cnt += n
    self.avg = self.sum / self.cnt


def accuracy(output, target, topk=(1,)):
  maxk = max(topk)
  batch_size = target.size(0)

  _, pred = output.topk(maxk, 1, True, True)
  pred = pred.t()
  correct = pred.eq(target.view(1, -1).expand_as(pred))

  res = []
  for k in topk:
    correct_k = correct[:k].reshape(-1).float().sum(0)
    res.append(correct_k.mul_(100.0/batch_size))
  return res


class Cutout(object):
  def __init__(self, length, prob = 1.0):
    self.length = length
    self.prob = prob

  def __call__(self, img):
    if np.random.binomial(1, self.prob):
      h, w = img.size(1), img.size(2)
      mask = np.ones((h, w), np.float32)
      y = np.random.randint(h)
      x = np.random.randint(w)

      y1 = np.clip(y - self.length // 2, 0, h)
      y2 = np.clip(y + self.length // 2, 0, h)
      x1 = np.clip(x - self.length // 2, 0, w)
      x2 = np.clip(x + self.length // 2, 0, w)

      mask[y1: y2, x1: x2] = 0.
      mask = torch.from_numpy(mask)
      mask = mask.expand_as(img)
      img *= mask
    return img


def _data_transforms_svhn(args):
  SVHN_MEAN = [0.4377, 0.4438, 0.4728]
  SVHN_STD = [0.1980, 0.2010, 0.1970]

  train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(SVHN_MEAN, SVHN_STD),
  ])
  if args.cutout:
    train_transform.transforms.append(Cutout(args.cutout_length,
                                             args.cutout_prob))

  valid_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(SVHN_MEAN, SVHN_STD),
  ])
  return train_transform, valid_transform


def _data_transforms_cifar100(args):
  CIFAR_MEAN = [0.5071, 0.4865, 0.4409]
  CIFAR_STD = [0.2673, 0.2564, 0.2762]

  train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
  ])
  if args.cutout:
    train_transform.transforms.append(Cutout(args.cutout_length,
                                             args.cutout_prob))

  valid_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
  ])
  return train_transform, valid_transform


def _data_transforms_cifar10(args):
  CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
  CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

  train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
  ])
  if args.cutout:
    train_transform.transforms.append(Cutout(args.cutout_length, args.cutout_prob))

  valid_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
  return train_transform, valid_transform


def count_parameters_in_MB(model):
  return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e6


def save_checkpoint(state, is_best, save):
  filename = os.path.join(save, 'checkpoint.pth.tar')
  torch.save(state, filename)
  if is_best:
    best_filename = os.path.join(save, 'model_best.pth.tar')
    shutil.copyfile(filename, best_filename)


def save(model, model_path):
  torch.save(model.state_dict(), model_path)


def load(model, model_path):
  model.load_state_dict(torch.load(model_path))


def drop_path(x, drop_prob):
  if drop_prob > 0.:
    keep_prob = 1.-drop_prob
    mask = Variable(torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob))
    x.div_(keep_prob)
    x.mul_(mask)
  return x


def create_exp_dir(path, scripts_to_save=None):
  if not os.path.exists(path):
    os.mkdir(path)
  print('Experiment dir : {}'.format(path))

  if scripts_to_save is not None:
    os.mkdir(os.path.join(path, 'scripts'))
    for script in scripts_to_save:
      dst_file = os.path.join(path, 'scripts', os.path.basename(script))
      shutil.copyfile(script, dst_file)


def process_step_vector(x, method, mask, tau=None):
  if method == 'softmax':
    output = F.softmax(x, dim=-1)
  elif method == 'dirichlet':
    output = torch.distributions.dirichlet.Dirichlet(
      F.elu(x) + 1).rsample()
  elif method == 'gumbel':
    output = F.gumbel_softmax(x, tau=tau, hard=False, dim=-1)
  
  if mask is None:
    return output
  else:
    output_pruned = torch.zeros_like(output)
    output_pruned[mask] = output[mask]
    output_pruned /= output_pruned.sum()
    assert (output_pruned[~mask] == 0.0).all()
    return output_pruned
    

def process_step_matrix(x, method, mask, tau=None):
  weights = []
  if mask is None:
    for line in x:
      weights.append(process_step_vector(line, method, None, tau))
  else:
    for i, line in enumerate(x):
      weights.append(process_step_vector(line, method, mask[i], tau))
  return torch.stack(weights)


def prune(x, num_keep, mask, reset=False):
  if not mask is None:
    x.data[~mask] -= 1000000
  src, index = x.topk(k=num_keep, dim=-1)
  if not reset:
    x.data.copy_(torch.zeros_like(x).scatter(dim=1, index=index, src=src))
  else:
    x.data.copy_(torch.zeros_like(x).scatter(dim=1, index=index, src=1e-3*torch.randn_like(src)))
  mask = torch.zeros_like(x, dtype=torch.bool).scatter(
      dim=1, index=index, src=torch.ones_like(src,dtype=torch.bool))
  return mask


def _data_transforms_cifar10_augmix(args):

  CIFAR_MEAN = [0.5, 0.5, 0.5]
  CIFAR_STD = [0.5, 0.5, 0.5]

  train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ]
    )
  if args.cutout:
    train_transform.transforms.append(Cutout(args.cutout_length, args.cutout_prob))

  valid_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ]
    )
  return train_transform, valid_transform

"""
Implementation of AUGMIX and test corruption as implemented in AUGMIX paper

"""




CORRUPTIONS = [
    'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
    'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
    'brightness', 'contrast', 'elastic_transform', 'pixelate',
    'jpeg_compression'
]

def test(net, test_loader):
    """Evaluate network on given dataset."""
    net.eval()
    net = net.cuda()
    total_loss = 0.
    total_correct = 0
    with torch.no_grad():
        for images, targets in test_loader:
            images, targets = images.cuda(), targets.cuda()
            logits = net(images)
            loss = torch.nn.functional.cross_entropy(logits, targets)
            pred = logits.data.max(1)[1]
            total_loss += float(loss.data)
            total_correct += pred.eq(targets.data).sum().item()

    return total_loss / len(test_loader.dataset), total_correct / len(
        test_loader.dataset)

def test_corr(net, dataset, args):
    """Evaluate network on given corrupted dataset."""
    corruption_accs = []
    base_path = "/work/ws-tmp/g059997-DrNas/DrNAS/data/cifar/"
    test_transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)])
    test_data = dset.CIFAR10(
            root=args.data, train=False, download=True, transform=test_transform
        )
    
    if dataset=="cifar10":
        base_path += "CIFAR-10-C/"        
    elif dataset == "cifar100":
        base_path += "CIFAR-100-C/"
        test_data = dset.CIFAR100(
            root=args.data, train=False, download=True, transform=test_transform
        )
    else:
        raise NotImplementedError               

    for corruption in CORRUPTIONS:
        # Reference to original data is mutated
        test_data.data = np.load(base_path + corruption + '.npy')
        test_data.targets = torch.LongTensor(np.load(base_path + 'labels.npy'))

        test_loader = torch.utils.data.DataLoader(
            test_data,
            batch_size=64,
            shuffle=False,
            num_workers=0,
            pin_memory=True)

        test_loss, test_acc = test(net, test_loader)
        corruption_accs.append(test_acc)
        logging.info('{}\n\tTest Loss {:.3f} | Test Error {:.3f}'.format(
            corruption, test_loss, 100 - 100. * test_acc))

    return (1 - np.mean(corruption_accs))

def aug(image):
  """Perform AugMix augmentations and compute mixture.

  Args:
    image: PIL.Image input image
    preprocess: Preprocessing function which should return a torch tensor.

  Returns:
    mixed: Augmented and mixed image.
  """
  try:
        all_ops = config.all_ops
        mixture_width = config.mixture_width
        mixture_depth = config.mixture_depth
        aug_severity = config.aug_severity
  except Exception as e:
        all_ops = True
        mixture_depth = -1
        mixture_width = 3
        aug_severity = 3
    
    
  aug_list = augmentations.augmentations_all
  ws = np.float32(np.random.dirichlet([1] * mixture_width))
  m = np.float32(np.random.beta(1, 1))

  mix = torch.zeros_like(image)
  image = transforms.ToPILImage()(image)
  for i in range(mixture_width):
    image_aug = image.copy()
    depth = mixture_depth if mixture_depth > 0 else np.random.randint(
        1, 4)
    for _ in range(depth):
      op = np.random.choice(aug_list)
      image_aug = op(image_aug, aug_severity)
    # Preprocessing commutes since all coefficients are convex
    image_aug = transforms.ToTensor()(image_aug)
    mix += ws[i] * (image_aug)
  image = transforms.ToTensor()(image)
  mixed = (1 - m) * (image) + m * mix
  return mixed


class AugMixDataset(torch.utils.data.Dataset):
  """Dataset wrapper to perform AugMix augmentation."""

  def __init__(self, dataset):
    self.dataset = dataset

  def __getitem__(self, i):
    x, y = self.dataset[i]
    im_tuple = (x, aug(x),aug(x))
    return im_tuple, y

  def __len__(self):
    return len(self.dataset)
  
## jsd loss integration with augmix
def jsd_loss(logits_train):
    logits_train, logits_aug1, logits_aug2 = torch.split(logits_train, len(logits_train) // 3)
    p_clean, p_aug1, p_aug2 = F.softmax(logits_train, dim=1), F.softmax(logits_aug1, dim=1), F.softmax(logits_aug2, dim=1)

    p_mixture = torch.clamp((p_clean + p_aug1 + p_aug2) / 3., 1e-7, 1).log()
    augmix_loss = 12 * (F.kl_div(p_mixture, p_clean, reduction='batchmean') +
                F.kl_div(p_mixture, p_aug1, reduction='batchmean') +
                F.kl_div(p_mixture, p_aug2, reduction='batchmean')) / 3.
    return logits_train, augmix_loss