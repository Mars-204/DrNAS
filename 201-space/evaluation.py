import os
import sys
sys.path.insert(0, '../')
import time
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn

from torch.autograd import Variable
from search_model_gdas import TinyNetworkGDAS
from search_model import TinyNetwork
from cell_operations import NAS_BENCH_201
from architect import Architect
from core.query_metrics import Metric
from core.graph import Graph
from copy import deepcopy
from numpy import linalg as LA

from torch.utils.tensorboard import SummaryWriter
from nas_201_api import NASBench201API as API

def evaluate(
        train_queue,
        valid_queue,
        test_queue,
        args,
        architect,
        retrain:bool=True,
        search_model:str="",
        resume_from:str="",
        best_arch:Graph=None,
        dataset_api:object=None,
        metric:Metric=None,
        api: object=None
        
    ):
        """
        Evaluate the final architecture as given from the optimizer.

        If the search space has an interface to a benchmark then query that.
        Otherwise train as defined in the config.

        Args:
            retrain (bool)      : Reset the weigh ts from the architecure search
            search_model (str)  : Path to checkpoint file that was created during search. If not provided,
                                  then try to load 'model_final.pth' from search
            resume_from (str)   : Resume retraining from the given checkpoint file.
            best_arch           : Parsed model you want to directly evaluate and ignore the final model
                                  from the optimizer.
            dataset_api         : Dataset API to use for querying model performance.
            metric              : Metric to query the benchmark for.
        """
        logging.info("Start evaluation")

        #Adding augmix and test corruption error to evalualtion
        test_corr = False
        test_corr = args.test_corr
        augmix_eval = args.augmix_eval

        # measuring stuff
        train_top1 = utils.AvgrageMeter()
        train_top5 = utils.AvgrageMeter()
        train_loss = utils.AvgrageMeter()
        val_top1 = utils.AvgrageMeter()
        val_top5 = utils.AvgrageMeter()
        val_loss = utils.AvgrageMeter()
        # if not best_arch:
        #     if not search_model:
        #         search_model = os.path.join(
        #             self.config.save, "search", "model_final.pth"
        #         )

        #     self._setup_checkpointers(resume_from = search_model, search= True)  # required to load the architecture
        #     best_arch = self.optimizer.get_final_architecture()
        # logging.info("Final architecture:\n" + best_arch.modules_str())
        best_arch = architect
        ## obtaining test accuracies from NASBench201API
        # if best_arch.QUERYABLE:
        #     if metric is None:
        #         metric = Metric.TEST_ACCURACY
        #     result = best_arch.query(
        #         metric=metric, dataset=args.dataset, dataset_api=dataset_api
        #     )
        #     logging.info("Queried results ({}): {}".format(metric, result))
            
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        best_arch = architect
        if retrain == False:
            print(retrain)
        else:
            best_arch.to(device)
            if retrain:
                logging.info("Starting retraining from scratch")
                logging.info("Evaluation with augmix:",args.augmix_eval)
                best_arch.reset_weights(inplace=True)

                optim = build_eval_optimizer(best_arch.arch_parameters(), args)
                scheduler = build_eval_scheduler(optim, args)

                start_epoch = self._setup_checkpointers(
                    resume_from,
                    search=False,
                    period=args.evaluation.checkpoint_freq,
                    model=best_arch,  # checkpointables start here
                    optim=optim,
                    scheduler=scheduler,
                )

                grad_clip = args.evaluation.grad_clip
                loss = torch.nn.CrossEntropyLoss()

                train_top1.reset()
                train_top5.reset()
                val_top1.reset()
                val_top5.reset()

                # Enable drop path
                best_arch.update_edges(
                    update_func=lambda edge: edge.data.set(
                        "op", DropPathWrapper(edge.data.op)
                    ),
                    scope=best_arch.OPTIMIZER_SCOPE,
                    private_edge_data=True,
                )

                # train from scratch
                epochs = args.evaluation.epochs
                for e in range(start_epoch, epochs):
                    best_arch.train()

                    if torch.cuda.is_available():
                        log_first_n(
                            logging.INFO,
                            "cuda consumption\n {}".format(torch.cuda.memory_summary()),
                            n=20,
                        )

                    # update drop path probability
                    drop_path_prob = self.config.evaluation.drop_path_prob * e / epochs
                    best_arch.update_edges(
                        update_func=lambda edge: edge.data.set(
                            "drop_path_prob", drop_path_prob
                        ),
                        scope=best_arch.OPTIMIZER_SCOPE,
                        private_edge_data=True,
                    )

                    # Train queue
                    for i, (input_train, target_train) in enumerate(train_queue):
                        if args.augment:
                            input_train = torch.cat(input_train,0).cuda()
                        
                        input_train = input_train.to(device)
                        target_train = target_train.to(device, non_blocking=True)

                        optim.zero_grad()
                        logits_train = best_arch(input_train)
                        
                        if args.augmix_eval:
                            logits_train, augmix_loss = utils.jsd_loss(logits_train)
                            train_loss = loss(logits_train, target_train)
                            train_loss =  train_loss + augmix_loss
                        elif args.augment and not args.augmix_eval:
                            logits_train, _, _ = torch.split(logits_train, len(logits_train) // 3)
                            train_loss = loss(logits_train, target_train)
                        else:
                            train_loss = loss(logits_train, target_train)

                        if hasattr(
                            best_arch, "auxilary_logits"
                        ):  # darts specific stuff
                            log_first_n(logging.INFO, "Auxiliary is used", n=10)
                            auxiliary_loss = loss(
                                best_arch.auxilary_logits(), target_train
                            )
                            train_loss += (
                                args.evaluation_auxiliary_weight * auxiliary_loss
                            )
                        train_loss.backward()
                        if grad_clip:
                            torch.nn.utils.clip_grad_norm_(
                                best_arch.parameters(), grad_clip
                            )
                        optim.step()

                        _store_accuracies(logits_train, target_train, "train", train_top1,train_top5,val_top1, val_top5)
                        log_every_n_seconds(
                            logging.INFO,
                            "Epoch {}-{}, Train loss: {:.5}, learning rate: {}".format(
                                e, i, train_loss, scheduler.get_last_lr()
                            ),
                            n=5,
                        )

                    # Validation queue
                    if self.valid_queue:
                        best_arch.eval()
                        for i, (input_valid, target_valid) in enumerate(self.valid_queue):
                            if self.augment:
                                input_valid = torch.cat(input_train,0).cuda()
                            input_valid = input_valid.to(self.device).float()
                            target_valid = target_valid.to(self.device).float()

                            # just log the validation accuracy
                            with torch.no_grad():
                                logits_valid = best_arch(input_valid)
                                if self.augmix_eval:
                                    logits_valid,_ ,_ = torch.split(logits_train, len(logits_valid) // 3)
                                self._store_accuracies(
                                    logits_valid, target_valid, "val"
                                )

                    scheduler.step()
                    self.periodic_checkpointer.step(e)
                    self._log_and_reset_accuracies(e)

            # Disable drop path
            best_arch.update_edges(
                update_func=lambda edge: edge.data.set(
                    "op", edge.data.op.get_embedded_ops()
                ),
                scope=best_arch.OPTIMIZER_SCOPE,
                private_edge_data=True,
            )

            # measure final test accuracy
            top1 = utils.AvgrageMeter()
            top5 = utils.AvgrageMeter()

            best_arch.eval()

            for i, data_test in enumerate(test_queue):
                input_test, target_test = data_test
                input_test = input_test.to(device)
                target_test = target_test.to(device, non_blocking=True)

                n = input_test.size(0)

                with torch.no_grad():
                    logits = best_arch(input_test)

                    prec1, prec5 = utils.accuracy(logits, target_test, topk=(1, 5))
                    top1.update(prec1.data.item(), n)
                    top5.update(prec5.data.item(), n)

                # log_every_n_seconds(
                #     logging.INFO,
                #     "Inference batch {} of {}.".format(i, len(self.test_queue)),
                #     n=5,
                # )

            logging.info(
                "Evaluation finished. Test accuracies: top-1 = {:.5}, top-5 = {:.5}".format(
                    top1.avg, top5.avg
                )
            )
            if test_corr:
                mean_CE = utils.test_corr(best_arch, self.eval_dataset, self.config)
                logging.info(
                "Corruption Evaluation finished. Mean Corruption Error: {:.9}".format(
                    mean_CE
                )
            )

def build_eval_optimizer(parameters, args):
        return torch.optim.SGD(
            parameters,
            lr=args.evaluation.learning_rate,
            momentum=args.evaluation.momentum,
            weight_decay=args.evaluation.weight_decay,
        )

def build_eval_scheduler(optimizer, args):
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.evaluation.epochs,
            eta_min=args.evaluation.learning_rate_min,
        )

def _store_accuracies(logits, target, split, train_top1,train_top5,val_top1, val_top5):
        """Update the accuracy counters"""
        logits = logits.clone().detach().cpu()
        target = target.clone().detach().cpu()
        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = logits.size(0)

        if split == "train":
            train_top1.update(prec1.data.item(), n)
            train_top5.update(prec5.data.item(), n)
        elif split == "val":
            val_top1.update(prec1.data.item(), n)
            val_top5.update(prec5.data.item(), n)
        else:
            raise ValueError("Unknown split: {}. Expected either 'train' or 'val'")
