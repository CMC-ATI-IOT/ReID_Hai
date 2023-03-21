# train_step -> run_step
# epoch_step -> train
# load_data -> build_train_loader, test_train_loader

import sys, os
sys.path.append('.')
lib_path = os.path.abspath(os.path.join('/home/tuantran/AI_TEAM/REID_HAI/models'))
sys.path.append(lib_path)
from optimizer import build_optimizer
from scheduler import build_lr_scheduler

import wandb

from fastreid.config import get_cfg
from fastreid.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from fastreid.utils.checkpoint import Checkpointer

import argparse
import logging
from collections import OrderedDict

import torch
from torch.nn.parallel import DistributedDataParallel

from fastreid.data import build_reid_test_loader, build_reid_train_loader
from fastreid.evaluation import (ReidEvaluator,
                                 inference_on_dataset, print_csv_format)
from fastreid.modeling.meta_arch import build_model
from fastreid.utils import comm
from fastreid.utils.checkpoint import Checkpointer
from fastreid.utils.collect_env import collect_env_info
from fastreid.utils.env import seed_all_rng
from fastreid.utils.events import CommonMetricPrinter, JSONWriter, TensorboardXWriter
from fastreid.utils.file_io import PathManager
from fastreid.utils.logger import setup_logger
from fastreid.engine import hooks
from fastreid.engine.hooks import CallbackHook, IterationTimer, PeriodicWriter, PeriodicCheckpointer, LRScheduler, AutogradProfiler, EvalHook, PreciseBN, LayerFreeze

from fastreid.engine.train_loop import TrainerBase, AMPTrainer, SimpleTrainer

from fastreid.engine import default_argument_parser, default_setup, launch


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


class Train_Pipeline(TrainerBase):
    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode):
        """
        super().__init__()
        logger = logging.getLogger("fastreid")
        if not logger.isEnabledFor(logging.INFO):  # setup_logger is not called for fastreid
            setup_logger()
        
        # setting config to log on wandb
        if args.eval_only:
            configs = {
                "batch_size": cfg.TEST.IMS_PER_BATCH,

            }
        else:
            configs = {
                "epochs": cfg.SOLVER.MAX_EPOCH,
                "learning_rate_init": cfg.SOLVER.BASE_LR,
                "batch_size": cfg.SOLVER.IMS_PER_BATCH,
                "backbone": "RepVGG_B3g4",
            }

        wandb.init(project = "ReID_Hai", entity = "ai-iot", config=configs, name = "RepVGG_B3g4_8data 1")

        # Assume these objects must be constructed in this order.
        data_loader = self.build_train_loader(cfg)
        cfg = self.auto_scale_hyperparams(cfg, data_loader.dataset.num_classes)
        model = self.build_model(cfg)
        optimizer, param_wrapper = self.build_optimizer(cfg, model)

        # For training, wrap with DDP. But don't need this for inference.
        if comm.get_world_size() > 1:
            # ref to https://github.com/pytorch/pytorch/issues/22049 to set `find_unused_parameters=True`
            # for part of the parameters is not updated.
            model = DistributedDataParallel(
                model, device_ids=[comm.get_local_rank()], broadcast_buffers=False,
            )

        self._trainer = (AMPTrainer if cfg.SOLVER.AMP.ENABLED else SimpleTrainer)(
            model, data_loader, optimizer, param_wrapper
        )

        self.iters_per_epoch = len(data_loader.dataset) // cfg.SOLVER.IMS_PER_BATCH
        self.scheduler = self.build_lr_scheduler(cfg, optimizer, self.iters_per_epoch)

        # Assume no other objects need to be checkpointed.
        # We can later make it checkpoint the stateful hooks
        self.checkpointer = Checkpointer(
            # Assume you want to save checkpoints together with logs/statistics
            model,
            cfg.OUTPUT_DIR,
            save_to_disk=comm.is_main_process(),
            optimizer=optimizer,
            **self.scheduler,
        )

        self.start_epoch = 0
        self.max_epoch = cfg.SOLVER.MAX_EPOCH
        self.max_iter = self.max_epoch * self.iters_per_epoch
        self.warmup_iters = cfg.SOLVER.WARMUP_ITERS
        self.delay_epochs = cfg.SOLVER.DELAY_EPOCHS
        self.cfg = cfg

        self.register_hooks(self.build_hooks())

    def resume_or_load(self, resume=True):
        """
        If `resume==True` and `cfg.OUTPUT_DIR` contains the last checkpoint (defined by
        a `last_checkpoint` file), resume from the file. Resuming means loading all
        available states (eg. optimizer and scheduler) and update iteration counter
        from the checkpoint. ``cfg.MODEL.WEIGHTS`` will not be used.
        Otherwise, this is considered as an independent training. The method will load model
        weights from the file `cfg.MODEL.WEIGHTS` (but will not load other states) and start
        from iteration 0.
        Args:
            resume (bool): whether to do resume or not
        """
        # The checkpoint stores the training iteration that just finished, thus we start
        # at the next iteration (or iter zero if there's no checkpoint).
        checkpoint = self.checkpointer.resume_or_load(self.cfg.MODEL.WEIGHTS, resume=resume)

        if resume and self.checkpointer.has_checkpoint():
            self.start_epoch = checkpoint.get("epoch", -1) + 1
            # The checkpoint stores the training iteration that just finished, thus we start
            # at the next iteration (or iter zero if there's no checkpoint).

    def build_hooks(self):
        """
        Build a list of default hooks, including timing, evaluation,
        checkpointing, lr scheduling, precise BN, writing events.
        Returns:
            list[HookBase]:
        """
        logger = logging.getLogger(__name__)
        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = 0  # save some memory and time for PreciseBN
        cfg.DATASETS.NAMES = tuple([cfg.TEST.PRECISE_BN.DATASET])  # set dataset name for PreciseBN

        ret = [
            hooks.IterationTimer(),
            hooks.LRScheduler(self.optimizer, self.scheduler),
        ]

        if cfg.TEST.PRECISE_BN.ENABLED and hooks.get_bn_modules(self.model):
            logger.info("Prepare precise BN dataset")
            ret.append(hooks.PreciseBN(
                # Run at the same freq as (but before) evaluation.
                self.model,
                # Build a new data loader to not affect training
                self.build_train_loader(cfg),
                cfg.TEST.PRECISE_BN.NUM_ITER,
            ))

        if len(cfg.MODEL.FREEZE_LAYERS) > 0 and cfg.SOLVER.FREEZE_ITERS > 0:
            ret.append(hooks.LayerFreeze(
                self.model,
                cfg.MODEL.FREEZE_LAYERS,
                cfg.SOLVER.FREEZE_ITERS,
            ))

        # Do PreciseBN before checkpointer, because it updates the model and need to
        # be saved by checkpointer.
        # This is not always the best: if checkpointing has a different frequency,
        # some checkpoints may have more precise statistics than others.

        def test_and_save_results():
            self._last_eval_results = self.test(self.cfg, self.model)
            results = self._last_eval_results
            return self._last_eval_results

        # Do evaluation before checkpointer, because then if it fails,
        # we can use the saved checkpoint to debug.
        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_results))

        if comm.is_main_process():
            ret.append(hooks.PeriodicCheckpointer(self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD))
            # run writers in the end, so that evaluation metrics are written
            ret.append(hooks.PeriodicWriter(self.build_writers(), 200))

        return ret

    def build_writers(self): 
        # Assume the default print/log frequency.
        return [
            # It may not always print what you want to see, since it prints "common" metrics only.
            CommonMetricPrinter(self.max_iter),
            JSONWriter(os.path.join(self.cfg.OUTPUT_DIR, "metrics.json")),
            TensorboardXWriter(self.cfg.OUTPUT_DIR),
        ]

    @classmethod
    def build_model(cls, cfg):
        model = build_model(cfg)
        logger = logging.getLogger(__name__)
        logger.info("Model:\n{}".format(model))
        return model

    @classmethod
    def build_optimizer(cls, cfg, model):
        return build_optimizer(cfg, model)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer, iters_per_epoch):
        return build_lr_scheduler(cfg, optimizer, iters_per_epoch)

    @classmethod
    def build_train_loader(cls, cfg):
        logger = logging.getLogger(__name__)
        logger.info("Prepare training set")
        return build_reid_train_loader(cfg, combineall=cfg.DATASETS.COMBINEALL)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        return build_reid_test_loader(cfg, dataset_name=dataset_name)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_dir=None):
        data_loader, num_query = cls.build_test_loader(cfg, dataset_name)
        return data_loader, ReidEvaluator(cfg, num_query, output_dir)

    def train(self):
        super().train(self.start_epoch, self.max_epoch, self.iters_per_epoch)
        if comm.is_main_process():
            assert hasattr(
                self, "_last_eval_results"
            ), "No evaluation results obtained during training!"
           
            return self._last_eval_results # = self.test(self.cfg, self.model)

    def run_step(self): # AMPTrainer or Simple Trainer
        self._trainer.iter = self.iter 
        self._trainer.run_step()
        
        # Loss xem tại fastreid.engine.train_loop, file config bagtricks
    
    @classmethod
    def test(cls, cfg, model):
        # dict: a dict of result metrics
        logger = logging.getLogger(__name__)
        results = OrderedDict()
        
        log_dict = {}
        
        for idx, dataset_name in enumerate(cfg.DATASETS.TESTS):
            logger.info("Prepare testing set")
            try:
                data_loader, evaluator = cls.build_evaluator(cfg, dataset_name)
            except NotImplementedError:
                logger.warn(
                    "No evaluator found. implement its `build_evaluator` method."
                )
                results[dataset_name] = {}
                continue
            results_i = inference_on_dataset(model, data_loader, evaluator, flip_test=cfg.TEST.FLIP.ENABLED)
            results[dataset_name] = results_i

            if comm.is_main_process():
                assert isinstance(
                    results, dict
                ), "Evaluator must return a dict on the main process. Got {} instead.".format(
                    results
                )
                logger.info("Evaluation results for {} in csv format:".format(dataset_name))
                results_i['dataset'] = dataset_name
                print_csv_format(results_i)
                
                # rank_1_name = "Rank-1 - " + dataset_name
                # mAP_name = "mAP - " + dataset_name
                log_dict["Rank-1 - " + dataset_name] = results_i['Rank-1']
                log_dict["mAP - " + dataset_name] = results_i['mAP']
                
                print("Rank-1 - " + dataset_name + ": ", results_i['Rank-1'])
                print("mAP - " + dataset_name + ": ", results_i['mAP'])

        wandb.log(log_dict)
        
        if len(results) == 1:
            results = list(results.values())[0]

        return results

    @staticmethod
    def auto_scale_hyperparams(cfg, num_classes):
        r"""
        This is used for auto-computation actual training iterations,
        because some hyper-param, such as MAX_ITER, means training epochs rather than iters,
        so we need to convert specific hyper-param to training iterations.
        """
        cfg = cfg.clone()
        frozen = cfg.is_frozen()
        cfg.defrost()

        # If you don't hard-code the number of classes, it will compute the number automatically
        if cfg.MODEL.HEADS.NUM_CLASSES == 0:
            output_dir = cfg.OUTPUT_DIR
            cfg.MODEL.HEADS.NUM_CLASSES = num_classes
            logger = logging.getLogger(__name__)
            logger.info(f"Auto-scaling the num_classes={cfg.MODEL.HEADS.NUM_CLASSES}")

            # Update the saved config file to make the number of classes valid
            if comm.is_main_process() and output_dir:
                # Note: some of our scripts may expect the existence of
                # config.yaml in output directory
                path = os.path.join(output_dir, "config.yaml")
                with PathManager.open(path, "w") as f:
                    f.write(cfg.dump())

        if frozen: cfg.freeze()

        return cfg
    
# Access basic attributes from the underlying trainer
for _attr in ["model", "data_loader", "optimizer", "grad_scaler"]:
    setattr(Train_Pipeline, _attr, property(lambda self, x=_attr: getattr(self._trainer, x, None)))


def main(args):
    cfg = setup(args)

    if args.eval_only:
        cfg.defrost()
        cfg.MODEL.BACKBONE.PRETRAIN = False
        model = Train_Pipeline.build_model(cfg)

        Checkpointer(model).load(cfg.MODEL.WEIGHTS)  # load trained model

        res = Train_Pipeline.test(cfg, model)
        return res

    trainer = Train_Pipeline(cfg)

    trainer.resume_or_load(resume=args.resume)
    return trainer.train()
    

if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)

    # Chạy trên nhiều GPU trên nhiều máy (nếu có)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )