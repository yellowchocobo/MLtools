import os
import sys
sys.path.append("/home/nilscp/GIT/")
sys.path.append("/home/nilscp/GIT/MLtools/projects")


from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import build_detection_train_loader, build_detection_test_loader
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from bouldering import (add_config, AlbumentMapper_polygon, AlbumentMapper_bitmask,
                        BoulderEvaluator, build_optimizer_sgd, build_optimizer_adam,
                        build_lr_scheduler)

class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains a number pre-defined logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can use the cleaner
    "SimpleTrainer", or write your own training loop.
    """

    @classmethod
    def build_train_loader(cls, cfg, is_train=True, sampler=None):
        if cfg.INPUT.MASK_FORMAT == "polygon":
            mapper = AlbumentMapper_polygon(cfg, is_train)
        elif cfg.INPUT.MASK_FORMAT == "bitmask":
            mapper = AlbumentMapper_bitmask(cfg, is_train)
        else:
            None
        return build_detection_train_loader(cfg, mapper=mapper, sampler=sampler)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        if cfg.INPUT.MASK_FORMAT == "polygon":
            mapper = AlbumentMapper_polygon(cfg, False)
        elif cfg.INPUT.MASK_FORMAT == "bitmask":
            mapper = AlbumentMapper_bitmask(cfg, False)
        else:
            None
        return build_detection_test_loader(
            cfg, dataset_name, mapper=mapper)


    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        output_folder = os.path.join(cfg.OUTPUT_DIR, "coco_eval")
        os.makedirs(output_folder, exist_ok=True)
        return BoulderEvaluator(dataset_name, ("segm",), False, output_folder,
                                max_dets_per_image=1000)

    @classmethod
    def build_optimizer(cls, cfg, model):
        if cfg.SOLVER.OPTIMIZER == "SGD":
            opt = build_optimizer_sgd(cfg, model)
        elif cfg.SOLVER.OPTIMIZER == "ADAM":
            opt = build_optimizer_adam(cfg, model)
        return opt

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        """
        It now calls :func:`detectron2.solver.build_lr_scheduler`.
        Overwrite it if you'd like a different scheduler.
        """
        return build_lr_scheduler(cfg, optimizer)


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_config(cfg, args.aug_path, args.min_area_npixels, args.optimizer_n)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )