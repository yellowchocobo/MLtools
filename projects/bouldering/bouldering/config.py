from detectron2.config import CfgNode as CN

__all__ = ["add_config"]

def add_config(cfg, aug_path, min_area_npixels, optimizer_n):
    """
    Add config for Panoptic-DeepLab.
    """

    # Optimizer parameters (ADAM, ADAMW, SGD)
    cfg.SOLVER.OPTIMIZER = optimizer_n

    # Only use for CyclicLR scheduler (hardcoded).
    # Based on 753 images, and image batch of 4.
    cfg.SOLVER.MAX_LR = 0.1
    cfg.SOLVER.STEP_SIZE_UP = 2000

    # thresholding mask (I don't have any thresholding possibility for polygons)
    cfg.INPUT.MIN_AREA_NPIXELS = min_area_npixels

    # augmentations
    cfg.MODEL.AUGMENTATIONS = CN()
    cfg.MODEL.AUGMENTATIONS.PATH = aug_path