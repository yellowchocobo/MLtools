from detectron2.data import transforms as T
from pathlib import Path

import copy
import logging
import imantics
import pandas as pd
import torch
import yaml
import os


from detectron2.utils.logger import setup_logger
from detectron2.config import get_cfg
from detectron2.structures import BoxMode
from detectron2.data import detection_utils as utils
from detectron2.evaluation import COCOEvaluator
from detectron2.data import build_detection_test_loader, build_detection_train_loader
from detectron2.engine import DefaultPredictor, DefaultTrainer, launch
from detectron2.engine.hooks import HookBase
from detectron2.config import configurable
import detectron2.utils.comm as comm
from typing import List, Optional, Union



from skimage.draw import polygon2mask
import numpy as np

import albumentations as A

from detectron2.config.config import CfgNode as CN

from dataclasses import dataclass, field
from typing import Dict
from typing import Any, Union

setup_logger()

def batchpolygon2mask(anno, height, width):
    masks = []
    for i in range(len(anno)):
        segm = anno[i]["segmentation"][0]
        r = np.array(segm).reshape((-1,2))[:,1]
        c = np.array(segm).reshape((-1,2))[:,0]
        # very important to change type otherwise bool make some aug to crash
        mask = polygon2mask((height,width),np.column_stack([r,c])).astype('uint8')
        masks.append(mask)
    return masks

def masks2polygons(masks):
    polygons = []
    idx = []
    for i, mask in enumerate(masks):
        try:
            p = imantics.Mask(mask).polygons()
            # Cannot create a polygon from 4 coordinates, 2 pairs of x,y
            if len(list(p[0])) > 4:
                polygons.append([list(p[0])])
                idx.append(i)
        except:
            None
    return idx, polygons

# need to fix when several of the same name (for example of OneOf)
# fix it the same way as in augments_DataMapper.
def augments(aug_kwargs):
    aug_list = []
    for key in aug_kwargs:
        if key == "OneOf":
            OneOf_list = []
            aug_oneOf = aug_kwargs[key].get("transforms")
            prob_oneOf = {'p':aug_kwargs[key].get("p")}
            OneOf_list.extend([getattr(A, name)(**kwargs) for name, kwargs in aug_oneOf.items()])
            aug_list.extend([A.OneOf(OneOf_list, **prob_oneOf)])
        else:
            kwargs = aug_kwargs[key]
            aug_list.extend([getattr(A, key)(**kwargs)])
    return aug_list

def augments_DataMapper(aug_kwargs):
    aug_list = []
    for key in aug_kwargs:
        if "_" in key:
            kwargs = aug_kwargs[key]
            aug_list.extend([getattr(T, key.split("_")[0])(**kwargs)])
        elif key in ["RandomContrast", "RandomBrightness"]:
            kwargs = aug_kwargs[key]
            aug_list.extend([T.RandomApply(getattr(T, key.split("_")[0])(**kwargs), prob=0.5)])
        else:
            kwargs = aug_kwargs[key]
            aug_list.extend([getattr(T, key)(**kwargs)])
    return aug_list

def save_yaml(filepath: Union[str, Path], content: Any, width: int = 120):
    with open(filepath, "w") as f:
        yaml.dump(content, f, width=width)

def load_yaml(filepath: Union[str, Path]) -> Any:
    with open(filepath, "r") as f:
        content = yaml.full_load(f)
    return content

def training_DatasetMapper(config_file, config_file_complete, augmentation_file):

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        print('__CUDNN VERSION:', torch.backends.cudnn.version())
        print('__Number CUDA Devices:', torch.cuda.device_count())
        print('__CUDA Device Name:', torch.cuda.get_device_name(0))
        print('__CUDA Device Total Memory [GB]:',
              torch.cuda.get_device_properties(0).total_memory / 1e9)

    device = "cuda" if use_cuda else "cpu"
    print("Device: ", device)

    def build_augmentation(cfg, is_train):
        """
        Create a list of default :class:`Augmentation` from config.
        Now it includes resizing and flipping.

        Returns:
            list[Augmentation]
        """
        if is_train:
            augmentation = augments_DataMapper(cfg.aug_kwargs)
        else:
            augmentation = [
                T.NoOpTransform()]  # or should I have the resizing by default?
            # augmentation = [ResizeShortestEdge(short_edge_length=[640, 672, 704, 736, 768, 800], max_size=1333, sample_style='choice')]
        return augmentation

    class DatasetMapper:
        """
        A callable which takes a dataset dict in Detectron2 Dataset format,
        and map it into a format used by the model.

        This is the default callable to be used to map your dataset dict into training data.
        You may need to follow it to implement your own one for customized logic,
        such as a different way to read or transform images.
        See :doc:`/tutorials/data_loading` for details.

        The callable currently does the following:

        1. Read the image from "file_name"
        2. Applies cropping/geometric transforms to the image and annotations
        3. Prepare data and annotations to Tensor and :class:`Instances`
        """

        @configurable
        def __init__(
                self,
                is_train: bool,
                *,
                augmentations: List[Union[T.Augmentation, T.Transform]],
                image_format: str,
                use_instance_mask: bool = False,
                use_keypoint: bool = False,
                instance_mask_format: str = "polygon",
                keypoint_hflip_indices: Optional[np.ndarray] = None,
                precomputed_proposal_topk: Optional[int] = None,
                recompute_boxes: bool = False,
        ):
            """
            NOTE: this interface is experimental.

            Args:
                is_train: whether it's used in training or inference
                augmentations: a list of augmentations or deterministic transforms to apply
                image_format: an image format supported by :func:`detection_utils.read_image`.
                use_instance_mask: whether to process instance segmentation annotations, if available
                use_keypoint: whether to process keypoint annotations if available
                instance_mask_format: one of "polygon" or "bitmask". Process instance segmentation
                    masks into this format.
                keypoint_hflip_indices: see :func:`detection_utils.create_keypoint_hflip_indices`
                precomputed_proposal_topk: if given, will load pre-computed
                    proposals from dataset_dict and keep the top k proposals for each image.
                recompute_boxes: whether to overwrite bounding box annotations
                    by computing tight bounding boxes from instance mask annotations.
            """
            if recompute_boxes:
                assert use_instance_mask, "recompute_boxes requires instance masks"
            # fmt: off
            self.is_train = is_train
            self.augmentations = T.AugmentationList(augmentations)
            self.image_format = image_format
            self.use_instance_mask = use_instance_mask
            self.instance_mask_format = instance_mask_format
            self.use_keypoint = use_keypoint
            self.keypoint_hflip_indices = keypoint_hflip_indices
            self.proposal_topk = precomputed_proposal_topk
            self.recompute_boxes = recompute_boxes
            # fmt: on
            logger = logging.getLogger(__name__)
            mode = "training" if is_train else "inference"
            logger.info(
                f"[DatasetMapper] Augmentations used in {mode}: {augmentations}")

        @classmethod
        def from_config(cls, cfg, is_train: bool = True):
            augs = build_augmentation(cfg, is_train)
            recompute_boxes = cfg.MODEL.MASK_ON  # I think this is good anyway, to recompute boxes!
            ret = {
                "is_train": is_train,
                "augmentations": augs,
                "image_format": cfg.INPUT.FORMAT,
                "use_instance_mask": cfg.MODEL.MASK_ON,
                "instance_mask_format": cfg.INPUT.MASK_FORMAT,
                "use_keypoint": cfg.MODEL.KEYPOINT_ON,
                "recompute_boxes": recompute_boxes,
            }

            if cfg.MODEL.KEYPOINT_ON:
                ret[
                    "keypoint_hflip_indices"] = utils.create_keypoint_hflip_indices(
                    cfg.DATASETS.TRAIN)

            if cfg.MODEL.LOAD_PROPOSALS:
                ret["precomputed_proposal_topk"] = (
                    cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TRAIN
                    if is_train
                    else cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TEST
                )
            return ret

        def _transform_annotations(self, dataset_dict, transforms, image_shape):
            # USER: Modify this if you want to keep them for some reason.
            for anno in dataset_dict["annotations"]:
                if not self.use_instance_mask:
                    anno.pop("segmentation", None)
                if not self.use_keypoint:
                    anno.pop("keypoints", None)

            # USER: Implement additional transformations if you have other types of data
            annos = [
                utils.transform_instance_annotations(
                    obj, transforms, image_shape,
                    keypoint_hflip_indices=self.keypoint_hflip_indices
                )
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            instances = utils.annotations_to_instances(
                annos, image_shape, mask_format=self.instance_mask_format
            )

            # After transforms such as cropping are applied, the bounding box may no longer
            # tightly bound the object. As an example, imagine a triangle object
            # [(0,0), (2,0), (0,2)] cropped by a box [(1,0),(2,2)] (XYXY format). The tight
            # bounding box of the cropped triangle should be [(1,0),(2,1)], which is not equal to
            # the intersection of original bounding box and the cropping box.
            if self.recompute_boxes:
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            dataset_dict["instances"] = utils.filter_empty_instances(instances)

        def __call__(self, dataset_dict):
            """
            Args:
                dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

            Returns:
                dict: a format that builtin models in detectron2 accept
            """
            dataset_dict = copy.deepcopy(
                dataset_dict)  # it will be modified by code below
            # USER: Write your own image loading if it's not from a file
            image = utils.read_image(dataset_dict["file_name"],
                                     format=self.image_format)
            utils.check_image_size(dataset_dict, image)

            # USER: Remove if you don't do semantic/panoptic segmentation.
            if "sem_seg_file_name" in dataset_dict:
                sem_seg_gt = utils.read_image(
                    dataset_dict.pop("sem_seg_file_name"), "L").squeeze(2)
            else:
                sem_seg_gt = None

            aug_input = T.AugInput(image, sem_seg=sem_seg_gt)
            transforms = self.augmentations(aug_input)
            image, sem_seg_gt = aug_input.image, aug_input.sem_seg

            image_shape = image.shape[:2]  # h, w
            # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
            # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
            # Therefore it's important to use torch.Tensor.
            dataset_dict["image"] = torch.as_tensor(
                np.ascontiguousarray(image.transpose(2, 0, 1)))
            if sem_seg_gt is not None:
                dataset_dict["sem_seg"] = torch.as_tensor(
                    sem_seg_gt.astype("long"))

            # USER: Remove if you don't use pre-computed proposals.
            # Most users would not need this feature.
            if self.proposal_topk is not None:
                utils.transform_proposals(
                    dataset_dict, image_shape, transforms,
                    proposal_topk=self.proposal_topk
                )

            # need to be commented for calculating validation loss (because annotations are needed)
            # if not self.is_train:
            #    # USER: Modify this if you want to keep them for some reason.
            #    dataset_dict.pop("annotations", None)
            #    dataset_dict.pop("sem_seg_file_name", None)
            #    return dataset_dict

            if "annotations" in dataset_dict:
                self._transform_annotations(dataset_dict, transforms,
                                            image_shape)

            return dataset_dict

    class MyTrainer(DefaultTrainer):
        @classmethod
        def build_train_loader(cls, cfg, is_train=True, sampler=None):
            return build_detection_train_loader(
                cfg, mapper=DatasetMapper(cfg, is_train), sampler=sampler
            )

        @classmethod
        def build_test_loader(cls, cfg, dataset_name):
            return build_detection_test_loader(
                cfg, dataset_name, mapper=DatasetMapper(cfg, False)
            )

        @classmethod
        def build_evaluator(cls, cfg, dataset_name):
            output_folder = os.path.join(cfg.OUTPUT_DIR, "coco_eval")
            os.makedirs(output_folder, exist_ok=True)
            return COCOEvaluator(dataset_name, ("segm",), False,
                                 output_folder, max_dets_per_image=1000)

    @dataclass
    class BOULDERconfig:
        # augmentations
        aug_kwargs: Dict = field(default_factory=lambda: {})

        def update(self, param_dict: Dict) -> "BOULDERconfig":
            # Overwrite by `param_dict`
            for key, value in param_dict.items():
                if not hasattr(self, key):
                    raise ValueError(
                        f"[ERROR] Unexpected key for flag = {key}")
                setattr(self, key, value)
            return self

    class ValidationLoss(HookBase):
        def __init__(self, cfg):
            super().__init__()
            self.cfg = cfg.clone()
            self.cfg.DATASETS.TRAIN = cfg.DATASETS.TEST[0]
            self._loader = iter(
                MyTrainer.build_detection_train_loader(self.cfg,
                                                       is_train=False))  # False, for not applying any transforms

        def after_step(self):
            data = next(self._loader)
            with torch.no_grad():
                loss_dict = self.trainer.model(data)

                losses = sum(loss_dict.values())
                assert torch.isfinite(losses).all(), loss_dict

                loss_dict_reduced = {"val_" + k: v.item() for k, v in
                                     comm.reduce_dict(loss_dict).items()}
                losses_reduced = sum(
                    loss for loss in loss_dict_reduced.values())
                if comm.is_main_process():
                    self.trainer.storage.put_scalars(
                        total_val_loss=losses_reduced,
                        **loss_dict_reduced)

    # read augmentations
    augmentations_dict = load_yaml(augmentation_file)
    flags = BOULDERconfig().update(augmentations_dict)
    cfg = get_cfg()
    cfg.merge_from_file(config_file)

    # Save complete config file
    with open(config_file_complete, "w") as f:
      f.write(cfg.dump())

    cfg.aug_kwargs = CN(flags.aug_kwargs)  # pass aug_kwargs to cfg
    cfg.MODEL.DEVICE = device

    # training
    trainer = MyTrainer(cfg)
    val_loss = ValidationLoss(cfg)
    trainer.register_hooks([val_loss])
    # swap the order of PeriodicWriter and ValidationLoss
    trainer._hooks = trainer._hooks[:-2] + trainer._hooks[-2:][::-1]
    trainer.resume_or_load(resume=False)
    trainer.train()


def training_AlbumentMapper(config_file, config_file_complete, augmentation_file):

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        print('__CUDNN VERSION:', torch.backends.cudnn.version())
        print('__Number CUDA Devices:', torch.cuda.device_count())
        print('__CUDA Device Name:', torch.cuda.get_device_name(0))
        print('__CUDA Device Total Memory [GB]:',
              torch.cuda.get_device_properties(0).total_memory / 1e9)

    device = "cuda" if use_cuda else "cpu"
    print("Device: ", device)

    class AlbumentationsMapper:
        """Mapper which uses `albumentations` augmentations"""

        def __init__(self, cfg, is_train: bool = True):
            aug_kwargs = cfg.aug_kwargs
            if is_train:
                aug_list = augments(aug_kwargs)
            else:
                aug_list = []
                # else, it gives an empty list, which is equivalent to NoOp
            self.transform = A.Compose(aug_list)
            self.is_train = is_train

            mode = "training" if is_train else "inference"
            print(
                f"[AlbumentationsMapper] Augmentations used in {mode}: {self.transform}")

        def __call__(self, dataset_dict):
            dataset_dict = copy.deepcopy(
                dataset_dict)  # it will be modified by code below
            image = utils.read_image(dataset_dict["file_name"], format="BGR")
            masks = batchpolygon2mask(dataset_dict["annotations"],
                                      dataset_dict["height"],
                                      dataset_dict["width"])

            # I could change it when it is is_train False --> to do nothing

            transformed = self.transform(image=image, masks=masks)
            transformed_image = transformed['image']
            transformed_masks = transformed['masks']

            # create empty dataframe
            df = pd.DataFrame()

            # generate new bounding box (could have used the self.transform but I prefer this way)
            transformed_bbox = []
            idx = []
            for i, mask in enumerate(transformed_masks):
                if np.nonzero(mask)[0].shape[0] == 0:
                    None
                else:
                    pos = np.nonzero(mask)
                    xmin = np.min(pos[1])
                    xmax = np.max(pos[1])
                    ymin = np.min(pos[0])
                    ymax = np.max(pos[0])
                    transformed_bbox.append(
                        [xmin, ymin, xmax, ymax])  # XYXY_ABS
                    idx.append(i)

            # equivalent to utils.filter_empty_instances
            transformed_masks = [transformed_masks[i] for i in idx]
            idx, polygons = masks2polygons(
                transformed_masks)  # some masks do not have enough coordinates to create a polygon
            transformed_bbox = [transformed_bbox[i] for i in idx]

            # Updating annotation
            df["iscrowd"] = [0] * len(transformed_bbox)
            df["bbox"] = transformed_bbox
            df["category_id"] = 0
            df["segmentation"] = polygons
            df["bbox_mode"] = BoxMode.XYXY_ABS
            annos = df.to_dict(orient='records')

            image_shape = transformed_image.shape[:2]  # h, w
            dataset_dict["image"] = torch.as_tensor(
                transformed_image.transpose(2, 0, 1).astype("float32"))
            instances = utils.annotations_to_instances(annos, image_shape,
                                                       mask_format="polygon")  # needs to be there
            dataset_dict[
                "instances"] = instances  # utils.filter_empty_instances(instances) #  --> this is done already above
            return dataset_dict

    class MyTrainer(DefaultTrainer):
        @classmethod
        def build_train_loader(cls, cfg, is_train=True, sampler=None):
            return build_detection_train_loader(
                cfg, mapper=AlbumentationsMapper(cfg, is_train), sampler=sampler
            )

        @classmethod
        def build_test_loader(cls, cfg, dataset_name):
            return build_detection_test_loader(
                cfg, dataset_name, mapper=AlbumentationsMapper(cfg, False)
            )

        @classmethod
        def build_evaluator(cls, cfg, dataset_name):
            output_folder = os.path.join(cfg.OUTPUT_DIR, "coco_eval")
            os.makedirs(output_folder, exist_ok=True)
            return COCOEvaluator(dataset_name, ("segm",), False, output_folder,
                                 max_dets_per_image=1000)

    class ValidationLoss(HookBase):
        def __init__(self, cfg):
            super().__init__()
            self.cfg = cfg.clone()
            self.cfg.DATASETS.TRAIN = cfg.DATASETS.TEST[0]
            self._loader = iter(MyTrainer.build_detection_train_loader(self.cfg,
                                                                       is_train=False))  # False, for not applying any transforms

        def after_step(self):
            data = next(self._loader)
            with torch.no_grad():
                loss_dict = self.trainer.model(data)

                losses = sum(loss_dict.values())
                assert torch.isfinite(losses).all(), loss_dict

                loss_dict_reduced = {"val_" + k: v.item() for k, v in
                                     comm.reduce_dict(loss_dict).items()}
                losses_reduced = sum(
                    loss for loss in loss_dict_reduced.values())
                if comm.is_main_process():
                    self.trainer.storage.put_scalars(
                        total_val_loss=losses_reduced,
                        **loss_dict_reduced)

    @dataclass
    class BOULDERconfig:
        # augmentations
        aug_kwargs: Dict = field(default_factory=lambda: {})

        def update(self, param_dict: Dict) -> "BOULDERconfig":
            # Overwrite by `param_dict`
            for key, value in param_dict.items():
                if not hasattr(self, key):
                    raise ValueError(f"[ERROR] Unexpected key for flag = {key}")
                setattr(self, key, value)
            return self

    # read augmentations
    augmentations_dict = load_yaml(augmentation_file)
    flags = BOULDERconfig().update(augmentations_dict)
    cfg = get_cfg()
    cfg.merge_from_file(config_file)

    # Save complete config file
    with open(config_file_complete, "w") as f:
        f.write(cfg.dump())

    cfg.aug_kwargs = CN(flags.aug_kwargs)  # pass aug_kwargs to cfg
    cfg.MODEL.DEVICE = device

    # training
    trainer = MyTrainer(cfg)
    val_loss = ValidationLoss(cfg)
    trainer.register_hooks([val_loss])
    # swap the order of PeriodicWriter and ValidationLoss
    trainer._hooks = trainer._hooks[:-2] + trainer._hooks[-2:][::-1]
    trainer.resume_or_load(resume=False)
    trainer.train()