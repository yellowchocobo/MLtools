import glob
import json
import os
import numpy as np
import sys

sys.path.append("/home/nilscp/GIT/")

from detectron2.structures import BoxMode
from pathlib import Path
from PIL import Image

class NpEncoder(json.JSONEncoder):
    """
    Convert numpy integer/floating and ++ to python integer, float and ++
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def instance2semantic(mask, color):

    """
    This function convert instance segmentation labels to semantic segmentation
    labels based on the specified color.

    NB! There are lot of assumptions in this function:
    - The mask is one band
    - the color is given as tuple (e.g, (255,0,0) for yellow...

    Need to be generalized for multiple inputs, colors, mask dimension and ++

    :param mask:
    :param color:
    :return:
    """
    mask = Path(mask)
    array = Image.open(mask)
    array = np.array(array)
    height, width = array.shape
    new_array = np.zeros((height, width, 3)).astype('uint8')

    # mask for values larger or equal to 1
    mask_array = array >= 1

    for i, c in enumerate(color):
        new_array[:, :, i][mask_array] = c

    output_filename = mask.with_name(mask.stem + "_semantic" + mask.suffix)
    new_image = Image.fromarray(new_array)
    new_image.save(output_filename)

def get_boxes_from_masks(masks):
    """ Helper, gets bounding boxes from masks. They seem a bit off, not sure why.. """
    bboxes = []
    updated_masks = []
    for i, mask in enumerate(masks):
        pos = np.nonzero(mask)
        xmin = np.min(pos[1])  # for some reasons the bbox seems a bit off...
        xmax = np.max(pos[1])
        ymin = np.min(pos[0])
        ymax = np.max(pos[0])

        # let's extract the coordinates as expected in COCO format
        px = pos[1]
        py = pos[0]
        mask_coord = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
        mask_coord = [p for x in mask_coord for p in x]

        # few entries have only one pixel
        # let's avoid those entries
        if ((xmin == xmax) or (ymin == ymax)):
            None
        else:
            bboxes.append([xmin, ymin, xmax, ymax])
            updated_masks.append(
                mask_coord)  # only add masks that have a correct bounding box

    num_masks = len(bboxes)

    return bboxes, updated_masks, num_masks

#TODO: maybe add imgs and masks as variables
def toCOCO(root):

    try:
        root = root.as_posix()
    except:
        None

    imgs_f = list(sorted(glob.glob(os.path.join(root, "images") + "/*image.png")))
    masks_f = list(sorted(glob.glob(os.path.join(root, "masks") + "/*.png")))

    n = len(imgs_f)

    dataset_dicts = []

    for i in range(n):

        record = {}

        img_path = Path(imgs_f[i])
        mask_path = Path(masks_f[i])

        # reading image
        image = Image.open(img_path)
        image = np.array(image)
        (height, width) = image.shape

        # reading mask
        mask = Image.open(mask_path)
        mask = np.array(mask).astype('uint16')  # (height,width)

        # instances are encoded as different colors
        obj_ids = np.unique(mask)

        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks

        masks = []
        for obj_id in obj_ids:
            masks.append((mask == obj_id) + 0)

        bboxes, updated_masks, num_masks = get_boxes_from_masks(masks)

        record["file_name"] = img_path.with_name(
            img_path.stem + "_fakergb" + img_path.suffix).as_posix()
        record["image_id"] = i
        record["height"] = height
        record["width"] = width

        objs = []
        for nm in range(num_masks):
            obj = {"bbox": bboxes[nm], "bbox_mode": BoxMode.XYXY_ABS,
                   "segmentation": [updated_masks[nm]], "category_id": 0}
            objs.append(obj)

        record["annotations"] = objs
        dataset_dicts.append(record)

    return (dataset_dicts)