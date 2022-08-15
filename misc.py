

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

def get_boxes_from_masks_XYXY(masks):
    """ Helper, gets bounding boxes from masks. They seem a bit off, not sure why.. """
    bboxes = []
    updated_masks = []
    for i, mask in enumerate(masks):
        pos = np.nonzero(mask)
        xmin = np.min(pos[1])  # for some reasons the bbox seems a bit off...
        xmax = np.max(pos[1])
        ymin = np.min(pos[0])
        ymax = np.max(pos[0])

        # let's extract the coordinates as expected in COCO format (the problem is that it cannot be converted back to a shapefile, because of the orders of tge
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

def toCOCO_datasetcatalog(root):

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

        bboxes, updated_masks, num_masks = get_boxes_from_masks_XYXY(masks)

        #record["file_name"] = img_path.name # use relative path
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


def toCOCO_register_instances(root):

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

        bboxes, updated_masks, num_masks = get_boxes_from_masks_XYXY(masks)

        record["file_name"] = img_path.as_posix() #using one band png (full path)
        #record["file_name"] = img_path.with_name(
        #    img_path.stem + "_fakergb" + img_path.suffix).as_posix()
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

# TODO figure out where to save the data....
def tiling_raster_from_dataframe(df, output_folder, block_width, block_height):
    output_folder = Path(output_folder)
    output_folder_tif = (output_folder / "tif")
    output_folder_png1 = (output_folder / "png-1band")
    output_folder_png3 = (output_folder / "png-3band")
    output_folder_tif.mkdir(parents=True, exist_ok=True)
    output_folder_png1.mkdir(parents=True, exist_ok=True)
    output_folder_png3.mkdir(parents=True, exist_ok=True)

    for index, row in tqdm(df.iterrows(), total=df.shape[0]):

        # this is only useful within the loop if generating tiling on multiple images
        in_raster = row.raster_ap
        src_profile = raster.get_raster_profile(in_raster)
        win_profile = src_profile
        win_profile["width"] = block_width
        win_profile["height"] = block_height

        arr = raster.read_raster(in_raster=in_raster,
                                 bbox=rio.windows.Window(*row.rwindows))

        # edge cases (in the East, and South, the extent can be beigger than the actual raster)
        # read_raster will then return an array with not the dimension
        h, w = arr.squeeze().shape

        if (h, w) != (block_height, block_width):
            arr = np.pad(arr.squeeze(),
                         [(0, block_height - h), (0, block_width - w)],
                         mode='constant', constant_values=0)
            arr = np.expand_dims(arr, axis=0)

        filename_tif = (
                    output_folder_tif / row.file_name.replace(".png", ".tif"))
        filename_png1 = (output_folder_tif / row.file_name)
        filename_png3 = (output_folder_tif / row.NAC_id + "_image_3band.png")
        win_profile["transform"] = Affine(*row["transform"])

        raster.save_raster(filename_tif, arr, win_profile, is_image=False)  # generate tif
        raster.tiff_to_png(filename_tif, filename_png1)  # generate png (1-band) # tif to png
        raster.fake_RGB(filename_png1, filename_png3)  # generate png (3-band) # png to fakepng


def split_global(df_bbox, split):
    """
    Global shuffling
    """

    np.random.seed(seed=27)
    n = df_bbox.shape[0]
    idx_shuffle = np.random.permutation(n)

    training_idx, remaining_idx = np.split(idx_shuffle,
                                           [int(split[0] * len(idx_shuffle))])
    split_val = split[1] / (1 - split[0])  # split compare to remaining data
    val_idx, test_idx = np.split(remaining_idx,
                                 [int(split_val * len(remaining_idx))])

    df_bbox["dataset"] = "train"
    df_bbox["dataset"].iloc[val_idx] = "validation"
    df_bbox["dataset"].iloc[test_idx] = "test"

    return (df_bbox)


def folder_structure(dataset_directory):
    dataset_directory = Path(dataset_directory)
    folders = ["train", "validation", "test"]
    sub_folders = ["images", "labels"]

    for f in folders:
        for s in sub_folders:
            new_folder = dataset_directory / f / s
            Path(new_folder).mkdir(parents=True, exist_ok=True)