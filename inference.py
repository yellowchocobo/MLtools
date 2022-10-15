import geopandas as gpd
import numpy as np
import rasterio as rio
import pandas as pd
import torch
import sys

from pathlib import Path
from PIL import Image
from rasterio import features
from shapely.geometry import box
from tqdm import tqdm

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor

sys.path.append("/home/nilscp/GIT/rastertools")
import raster

def predict(config_file, model_weights, device, image_dir, out_shapefile,
            search_pattern="*.tif", scores=0.5):
    # load model and weight of the models
    cfg = get_cfg()
    cfg.merge_from_file(config_file)
    cfg.MODEL.WEIGHTS = model_weights.as_posix()
    cfg.MODEL.DEVICE = device
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = scores

    # predictor for one image
    predictor = DefaultPredictor(cfg)

    image_dir = Path(image_dir)
    geoms = []
    scores_list = []
    boulder_id = []

    bid = 0
    iid = 0

    for tif in tqdm(sorted(image_dir.glob(search_pattern))):
        #print(tif.name)
        png = tif.with_name(tif.name.replace('tif', 'png'))

        with rio.open(tif) as src:
            meta = src.meta

            # loading image
            array = Image.open(png)
            array = np.array(array)
            array = np.expand_dims(array, 2)

            # inference
            outputs = predictor(array)

            for i, pred in enumerate(outputs["instances"].pred_masks.to("cpu")):
                pred_mask = torch.Tensor.numpy(pred)
                pred_mask = (pred_mask + 0.0).astype('uint8')
                results = (
                    {'properties': {'raster_val': v}, 'geometry': s}
                    for j, (s, v)
                    in enumerate(
                    features.shapes(pred_mask, mask=pred_mask,
                                    transform=src.transform)))

                geoms.append(list(results)[0])
                scores_list.append(float(torch.Tensor.numpy(
                    outputs["instances"].scores.to("cpu")[i])))
                boulder_id.append(bid)
                bid = bid + 1

    gpd_polygonized_raster = gpd.GeoDataFrame.from_features(geoms,
                                                            crs=meta["crs"])
    gpd_polygonized_raster["scores"] = scores_list
    gpd_polygonized_raster["boulder_id"] = boulder_id
    gpd_polygonized_raster.to_file(out_shapefile)

#TODO
def fix_edge_cases(predictions_no_stride, predictions_with_stride,
                   graticule_no_stride, graticule_with_stride, in_raster):

    """
    I have found 5 potential outcome, and 4 edge cases so far:

    - (1) Originating from the non-shifted graticules,
    not intersecting any edges.

    - (2) Boulders intersecting the non-shifted graticules edges, so replaced by
    boulders from the shifted graticules (intersecting also the non-shifted
    graticules edges).
    ----------------------------------------------------------------------------
    intersect of non-shifted graticule edges (as a line) with non-shifted boulders
    replaced by
    intersect of non-shifted graticule edges (as a line) with shifted boulders
    ----------------------------------------------------------------------------

    - (3) Boulders might touch the edge og the non-shifted graticules but be on
    the outside of the shifted graticule (where there are no predictions!).
    In this case, the non-shifted predictions should be kept.
    ----------------------------------------------------------------------------
    symmetrical difference of both graticules
    intersect of non-shifted graticule edges
    ----------------------------------------------------------------------------

    - (4) Boulders might overlap if one boulder is one pixel off in the edge of
    the patch (not selected), but is touching/intersecting in the prediction
    with stride --> lead to duplicates, and therefore need to be merged...

    - (5) Boulders touch the edge(s) of the non-shifted graticules and shifted
    graticules (combination of left∕right with top∕down or vice-versa).
    A routine is run to merge boulders intersecting each other (merging...)

    - (6) same as 5 but in the area with only
    (intersection of non shifted grid and symmetrical difference of two graticules)
    --> lead to duplicates, and therefore need to be merged...,
    duplicates in (4) and (6) can be merged at the same time?

    OR do like Mathieu say, take the minimum boundary of all boulders within a
    patch, but it is not going to be that nice...

    :param predictions_no_stride:
    :param predictions_with_stride:
    :param graticule_no_stride:
    :param graticule_with_stride:
    :param in_raster:
    :return:
    """

    # reading input files + resolution
    gdf_no_stride = gpd.read_file(predictions_no_stride)
    gdf_w_stride = gpd.read_file(predictions_with_stride)
    gra_no_stride = gpd.read_file(graticule_no_stride)
    gra_w_stride = gpd.read_file(graticule_with_stride)
    in_res = raster.get_raster_resolution(in_raster)[0]

    # boundary of the strided area
    geom_whole_area = gra_w_stride.geometry.unary_union # return a geometry

    # symmetrical difference of the two graticules (non-shifted and shifted)
    gra_edge = gra_no_stride.geometry.unary_union.symmetric_difference(gra_w_stride.geometry.unary_union)

    # polygon to polyline (for no stride and stride graticules)
    gra_no_stride_line = gra_no_stride.geometry.boundary
    gra_w_stride_line = gra_w_stride.geometry.boundary
    gra_no_stride_line_without_outer_edge = gpd.GeoSeries(
        box(*gra_no_stride_line.total_bounds).boundary.symmetric_difference(
            gra_no_stride_line.geometry.unary_union), crs=gra_no_stride_line.crs).explode()
    idx_test = gra_no_stride_line_without_outer_edge.intersects(gra_edge)
    gra_no_stride_line_without_outer_edge2 = gra_no_stride_line_without_outer_edge[idx_test]

    # generate buffer of three pixel along this polyline
    #gra_no_stride_line_buffer = gra_no_stride_line.buffer(in_res*3)
    #gra_w_stride_line_buffer = gra_w_stride_line.buffer(in_res*3)

    # no stride edges clipped to the bounding box of the whole stride area (can speed up I think...)
    intersect_buffer = gra_no_stride_line.geometry.intersection(gra_w_stride.geometry.unary_union)
    intersect_buffer_union = intersect_buffer.geometry.unary_union

    # polygon to polylines for graticule with stride
    intersect_buffer_union_w_stride = gra_w_stride_line.geometry.unary_union



    # 1. first select all boulders that are not covered by the shifted graticules (at the edge)
    idx_edge_boulders = gdf_no_stride.geometry.intersects(gra_edge)
    gdf_edge_boulders_no_stride = gdf_no_stride[idx_edge_boulders]
    gdf_other_boulders_no_stride = gdf_no_stride[~idx_edge_boulders]

    # select boulders intersecting the no-stride grid (without the extreme edge)
    # + union and then explode them (easy way to merge them)
    idx_edge_case1 = gdf_edge_boulders_no_stride.geometry.intersects(
        gra_no_stride_line_without_outer_edge2.geometry.unary_union)
    gdf_edge_case1 = gdf_edge_boulders_no_stride[idx_edge_case1]

    # remove the merged boulders and others (to avoid duplicates)
    gdf_edge_boulders_no_stride = gdf_edge_boulders_no_stride[~idx_edge_case1] # DATA (1)

    # DATA different dataframe, eventually can use overlay to go around this problem
    # can use intersection.
    gdf_edge_case1_merged = gpd.GeoDataFrame(gpd.GeoSeries(gdf_edge_case1.geometry.unary_union, crs=gra_no_stride_line.crs).explode()) # DATA (2)
    gdf_edge_case1_merged = gdf_edge_case1_merged.set_geometry(0, drop=True, crs=gra_no_stride_line.crs).drop(columns=[0])

    # 2.
    # continue to work from gdf_other_boulders_no_stride
    # should select_by_loc_no_stride include the outer boundary? I don't think so!
    # select polygon intersecting this buffer polygon in no and with stride datasets
    # this step is extremely slow (can be done much more easily with sym difference..)
    select_by_loc_no_stride = gdf_other_boulders_no_stride.geometry.intersects(intersect_buffer_union)
    select_by_loc_w_stride = gdf_w_stride.geometry.intersects(intersect_buffer_union)

    #select_by_loc_w_stride_double = gdf_w_stride.geometry.intersects(
    #    intersect_buffer_union) & gdf_w_stride.geometry.intersects(intersect_buffer_union_w_stride)

    # select corresponding boulders in shifted graticules (2)
    edge_boulders_w_stride = gdf_w_stride[select_by_loc_w_stride] # DATA (3)

    # select inverse of no stride datasets (1)
    boulders_not_at_edge_no_stride = gdf_other_boulders_no_stride[~select_by_loc_no_stride] #DATA (4)

    # let's save the method we use to correct the data
    gdf_edge_boulders_no_stride["method"] = 1
    gdf_edge_case1_merged["method"] = 2
    edge_boulders_w_stride["method"] = 3
    boulders_not_at_edge_no_stride["method"] = 4

    # FINAL add selection of with stride datasets
    gdf_final = gpd.GeoDataFrame(pd.concat([gdf_edge_boulders_no_stride, gdf_edge_case1_merged, edge_boulders_w_stride, boulders_not_at_edge_no_stride], ignore_index=True))
    gdf_final["boulder_id"] = np.arange(1, gdf_final.shape[0] + 1)

    # need additional cleaning
    # There are cases where both no-stride and stride predictions are located at
    # an edge

    # + cases where overlapping (if overlapping between intersect more than X%)
    # merge them together... Differences or symmetric differences
    overlapping = gpd.overlay(gdf_final, gdf_final, how='intersection', keep_geom_type=False)
    overlapping = overlapping[overlapping.boulder_id_1 != overlapping.boulder_id_2]

    # line and points and other stuff do not represent a good overlap
    overlapping = overlapping[np.logical_or(overlapping.geometry.geom_type == "Polygon",
                          overlapping.geometry.geom_type == "MultiPolygon")]

    boulder_idx = list(overlapping.boulder_id_1.values) + list(overlapping.boulder_id_2.values)
    gdfb = gdf_final[gdf_final.boulder_id.isin(boulder_idx)]  # the inverse can be taken for non-overlapping

    # what if multiple overlap.....

    # check one by one (for twos)
    # if overlap add geometry to a new dataframe, list?
    # if not re-add geometries as they were at the start (separated)

    # for three or more... (maybe intersect of all of them, and similar? or same as two but step-wise?)

    # save it to a new shapefile


