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

sys.path.append("/home/nilscp/GIT/")
from MLtools import create_annotations

def predict(config_file, model_weights, device, image_dir, out_shapefile,
            search_pattern="*.tif", scores=0.5):
    """
    This function crashes if not a single prediction is detected!

    :param config_file:
    :param model_weights:
    :param device:
    :param image_dir:
    :param out_shapefile:
    :param search_pattern:
    :param scores:
    :return:
    """
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
    if gpd_polygonized_raster.shape[0] > 0:
        gpd_polygonized_raster.to_file(out_shapefile)
    else:
        schema = {"geometry": "Polygon",
                  "properties": {"raster_val": "float", "scores": "float",
                                 "boulder_id": "int"}}
        gdf_empty = gpd.GeoDataFrame(geometry=[])
        gdf_empty.to_file(out_shapefile, driver='ESRI Shapefile', schema=schema, crs=meta["crs"])

def default_predictions(in_raster, config_file, model_weights, device, scores, search_tif_pattern,
                        graticule_no_stride_p, graticule_with_stride_p, graticule_top_bottom_p,
                        graticule_left_right_p,
                        block_width, block_height, output_dir):

    output_dir = Path(output_dir)

    # no stride
    dataset_directory = output_dir / "inference-no-stride" / "images"
    out_shapefile = output_dir / "predictions-no-stride.shp"
    if out_shapefile.is_file():
        print(out_shapefile.as_posix() + " already exists. Delete file if it needs to be recomputed... ")
    else:
        (df_no_stride, gdf_no_stride) = create_annotations.generate_graticule_from_raster(in_raster, block_width, block_height, graticule_no_stride_p, stride=(0, 0))
        df_no_stride["dataset"] = "inference-no-stride"
        create_annotations.tiling_raster_from_dataframe(df_no_stride, output_dir, block_width, block_height)
        predict(config_file, model_weights, device, dataset_directory, out_shapefile, search_pattern=search_tif_pattern, scores=scores)

    # with stride
    dataset_directory = output_dir / "inference-w-stride" / "images"
    out_shapefile = output_dir / "predictions-w-stride.shp"
    if out_shapefile.is_file():
        print(out_shapefile.as_posix() + " already exists. Delete file if it needs to be recomputed... ")
    else:
        (df_w_stride, gdf_w_stride) = create_annotations.generate_graticule_from_raster(in_raster, block_width, block_height, graticule_with_stride_p, stride=(250, 250))
        df_w_stride["dataset"] = "inference-w-stride"
        create_annotations.tiling_raster_from_dataframe(df_w_stride, output_dir, block_width, block_height)
        predict(config_file, model_weights, device, dataset_directory, out_shapefile, search_pattern=search_tif_pattern, scores=scores)

    # top bottom
    dataset_directory = output_dir / "inference-top-bottom" / "images"
    out_shapefile = output_dir / "predictions-top-bottom.shp"
    if out_shapefile.is_file():
        print(out_shapefile.as_posix() + " already exists. Delete file if it needs to be recomputed... ")
    else:
        (df3, gdf3) = create_annotations.generate_graticule_from_raster(in_raster, block_width, block_height, graticule_top_bottom_p, stride=(250, 0))
        gdf_bounds = gdf3.geometry.bounds
        gdf_bounds["tile_id"] = gdf3.tile_id.values
        tile_id_edge = list(gdf_bounds.tile_id[gdf_bounds.maxy == gdf3.geometry.total_bounds[-1]].values) + list(
            gdf_bounds.tile_id[gdf_bounds.miny == gdf3.geometry.total_bounds[1]].values)
        gdf_test = gdf3[gdf3.tile_id.isin(tile_id_edge)]
        gdf_test.to_file(graticule_top_bottom_p)
        df3 = df3[df3.tile_id.isin(tile_id_edge)]
        df3["dataset"] = "inference-top-bottom"
        create_annotations.tiling_raster_from_dataframe(df3, output_dir, block_width, block_height)
        predict(config_file, model_weights, device, dataset_directory, out_shapefile, search_pattern=search_tif_pattern, scores=scores)

    # left right
    dataset_directory = output_dir / "inference-left-right" / "images"
    out_shapefile = output_dir / "predictions-left-right.shp"
    if out_shapefile.is_file():
        print(out_shapefile.as_posix() + " already exists. Delete file if it needs to be recomputed... ")
    else:
        (df4, gdf4) = create_annotations.generate_graticule_from_raster(in_raster, block_width, block_height, graticule_left_right_p, stride=(0, 250))
        gdf_bounds = gdf4.geometry.bounds
        gdf_bounds["tile_id"] = gdf4.tile_id.values
        tile_id_edge = list(gdf_bounds.tile_id[gdf_bounds.maxx == gdf4.geometry.total_bounds[-2]].values) + list(gdf_bounds.tile_id[gdf_bounds.minx == gdf4.geometry.total_bounds[0]].values)
        gdf_test = gdf4[gdf4.tile_id.isin(tile_id_edge)]
        gdf_test.to_file(graticule_left_right_p)
        df4 = df4[df4.tile_id.isin(tile_id_edge)]
        df4["dataset"] = "inference-left-right"
        create_annotations.tiling_raster_from_dataframe(df4, output_dir, block_width, block_height)
        predict(config_file, model_weights, device, dataset_directory, out_shapefile, search_pattern=search_tif_pattern, scores=scores)

    # fixing edge issues
    predictions_no_stride = output_dir / "predictions-no-stride.shp"
    predictions_with_stride = output_dir / "predictions-w-stride.shp"
    predictions_left_right = output_dir / "predictions-left-right.shp"
    predictions_top_bottom = output_dir / "predictions-top-bottom.shp"

    # fix invalid geometry (not sure if this work in all cases!)
    __ = quickfix_invalid_geometry(predictions_no_stride)
    __ = quickfix_invalid_geometry(predictions_with_stride)
    __ = quickfix_invalid_geometry(predictions_left_right)
    __ = quickfix_invalid_geometry(predictions_top_bottom)

    gdf = fix_edge_cases(predictions_no_stride, predictions_with_stride,
                         predictions_top_bottom, predictions_left_right,
                         graticule_no_stride_p, graticule_with_stride_p,
                         output_dir)

    return gdf

def geometry_for_inference(gra_no_stride, output_filename, output_dir):

    # ---------------------------------------------------------------------------
    # Generating edges as polyline for the center parts of the image
    print("...generating geometries for correcting predictions located at edges...")
    gra_center = gra_no_stride.geometry.bounds
    gra_center["tile_id"] = gra_no_stride.tile_id.values
    a = (gra_center.minx != gra_no_stride.geometry.total_bounds[0]) & (
                gra_center.maxx != gra_no_stride.geometry.total_bounds[2])

    b = (gra_center.miny != gra_no_stride.geometry.total_bounds[1]) & (
                gra_center.maxy != gra_no_stride.geometry.total_bounds[3])

    c = gra_center[a & b]
    tile_id_edge = c.tile_id.values
    d = gra_no_stride[gra_no_stride.tile_id.isin(tile_id_edge)]
    gra_center = d.geometry.boundary
    gra_center = gpd.GeoDataFrame(geometry=gpd.GeoSeries(gra_center))

    # ---------------------------------------------------------------------------
    # Generating edges as polyline for the top and bottom parts of the image
    gra_edge_top_bottom = gra_no_stride.geometry.bounds
    gra_edge_top_bottom["tile_id"] = gra_no_stride.tile_id.values
    tile_id_edge = list(
        gra_edge_top_bottom.tile_id[gra_edge_top_bottom.maxy == gra_no_stride.geometry.total_bounds[-1]].values) + list(
        gra_edge_top_bottom.tile_id[gra_edge_top_bottom.miny == gra_no_stride.geometry.total_bounds[1]].values)
    gra_edge_top_bottom = gra_no_stride[gra_no_stride.tile_id.isin(tile_id_edge)]
    gra_edge_top_bottom = gra_edge_top_bottom.geometry.boundary
    gra_edge_top_bottom = gpd.GeoDataFrame(geometry=gpd.GeoSeries(gra_edge_top_bottom))
    gra_edge_top_bottom = gpd.GeoDataFrame(geometry=gpd.GeoSeries(gra_edge_top_bottom.difference(
            box(*gra_no_stride.total_bounds).boundary), crs=gra_no_stride.crs))

    gra_edge_top_bottom = gpd.GeoDataFrame(geometry=gpd.GeoSeries(gra_edge_top_bottom.difference(d.geometry.unary_union.boundary),
        crs=gra_no_stride.crs))

    # ---------------------------------------------------------------------------
    # Generating edges as polyline for the left and right parts of the image
    gra_edge_left_right = gra_no_stride.geometry.bounds
    gra_edge_left_right["tile_id"] = gra_no_stride.tile_id.values

    tile_id_edge = list(gra_edge_left_right.tile_id[gra_edge_left_right.maxx == gra_no_stride.geometry.total_bounds[-2]].values) + list(
        gra_edge_left_right.tile_id[gra_edge_left_right.minx == gra_no_stride.geometry.total_bounds[0]].values)
    gra_edge_left_right = gra_no_stride[gra_no_stride.tile_id.isin(tile_id_edge)]
    gra_edge_left_right = gra_edge_left_right.geometry.boundary
    gra_edge_left_right = gpd.GeoDataFrame(geometry=gpd.GeoSeries(gra_edge_left_right))
    gra_edge_left_right = gpd.GeoDataFrame(geometry=gpd.GeoSeries(gra_edge_left_right.difference(
            box(*gra_no_stride.total_bounds).boundary), crs=gra_no_stride.crs))
    gra_edge_left_right = gpd.GeoDataFrame(geometry=gpd.GeoSeries(
        gra_edge_left_right.difference(d.geometry.unary_union.boundary),
        crs=gra_no_stride.crs))

    # ---------------------------------------------------------------------------
    # Updating the two last ones to get the right polylines
    sp = gpd.overlay(gra_edge_top_bottom, gra_edge_left_right, how='intersection', keep_geom_type=False)
    sp = sp[sp.geometry.geom_type == "LineString"]
    sp_bounds = sp.bounds
    gra_line_top_bottom = sp_bounds[
        np.logical_or(sp_bounds.miny == np.min(sp_bounds.miny),
                      sp_bounds.maxy == np.max(sp_bounds.maxy))]
    gra_line_left_right = sp_bounds[
        np.logical_or(sp_bounds.minx == np.min(sp_bounds.minx),
                      sp_bounds.maxx == np.max(sp_bounds.maxx))]

    gra_edge_left_right_final = gpd.overlay(gra_edge_left_right, sp.loc[gra_line_top_bottom.index],
                how="symmetric_difference", keep_geom_type=False)

    gra_edge_top_bottom_final = gpd.overlay(gra_edge_top_bottom, sp.loc[gra_line_left_right.index],
                how="symmetric_difference", keep_geom_type=False)

    gra_center.to_file(output_dir / output_filename[0]) # "edge-lines-center.shp"
    gra_edge_left_right_final.to_file(output_dir / output_filename[1]) # "edge-lines-left-right.shp"
    gra_edge_top_bottom_final.to_file(output_dir / output_filename[2]) # "edge-lines-top-bottom.shp"

    return (gra_center, gra_edge_left_right_final, gra_edge_top_bottom_final)

def lines_where_boulders_intersect_edges(gdf_boulders, gdf_lines, output_filename, output_dir):

    """
    Maybe better to give filename? --> avoid loading the dataset every times..

    :param boulders_shp:
    :param lines_shp:
    :param output_dir:
    :return:
    """
    edge_issues = gpd.overlay(gdf_boulders, gdf_lines, how='intersection', keep_geom_type=False)
    edge_issues.geometry.geom_type.unique()

    # only keep the longest line when a multilinestring is generated
    gdf_MultiLineString = edge_issues[edge_issues.geometry.geom_type == 'MultiLineString']
    gdf_MultiLineString = gdf_MultiLineString.explode()
    gdf_MultiLineString["length"] = gdf_MultiLineString.geometry.length
    gdf_MultiLineString = gdf_MultiLineString.sort_values(by=["boulder_id", "length"])
    gdf_MultiLineString = gdf_MultiLineString.drop_duplicates(subset="boulder_id", keep='last')

    gdf_LineString = edge_issues[edge_issues.geometry.geom_type == 'LineString']
    gdf_AllLines = gpd.GeoDataFrame(pd.concat([gdf_LineString, gdf_MultiLineString], ignore_index=True))
    if gdf_AllLines.shape[0] > 0:
        gdf_AllLines.to_file(output_dir / output_filename)
    else:
        None
    return (gdf_AllLines)

# should replace
def replace_boulder_intersecting(gdf_boulders_original, gdf_boulders_replace, gdf_edge_intersections, output_filename, output_dir):

    """
    VERY IMPORTANT:
    Should I have a flag, only replace if a "hit" is found for the same?
    or should I just expect that in order for a detection to be robust,
    it needs to be detected in multiple predictions?

    :param gdf_boulders_original:
    :param gdf_boulders_replace:
    :param gdf_edge_intersections:
    :param output_filename:
    :return:
    """
    print("...replacing boulders at edge...")
    if gdf_edge_intersections.shape[0] > 0:
        gdf_edge_intersections.boulder_id = gdf_edge_intersections.boulder_id.astype('int')
        idx_boulders_at_edge = gdf_edge_intersections.boulder_id.unique()

        gdf_boulders_original_at_edge = gdf_boulders_original[gdf_boulders_original.boulder_id.isin(idx_boulders_at_edge)]
        gdf_boulders_original_not_at_edge = gdf_boulders_original[~(gdf_boulders_original.boulder_id.isin(idx_boulders_at_edge))]

        # Finding intersecting boulders in other
        gdf_boulders_replace_intersecting = gpd.overlay(gdf_boulders_replace, gdf_edge_intersections, how='intersection', keep_geom_type=False)
        idx_boulders_at_edge = gdf_boulders_replace_intersecting.boulder_id_1.unique()
        gdf_boulders_replace_at_edge = gdf_boulders_replace[gdf_boulders_replace.boulder_id.isin(idx_boulders_at_edge)]

        gdf = gpd.GeoDataFrame(pd.concat([gdf_boulders_original_not_at_edge, gdf_boulders_replace_at_edge], ignore_index=True))
        gdf.boulder_id = np.arange(gdf.shape[0])
    else:
        gdf = gdf_boulders_original
    gdf.to_file(output_dir / output_filename)
    return (gdf)

def fix_double_edge_cases(gra_no_stride, gra_w_stride):
    hot_spot = gpd.overlay(gpd.GeoDataFrame(geometry=gpd.GeoSeries(gra_no_stride.boundary)),
                           gpd.GeoDataFrame(geometry=gpd.GeoSeries(gra_w_stride.boundary)),
                           how='intersection', keep_geom_type=False)

    hot_spot = hot_spot.explode().drop_duplicates()
    return (hot_spot)

def replace_boulders_at_double_edge(gra_no_stride, gra_w_stride, gdf_no_stride, gdf_w_stride, gdf_last, output_filename, output_dir):

    """
    One of the problem is that even if we use with-stride predictions for boulders
    intersecting the edge of the no-stride grid, there are still places where the
    same boulders will remain splited. This is true for intersections between the
    no-stride and with-stride grids. This function merge boulders that touch
    the two intersections (here referred as hot spot) as a solution.

    :param graticule_no_stride:
    :param graticule_with_stride:
    :param gdf_no_stride:
    :param gdf_w_stride:
    :param gdf_last:
    :return:
    """
    print("...replacing boulders that intersects no- and with- stride graticules...")
    # hotspot for errors
    hot_spot = fix_double_edge_cases(gra_no_stride, gra_w_stride)

    # fixing errors
    boulders_at_double_edge_ns = gpd.overlay(gdf_no_stride, hot_spot, how='intersection', keep_geom_type=False)
    boulders_at_double_edge_ws = gpd.overlay(gdf_w_stride, hot_spot, how='intersection', keep_geom_type=False)
    boulders_at_double_edge_fp = gpd.overlay(gdf_last, hot_spot, how='intersection', keep_geom_type=False)

    idx1 = boulders_at_double_edge_ns.boulder_id.unique()
    idx2 = boulders_at_double_edge_ws.boulder_id.unique()
    idx3 = boulders_at_double_edge_fp.boulder_id.unique()

    gdf_tom = gpd.GeoDataFrame(pd.concat([gdf_no_stride[gdf_no_stride.boulder_id.isin(idx1)],
                                          gdf_w_stride[gdf_w_stride.boulder_id.isin(idx2)],
                                          gdf_last[gdf_last.boulder_id.isin(idx3)]], ignore_index=True))

    gdf_double_edge = gpd.GeoDataFrame(geometry=gpd.GeoSeries(gdf_tom.geometry.unary_union, crs=gdf_no_stride.crs)).explode()
    gdf_not_double_edge = gdf_last[~(gdf_last.boulder_id.isin(idx3))]
    gdf = gpd.GeoDataFrame(pd.concat([gdf_double_edge, gdf_not_double_edge], ignore_index=True))
    gdf.boulder_id = np.arange(gdf.shape[0])
    gdf.to_file(output_dir / output_filename)
    return (gdf)

def merging_overlapping_boulders(gdf_final, output_filename, output_dir):

    """
    This function merge all of overlapping features that intersect each other,
    and have more than 50% overlap between the smallest and the largest polygon
    of the overlapping features. This function works with multiple polygons if
    all polygons intersect each other.

    For example if you have polygons A, B and C, A being the largest polygon,
    and that AB, AC and BC intersect each other, all good.

    If B intersects A, but C intersects only B, it can
    (if overlap is more than 50%) potentially result in two merged polygons. So,
    you still may have overlapping features at the end... but it should (hopefully)
    be for a very tiny portion of boulders...

    :param gdf_final:
    :return:
    """
    print("...merging overlapping boulders...")
    overlapping = gpd.overlay(gdf_final, gdf_final, how='intersection', keep_geom_type=False)
    overlapping = overlapping[overlapping.boulder_id_1 != overlapping.boulder_id_2]

    # line and points and other stuff do not represent a good overlap
    overlapping = overlapping[np.logical_or(overlapping.geometry.geom_type == "Polygon", overlapping.geometry.geom_type == "MultiPolygon")]

    boulder_idx = list(overlapping.boulder_id_1.values)
    gdf_overlap = gdf_final[gdf_final.boulder_id.isin(boulder_idx)]
    gdf_overlap["area"] = gdf_overlap.geometry.area
    gdf_non_overlapping = gdf_final[~(gdf_final.boulder_id.isin(boulder_idx))] # data, the inverse can be taken for non-overlapping

    # overlapping contains combination A and B and B and A (which is the same)
    # we get rid of it by multiplying boulder_id_1 by boulder_id_2
    # it creates an unique combination, which we use to drop duplicates
    # combinations
    overlapping["multi"] = (overlapping.boulder_id_1 + 1) * (overlapping.boulder_id_2 + 1)
    overlapping.drop_duplicates(subset="multi", keep='first', inplace=True)
    overlapping = overlapping.drop(columns=['multi'])

    merge_list = []
    geom_of_merged = []

    # looping through potential combination of boulders that can be merged
    for index, row in tqdm(overlapping.iterrows(), total=overlapping.shape[0]):
        gdf_selection = gdf_overlap[gdf_overlap.boulder_id.isin([row.boulder_id_1, row.boulder_id_2])]
        gdf_selection = gdf_selection.sort_values(by=["area"])
        value = gdf_selection.iloc[0].geometry.intersection(
            gdf_selection.iloc[1].geometry).area / gdf_selection.iloc[0].geometry.area
        if value > 0.50:
            merge_list.append(True)
            geom_of_merged.append(gdf_selection.geometry.unary_union)
        else:
            merge_list.append(False)
            geom_of_merged.append(0)

    overlapping["is_merged"] = merge_list
    overlapping["geom_merged"] = geom_of_merged
    overlapping["geometry"] = overlapping["geom_merged"]
    overlapping = overlapping.drop(columns=["geom_merged"])

    overlapping_tbm = overlapping[overlapping["is_merged"] == True]
    not_overlapping = overlapping[overlapping["is_merged"] == False]

    idx_not_overlapping = sorted(list(not_overlapping.boulder_id_1.unique()) + list(not_overlapping.boulder_id_2.unique()))
    gdf_overlap_but_not = gdf_overlap[gdf_overlap.boulder_id.isin(idx_not_overlapping)]  # DATA

    # need to drop a few values
    gdf_non_overlapping = gdf_non_overlapping[gdf_non_overlapping.columns[gdf_non_overlapping.columns.isin(['geometry', 'boulder_id'])]]
    overlapping_tbm = overlapping_tbm[overlapping_tbm.columns[overlapping_tbm.columns.isin(['geometry'])]]
    overlapping_tbm["boulder_id"] = 0
    gdf_overlap_but_not = gdf_overlap_but_not[gdf_overlap_but_not.columns[gdf_overlap_but_not.columns.isin(['geometry'])]]
    gdf_overlap_but_not["boulder_id"] = 0

    gdf = gpd.GeoDataFrame(pd.concat([gdf_non_overlapping, overlapping_tbm, gdf_overlap_but_not], ignore_index=True))
    gdf.boulder_id = np.arange(gdf.shape[0])
    gdf.to_file(output_dir / output_filename)
    return (gdf)

def fix_edge_cases(predictions_no_stride, predictions_with_stride,
                       predictions_top_bottom, predictions_left_right,
                       graticule_no_stride, graticule_with_stride, output_dir):

    output_dir = Path(output_dir)

    gdf_no_stride = gpd.read_file(predictions_no_stride)
    gdf_w_stride = gpd.read_file(predictions_with_stride)
    gdf_left_right = gpd.read_file(predictions_left_right)
    gdf_top_bottom = gpd.read_file(predictions_top_bottom)

    gra_no_stride = gpd.read_file(graticule_no_stride)
    gra_w_stride = gpd.read_file(graticule_with_stride)

    gdf_no_stride["boulder_id"] = gdf_no_stride["boulder_id"].values.astype('int')
    gdf_w_stride["boulder_id"] = gdf_w_stride["boulder_id"].values.astype('int')
    gdf_left_right["boulder_id"] = gdf_left_right["boulder_id"].values.astype('int')
    gdf_top_bottom["boulder_id"] = gdf_top_bottom["boulder_id"].values.astype('int')

    output_filename = ("edge-lines-center.shp", "edge-lines-left-right.shp", "edge-lines-top-bottom.shp")
    (gra_center, gra_edge_left_right_final,gra_edge_top_bottom_final) = geometry_for_inference(gra_no_stride, output_filename, output_dir)

    lines_center = lines_where_boulders_intersect_edges(gdf_no_stride, gra_center, 'lines-where-boulder-intersects-center.shp', output_dir)
    lines_topbottom = lines_where_boulders_intersect_edges(gdf_no_stride, gra_edge_top_bottom_final, 'lines-where-boulder-intersects-top-bottom.shp', output_dir)
    lines_leftright = lines_where_boulders_intersect_edges(gdf_no_stride, gra_edge_left_right_final, 'lines-where-boulder-intersects-left-right.shp', output_dir)

    gdf1 = replace_boulder_intersecting(gdf_no_stride, gdf_w_stride, lines_center, "preliminary-predictions.shp", output_dir)
    gdf2 = replace_boulder_intersecting(gdf1, gdf_top_bottom, lines_topbottom, "preliminary-predictions.shp", output_dir)
    gdf3 = replace_boulder_intersecting(gdf2, gdf_left_right, lines_leftright, "preliminary-predictions.shp", output_dir)
    gdf_degde = replace_boulders_at_double_edge(gra_no_stride, gra_w_stride, gdf_no_stride, gdf_w_stride, gdf3, "preliminary-predictions.shp", output_dir)
    gdf_final = merging_overlapping_boulders(gdf_degde, "predictions-including-edge-fix.shp", output_dir)
    return gdf_final


def quickfix_invalid_geometry(boulders_shp):
    print ("...fixing invalid geometries...")
    gdf_boulders = gpd.read_file(boulders_shp)
    valid_geom_idx = gdf_boulders.geometry.is_valid

    if ~valid_geom_idx.all():
        n = gdf_boulders[valid_geom_idx == False].shape[0]
        print(str(n) + " invalid geometry(ies) detected")
        gdf_valid = gdf_boulders[valid_geom_idx]
        gdf_invalid = gdf_boulders[~valid_geom_idx]
        gdf_invalid["geometry"] = gdf_invalid.geometry.buffer(0)
        gdf_boulders = gpd.GeoDataFrame(
            pd.concat([gdf_valid, gdf_invalid], ignore_index=False))
    else:
        None
    if gdf_boulders.shape[0] > 0: # if non-empty
        gdf_boulders.to_file(boulders_shp)
    else:
        None
    return (gdf_boulders)
