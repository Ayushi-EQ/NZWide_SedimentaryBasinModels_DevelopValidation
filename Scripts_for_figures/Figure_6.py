# -*- coding: utf-8 -*-
"""
Created on Fri Jul  4 15:52:15 2025

@author: ati47
"""

import typer
from pathlib import Path
import pandas as pd
import pygmt
from pygmt_helper import plotting
import xarray as xr
import shapely
import geojson
import numpy as np
import scipy as sp
import geopandas as gpd
import logging
import sys
import matplotlib.pyplot as plt

# configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

app = typer.Typer()

def region_from_polygons(ref_polygons, pad=0.02):

    xmin, ymin, xmax, ymax = ref_polygons[0].bounds

    for p in ref_polygons[1:]:
        b = p.bounds
        xmin = min(xmin, b[0])
        ymin = min(ymin, b[1])
        xmax = max(xmax, b[2])
        ymax = max(ymax, b[3])

    region = [
        xmin - pad,
        xmax + pad,
        ymin - pad,
        ymax + pad,
    ]
    return region

def load_outline(outline_file: Path) -> list[shapely.Polygon]:
    from shapely.geometry import shape, Polygon, MultiPolygon

    with open(outline_file, "r") as f:
        geojson_data = geojson.load(f)

    polygons = []
    for feature in geojson_data["features"]:
        geom = feature.get("geometry")
        if geom is None:
            continue
        shapely_geom = shape(geom)
        if isinstance(shapely_geom, Polygon):
            polygons.append(shapely_geom)
        elif isinstance(shapely_geom, MultiPolygon):
            polygons.extend(shapely_geom.geoms)

    return polygons


def plot_polygon(
    fig: pygmt.Figure,
    polygon: (
        shapely.LineString
        | shapely.MultiLineString
        | shapely.Polygon
        | shapely.MultiPolygon
    ),
    **kwargs,
) -> None:

    if isinstance(
        polygon, (shapely.MultiPolygon, shapely.MultiLineString)
    ):  # Simplified isinstance check
        for part in polygon.geoms:
            plot_polygon(fig, part, **kwargs)
    elif isinstance(polygon, shapely.LineString):
        coords = np.array(polygon.coords)
        fig.plot(
            x=coords[:, 0],
            y=coords[:, 1],
            **kwargs,
        )
    else:  # Assumes shapely.Polygon
        polygon_coords = np.array(polygon.exterior.coords)
        fig.plot(
            x=polygon_coords[:, 0],
            y=polygon_coords[:, 1],
            **kwargs,
        )
        # Plotting interior rings if any (optional, but good for completeness)
        for interior in polygon.interiors:
            interior_coords = np.array(interior.coords)
            fig.plot(x=interior_coords[:, 0], y=interior_coords[:, 1], **kwargs)


def basin_depth(basin_elevation: xr.Dataset, dem: xr.Dataset) -> xr.DataArray:
    method = "linear"

    interpolator = sp.interpolate.RegularGridInterpolator(
        (dem.latitude.values, dem.longitude.values),
        dem.elevation.values,
        method=method,
    )

    lon, lat = np.meshgrid(
        basin_elevation.longitude.values, basin_elevation.latitude.values
    )

    dem_height = interpolator((lat, lon))

    depth = dem_height - basin_elevation.elevation.values

    return xr.DataArray(
        data=depth,
        coords={
            "latitude": basin_elevation.latitude.values,
            "longitude": basin_elevation.longitude.values,
        },
        dims=("latitude", "longitude"),
    )


def load_basins(basin_data: Path) -> dict[str, xr.Dataset]:
    basins = {}
    for basin_directory in basin_data.iterdir():
        latest_basin_path = max(basin_directory.glob("*basement*.h5"), default=None)
        if not latest_basin_path:
            logging.warn(f"Skipping basin {basin_directory}")
            continue
        logging.info(f"Loading basin file: {latest_basin_path}")

        basins[basin_directory.stem] = xr.open_dataset(latest_basin_path)
    return basins


def load_outlines(basin_data):
    basin_outlines = {}
    for basin_directory in basin_data.iterdir():  # Renamed for clarity
        if not basin_directory.is_dir():
            continue
        polygons = []
        for outline in basin_directory.glob("*.geojson"):
            polygons.extend(load_outline(outline))
        polygon_for_region_mask = shapely.union_all(polygons)
        # This is the original outline from GeoJSON, assumed to be WGS84
        basin_name = basin_directory.stem
        basin_outlines[basin_name] = polygon_for_region_mask
    return basin_outlines


def clip_polygon(polygon, grid):

    lon, lat = np.meshgrid(grid.longitude.values, grid.latitude.values)

    mask = shapely.contains_xy(polygon, lon.ravel(), lat.ravel()).reshape(lon.shape)
    return grid.where(mask)





@app.command()
def single_basin(
    basin_elevation_file: Path = typer.Argument(..., help="Basin basement .h5 file"),
    reference_geojson: Path = typer.Argument(..., help="Reference basin outline GeoJSON"),
    output_path: Path = typer.Argument(..., help="Output figure path"),
    width: int = 17,
):
    """
    Plot basin depth for a SINGLE basin using a reference GeoJSON outline
    """

    logging.info("Starting single-basin depth plot")
    # ------------------------------------------------------------
    # Load HVSR inferred depths CSV
    # ------------------------------------------------------------
    hvsr_csv = Path("HVSR_inferred_depths.csv")  
    logging.info(f"Loading HVSR data: {hvsr_csv}")
    
    hvsr_df = pd.read_csv(hvsr_csv)
    # ------------------------------------------------------------
    # Load reference outline polygons
    # ------------------------------------------------------------
    logging.info(f"Loading reference GeoJSON: {reference_geojson}")
    ref_polygons = load_outline(reference_geojson)

    if not ref_polygons:
        raise RuntimeError("No valid polygons found in reference GeoJSON")

    # ------------------------------------------------------------
    # Compute plotting region (your exact logic)
    # ------------------------------------------------------------
    region = region_from_polygons(ref_polygons)
    logging.info(f"Computed plotting region: {region}")

    # ------------------------------------------------------------
    # Generate base figure
    # ------------------------------------------------------------
    fig = plotting.gen_region_fig(
        plot_kwargs={
            "water_color": "white",
            "topo_cmap_min": -900,
            "topo_cmap_max": 3100,
            "topo_cmap": "gray",
        },
        plot_highways=False,
        region=region,
        plot_topo=True,
        high_res_topo=True,
        config_options=dict(
            MAP_FRAME_TYPE="fancy",
            FORMAT_GEO_MAP="ddd.x",
            MAP_GRID_PEN="0.5p,gray",
            MAP_TICK_PEN_PRIMARY="1p,black",
            MAP_FRAME_PEN="1p,black",
            MAP_FRAME_AXES="WSne",
            FONT_ANNOT_PRIMARY="14p,Helvetica,black",
            FONT_LABEL="18p,Helvetica,black",
        ),
    )

    # ------------------------------------------------------------
    # Load DEM
    # ------------------------------------------------------------
    vel_model_path = Path("/home/ati47/.local/cache/nzcvm_data_root")
    dem_path = vel_model_path / "surface" / "NZ_DEM_HD.h5"
    logging.info(f"Loading DEM: {dem_path}")
    dem = xr.open_dataset(dem_path, engine="h5netcdf")

    # ------------------------------------------------------------
    # Load basin elevation model
    # ------------------------------------------------------------
    logging.info(f"Loading basin elevation model: {basin_elevation_file}")
    basin_elevation = xr.open_dataset(basin_elevation_file)

    # ------------------------------------------------------------
    # Compute basin depth
    # ------------------------------------------------------------
    depth = basin_depth(basin_elevation, dem)

    # ------------------------------------------------------------
    # Clip basin depth to reference polygon
    # ------------------------------------------------------------
    basin_polygon = shapely.union_all(ref_polygons)
    # Keep only points inside basin polygon
    mask = shapely.contains_xy(basin_polygon, hvsr_df["Longitude"].values, hvsr_df["Latitude"].values)

    hvsr_df = hvsr_df[mask]
    depth = clip_polygon(basin_polygon, depth)

    # ------------------------------------------------------------
    # Plot basin depth
    # ------------------------------------------------------------
    pygmt.makecpt(cmap="inferno", series=[0, 1000], reverse=True)

    fig.grdimage(
        grid=depth,
        cmap=True,
        nan_transparent=True,
        projection=f"M{width}c",
    )
    
    fig.grdcontour(grid=depth, interval=250, annotation = 500, pen="0.5p,black")
    # ------------------------------------------------------------
    # Plot outline on top
    # ------------------------------------------------------------
    plot_polygon(
        fig,
        basin_polygon,
        pen="1.5p,black",
    )
    


    # ------------------------------------------------------------
    # Coastlines + colorbar
    # ------------------------------------------------------------
    fig.coast(shorelines="1/0.25p,gray26,solid")
    fig.colorbar(frame=["x+lBasin depth (m)"])
    # ------------------------------------------------------------
    # Plot HVSR measurement locations
    # ------------------------------------------------------------
    fig.plot(
        x=hvsr_df["Longitude"],
        y=hvsr_df["Latitude"],
        style="c0.25c",
        fill=hvsr_df["InferredDepth"],
        cmap=True,             
        pen="0.6p,black",
    )
    # ------------------------------------------------------------
    # Save
    # ------------------------------------------------------------
    logging.info(f"Saving figure to {output_path}")
    fig.savefig(output_path, dpi=900, anti_alias=True)



if __name__ == "__main__":
    app()
