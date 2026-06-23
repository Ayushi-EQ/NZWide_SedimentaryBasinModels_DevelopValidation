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
def main(output_path: Path, width: int = 17):
    logging.info("Starting main plotting function...")
    region = [165.00, 180.00, -47.50, -34.00]

    fig = plotting.gen_region_fig(
        plot_kwargs={
            "water_color": "white",
            "topo_cmap_min": -900,
            "topo_cmap_max": 3100,
            'topo_cmap': 'gray'
        },
        plot_highways=False,
        region = region,
        high_res_topo=True,
        config_options=dict(
            MAP_FRAME_TYPE="fancy",
            FORMAT_GEO_MAP="ddd",
            MAP_GRID_PEN="0.5p,gray",
            MAP_TICK_PEN_PRIMARY="1p,black",
            MAP_FRAME_PEN="1p,black",
            MAP_FRAME_AXES="WSne",
            FONT_ANNOT_PRIMARY="14p,Helvetica,black",   # ✅ tick label font size
            FONT_LABEL="18p,Helvetica,black",           # ✅ axis label font size
        ),
    )
    vel_model_path = Path("/home/ati47/.local/cache/nzcvm_data_root")
    dem_path = (
        vel_model_path
        / "surface"
        / "NZ_DEM_HD.h5"
    )
    logging.info(f"Loading DEM dataset: {dem_path}")
    dem = xr.open_dataset(dem_path, engine="h5netcdf")
    # dem = xr.open_dataset(dem_path)

    basin_dir = vel_model_path / "regional"
    logging.info(f"Loading basins from directory: {basin_dir}")
    basin_outlines = {}

    basins = load_basins(basin_dir)
    basin_outlines = load_outlines(basin_dir)
    pygmt.makecpt(cmap='lajolla', series=[0, 4000],reverse=True) #reverse = True depends on pygmt_version 

    for basin_name, basin_elevation in basins.items():
        logging.info(f"Processing basin: {basin_name}")
        depth = basin_depth(basin_elevation, dem)
        depth = clip_polygon(basin_outlines[basin_name], depth)
        fig.grdimage(depth, nan_transparent=True, cmap=True, projection='M17c')
        
    fig.coast(shorelines='1/0.25p,gray26,solid')
    logging.info(f"Saving figure to {output_path}")
    # fig.colorbar(cmap=True, frame=['x+lDepth', 'y+lm'])
    fig.colorbar(cmap=True, frame=["x+lBasin depth (m)"])
    # stations = pd.read_csv("LATEST_NZGMDBv_4.3_Categorizations.csv")
    # stations_filtered = stations[stations["Basin Type Latest"] != "Non-Basin"]
    # fig.plot(
    # x=stations_filtered['Longitude'],
    # y=stations_filtered['Latitude'],
    # style="t0.03c",         
    # fill="black",
    # pen="black"
    # )
    fig.savefig(output_path, dpi=900, anti_alias=True)


if __name__ == "__main__":
    app()
