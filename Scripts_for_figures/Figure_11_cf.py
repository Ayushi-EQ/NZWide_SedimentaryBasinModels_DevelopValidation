# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 10:51:48 2026

@author: ati47
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from pygmt_helper import plotting
import json
import geopandas as gpd

from shapely.geometry import Polygon

main_region = (165, 180, -48, -33)

def auto_region_from_points(lon, lat, pad_deg=0.0, min_span_deg=1.5, clamp=None):
    xmin, xmax = np.min(lon), np.max(lon)
    ymin, ymax = np.min(lat), np.max(lat)
    if (xmax - xmin) < min_span_deg:
        mid = 0.5 * (xmin + xmax)
        xmin, xmax = mid - min_span_deg / 2, mid + min_span_deg / 2
    if (ymax - ymin) < min_span_deg:
        mid = 0.5 * (ymin + ymax)
        ymin, ymax = mid - min_span_deg / 2, mid + min_span_deg / 2
    xmin -= pad_deg; xmax += pad_deg
    ymin -= pad_deg; ymax += pad_deg
    if clamp:
        xmin = max(xmin, clamp[0]); xmax = min(xmax, clamp[1])
        ymin = max(ymin, clamp[2]); ymax = min(ymax, clamp[3])
    return (xmin, xmax, ymin, ymax)

def plot_basins(fig, basin_dir, domain_poly=None, pen="0.2p,black"):
    for gj in sorted(Path(basin_dir).glob("*.geojson")):
        gdf = gpd.read_file(gj)
        if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs(4326)
        if domain_poly is not None:
            gdf = gdf[gdf.intersects(domain_poly)]   
            if gdf.empty:
                continue
        fig.plot(data=gdf, pen=pen)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate an IM Ratio Map between two specific simulation files."
    )
    parser.add_argument(
        "file_a",
        type=Path,
        help="Path to the first simulation H5 file (Numerator)",
    )
    parser.add_argument(
        "file_b",
        type=Path,
        help="Path to the second simulation H5 file (Denominator)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Path to save the output PNG map",
    )
    parser.add_argument(
        "--im",
        default="pSA_2.0",
        help="Intensity Measure to plot (default: pSA_2.0)",
    )
    parser.add_argument(
        "--component",
        default="rotd50",
        help="IM component (default: rotd50)",
    )
    
    parser.add_argument("--basins", type=Path, default=Path("/scratch/jobs/ati47/Paper2_GMTFigues/NZVM2p09"),help="Folder of basin boundary GeoJSONs")
    return parser.parse_args()
    
def lon360(lon):
    return lon % 360.0

def read_domain_corners(realisation_path):
    with open(realisation_path) as f:
        data = json.load(f)
    pts = data["domain"]["domain"]
    lons = [lon360(p["longitude"]) for p in pts]
    lats = [p["latitude"]          for p in pts]
    lons.append(lons[0]); lats.append(lats[0])
    return lons, lats

        
def extract_sim_ims(dset, im_name, component):
    """Extracts specific IM and component from xarray dataset."""
    if im_name.startswith("pSA"):
        try:
            period = float(im_name.split("_")[1])
            return dset["pSA"].sel(component=component, period=period)
        except (IndexError, ValueError):
            raise ValueError(f"Invalid pSA format: {im_name}. Expected pSA_1.0")
    return dset[im_name].sel(component=component)



def main():
    args = parse_args()
    realisation_path = args.file_a.parent / "realisation.json"
    dom_lon, dom_lat = read_domain_corners(realisation_path)


    
    if not (args.file_a.exists() and args.file_b.exists()):
        print("Error: One or both input files do not exist.")
        return

    # Open datasets
    with xr.open_dataset(args.file_a, engine="h5netcdf") as sim_a, \
         xr.open_dataset(args.file_b, engine="h5netcdf") as sim_b:

        lons = sim_a.longitude.values
        lats = sim_a.latitude.values
        finite_xy = np.isfinite(lons) & np.isfinite(lats)
        region = auto_region_from_points(
            lons[finite_xy], lats[finite_xy],
            pad_deg=0.5, min_span_deg=1.8, clamp=main_region,
        )

        print(f"Processing IM: {args.im}")

        # Extract data
        val_a = extract_sim_ims(sim_a, args.im, args.component)
        val_b = extract_sim_ims(sim_b, args.im, args.component)
        
        im = args.im.split('_')[0]
        dfa = val_a.to_dataframe().reset_index()[['latitude','longitude', im]].rename(columns={im: 'a'})
        dfb = val_b.to_dataframe().reset_index()[['latitude','longitude', im]].rename(columns={im: 'b'})

        m = dfa.merge(dfb, on=['latitude', 'longitude'], how='inner')

        m['logratio'] = np.log(m['a'] / m['b'])
        df = m.rename(columns=dict(latitude='lat', longitude='lon'))[['lat', 'lon', 'logratio']]

        grid = plotting.create_grid(df, 'logratio', grid_spacing="100e/100e", region=region, interp_method="linear",set_water_to_nan=True)
                                  
        
        diff_threshold = 0.05
        grid = grid.where(np.abs(grid) >= diff_threshold)
        
        # Dynamic color scale
        abs_max = 2.0
        inc = 0.2
        # Plotting logic
        fig = plotting.gen_region_fig(
            region=region,
            plot_kwargs={
                "water_color": "white",
                "topo_cmap_min": -2000,
                "topo_cmap_max": 7000,
                "topo_cmap": "gray",
            },
            plot_highways=False,
            plot_topo=True,
        )
        print(np.nanmax(np.abs(grid)))
        print(np.nanpercentile(np.abs(grid), 97.5)) 
        plotting.plot_grid(
            fig,
            grid,
            cmap="polar",
            cmap_limits=(-abs_max, abs_max, inc),
            cmap_limit_colors=("red", "blue"),
            #continuous_cmap = True,
            #transparency = 20,
            cb_label=None,
            plot_contours=False,
            
        )

        #fig.colorbar(frame=["xaf0.5", 'x+l With/without basins ln SA (2.0 s) ratio'])
        domain_poly = Polygon(zip(dom_lon, dom_lat))   
        plot_basins(fig, args.basins, domain_poly=domain_poly, pen="0.5p,black")
        fig.plot(x=dom_lon, y=dom_lat, pen="1.5p,black")
        fig.plot(x=[sim_a.attrs.get("hypo_lon", 0)],
                 y=[sim_a.attrs.get("hypo_lat", 0)],
                 style="a0.6c", fill="white", pen="1.5p,black")
        
        # Ensure directory exists and save
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output)
        print(f"Successfully saved map to: {args.output}")


if __name__ == "__main__":
    main()