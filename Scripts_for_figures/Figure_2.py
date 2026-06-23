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
import shapely
import geojson
import numpy as np
import geopandas as gpd

# def default_pygmt_config():
#     pygmt.config(
#         MAP_FRAME_TYPE="plain",
#         FORMAT_GEO_MAP="ddd.xx",
#         MAP_FRAME_PEN="thinner,black",
#         FONT_ANNOT_PRIMARY="10p,Helvetica,black",
#     )


# def plot_basemap(fig: pygmt.Figure, region, width: int = 17,frame=None):
#     if frame is not False:
#         fig.basemap(
#             region=region,
#             projection=f"M{width}c",
#             frame=["xa2f", "ya2f", "xaf+lLongitude", "yaf+lLatitude"],
#         )
#     nz_map_data = plotting.NZMapData.load(high_res_topo=True)
#     pygmt.makecpt(
#         cmap="gray",
#         series=[-900, 3100, 1],
#         continuous=True,
#         reverse=True,
#     )
#     fig.grdimage(
#         grid=nz_map_data.topo_grid,
#         projection=f"M{width}c",
#         region=region,
#         cmap=True,
#         shading=True,
#         nan_transparent=True,
#     )
#     fig.plot(data=nz_map_data.water_df, fill="white")



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
    polygon: shapely.LineString
    | shapely.MultiLineString
    | shapely.Polygon
    | shapely.MultiPolygon,
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



def main(output_path: Path, width: int = 17):
    region = [165.00, 181.00, -48, -34.00]
    fig = plotting.gen_region_fig(
        plot_kwargs={
            "water_color": "white",
            "topo_cmap_min": -900,
            "topo_cmap_max": 3100,
            'topo_cmap': 'gray'
        },
        plot_highways=False,
        region=region,
        plot_topo=True,
        high_res_topo=False,
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
    # region = [165.00, 180.00, -47.50, -34.00]

    # plot_basemap(fig, region, width)

    # Load your 3 core files
    im_obs = pd.read_csv("selected_gmotions.csv")
    station_data = pd.read_csv("Features.csv")
    station_data["Longitude"] = pd.to_numeric(station_data["Longitude"], errors="coerce")
    station_data["Latitude"]  = pd.to_numeric(station_data["Latitude"],  errors="coerce")
    event_data = pd.read_csv("events.csv")
    plateBoundaryNZ_ffp = Path(r"plateboundary2.ll")
    nz_geometry = gpd.read_file('nz_coastlines/nz-coastlines-and-islands-polygons-topo-150k.shp')
    nz_polygon = shapely.MultiPolygon(nz_geometry.geometry)
        
    basin_outlines = {}

    geojson_path = Path('NZVM2p09')

    for regional_model_path in geojson_path.glob('*.geojson'):  # Renamed for clarity basin_name = regional_model_path.stem
        # This is the original outline from GeoJSON, assumed to be WGS84
        polygons_for_regionmask = load_outline(regional_model_path)
        basin_name = regional_model_path.stem
        basin_outlines[basin_name] = polygons_for_regionmask

    # Intersect basins with NZ geometry
    clipped_basins = {}
    for basin_name, polygons in basin_outlines.items():
        clipped_polygons = [shapely.intersection(nz_polygon, p) for p in polygons if not p.is_empty]
        clipped_basins[basin_name] = [p for p in clipped_polygons if not p.is_empty]
    
    # type_map = {
    #     'Canterbury_outline_WGS84': '#ADD8E6',
    #     'Wellington_outline_WGS84': '#48D1CC',
    #     'Nelson_outline_WGS84': '#48D1CC',
    #     'Kaikoura_outline_WGS84': '#4169E1',
    #     'Southland_outline_WGS84_1': '#4169E1',
    #     'Southland_outline_WGS84_2': '#4169E1',
        
    # }


    used_stat_ids = im_obs["sta"].unique()
    used_event_ids = im_obs["evid"].unique()
    stations_used = station_data[station_data["sta"].isin(used_stat_ids)]
    events_used = event_data[event_data["evid"].isin(used_event_ids)]

    # Plot lines between sources and stations
    merged = im_obs.merge(
        station_data[["sta", "Latitude", "Longitude"]],
        on="sta", how="left"
    ).merge(
        event_data[["evid", "hlat", "hlon"]],
        on="evid", how="left"
    )

    for _, row in merged.iterrows():
        fig.plot(
            x=[row["hlon"], row["Longitude"]],
            y=[row["hlat"], row["Latitude"]],
            pen="0.05p,gray70,solid"
        )


      
    type_map = {
        'Canterbury_outline_WGS84': '#ADD8E6',
        'BanksPeninsulaVolcanics_outline_WGS84': '#ADD8E6',
        'Wellington_outline_WGS84': '#48D1CC',  
        'Nelson_outline_WGS84': '#48D1CC',
        'Kaikoura_outline_WGS84': '#4169E1',
        'Southland_outline_WGS84_1': '#4169E1',
        'Southland_outline_WGS84_2': '#4169E1',
    }
    # Plot clipped basins
    for basin_name, polygons in clipped_basins.items():
        print(f"{basin_name}: {len(polygons)} polygons loaded")
        for poly in polygons:
            fill_color = type_map.get(basin_name, '#191970')
            plot_polygon(fig, poly, fill=fill_color, pen="0.1p,black", transparency=20)
        
    # Plot dummy points for legend entries (outside map region)
    # --- Combined legend entries: interleaved so col 1 = Types, col 2 = categories ---
    fig.plot(x=[999], y=[999], style="s0.5c",  fill="#191970", pen="0.2p,black", label="Type 1+N2")
    fig.plot(x=[999], y=[999], style="c0.3c",  fill="#1E90FF", pen="0.2p,black", label="Basin")
    fig.plot(x=[999], y=[999], style="s0.5c",  fill="#4169E1", pen="0.2p,black", label="Type 2")
    fig.plot(x=[999], y=[999], style="d0.35c", fill="#66C266", pen="0.2p,black", label="Basin Edge")
    fig.plot(x=[999], y=[999], style="s0.5c",  fill="#48D1CC", pen="0.2p,black", label="Type 3")
    fig.plot(x=[999], y=[999], style="s0.35c", fill="#9370DB", pen="0.2p,black", label="Valley")
    fig.plot(x=[999], y=[999], style="s0.5c",  fill="#ADD8E6", pen="0.2p,black", label="Type 4")
    fig.plot(x=[999], y=[999], style="t0.35c", fill="#993333", pen="0.2p,black", label="Hill")

    with pygmt.config(FONT_ANNOT_PRIMARY="15p"):
        fig.legend(position="JTR+jTR+o0.2c+w7.4c", box="+gwhite+p0.5p,black,solid")
    

    
    # Plot events as beachballs scaled by magnitude
    for _, row in events_used.iterrows():
        focal_mechanism = {
            "strike": row["strike"],
            "dip": row["dip"],
            "rake": row["rake"],
            "magnitude": row["mag"],
        }
        scale = 0.05* row["mag"]
        fig.meca(
            spec=focal_mechanism,
            scale=f"{scale}c",
            compressionfill = "firebrick",
            pen="0.05p,black,solid",
            longitude=row["hlon"],
            latitude=row["hlat"],
            depth=row["hdepth"],
    )
        
    # Plot stations by category
    category_styles = {
        "Basin": {"symbol": "c0.1c", "fill": "#1E90FF", "pen": "0.2p,black"},
        "Basin Edge": {"symbol": "d0.15c", "fill": "#66C266", "pen": "0.2p,black"},
        "Valley": {"symbol": "s0.15c", "fill": "#9370DB", "pen": "0.2p,black"},
        "Hill": {"symbol": "t0.15c", "fill": "#993333", "pen": "0.2p,black"},
    }

    for category, style in category_styles.items():
      subset = stations_used[stations_used["Geomorphology"] == category].dropna(subset=["Longitude", "Latitude"])
      if len(subset) == 0:
        continue
      fig.plot(x=subset["Longitude"].to_numpy(dtype="float64"),y=subset["Latitude"].to_numpy(dtype="float64"),style=style["symbol"],fill=style["fill"],pen=style["pen"])

    
    # Plot plate boundary
    fig.plot(data = plateBoundaryNZ_ffp, pen="1.0p,black")
    fig.text(text="Australian-Pacific", x=176.9, y=-42.48, justify="LM", font="12p,Helvetica,black")
    fig.text(text="Plate Boundary", x=176.9, y=-42.78, justify="LM", font="12p,Helvetica,black")
    fig.plot(x=177.2, y=-42.35, style="v0.6c+eA", direction=[[105], [0.95]], pen="0.7p,black")
    
    # larger symbols for the zoomed insets
    inset_category_styles = {
        "Basin":      {"symbol": "c0.2c",  "fill": "#1E90FF", "pen": "0.2p,black"},
        "Basin Edge": {"symbol": "d0.25c", "fill": "#66C266", "pen": "0.2p,black"},
        "Valley":     {"symbol": "s0.25c", "fill": "#9370DB", "pen": "0.2p,black"},
        "Hill":       {"symbol": "t0.25c", "fill": "#993333", "pen": "0.2p,black"},
    }
    # ----------------------------------------------------------------------
    # Regional insets (basins + stations only — no events)
    # ----------------------------------------------------------------------
    
    def draw_inset(inset_region, position, label, label_xy, inset_width=7.5):
        sub = stations_used[
            (stations_used["Longitude"] > inset_region[0]) &
            (stations_used["Longitude"] < inset_region[1]) &
            (stations_used["Latitude"]  > inset_region[2]) &
            (stations_used["Latitude"]  < inset_region[3])
        ]
        with fig.inset(position=position, region=inset_region,
                       projection=f"M{inset_width}c", margin=0,
                       box="+p0.7p,black"):
            # same basemap as the main map, drawn into the inset
            plotting.gen_region_fig(
                region=inset_region,
                plot_kwargs={"water_color": "white", "topo_cmap_min": -900,
                             "topo_cmap_max": 3100, "topo_cmap": "gray"},
                plot_highways=False, high_res_topo=False,
                config_options=dict(
                    MAP_FRAME_TYPE="plain", FORMAT_GEO_MAP="ddd",
                    MAP_GRID_PEN="0.5p,gray", MAP_TICK_PEN_PRIMARY="1p,black",
                    MAP_FRAME_PEN="1p,black", MAP_FRAME_AXES="WSne",
                    FONT_ANNOT_PRIMARY="9p,Helvetica,black",
                ),
                fig=fig,
            )
            # basins — same colours as the main map
            for basin_name, polygons in clipped_basins.items():
                for poly in polygons:
                    fill_color = type_map.get(basin_name, "#191970")
                    plot_polygon(fig, poly, fill=fill_color, pen="0.1p,black", transparency=20)
            # stations only (no beachballs)
            for category, style in inset_category_styles.items():
              s = sub[sub["Geomorphology"] == category].dropna(subset=["Longitude", "Latitude"])
              if len(s) == 0:
                continue
              fig.plot(x=s["Longitude"].to_numpy(dtype="float64"),y=s["Latitude"].to_numpy(dtype="float64"),style=style["symbol"],fill=style["fill"],pen=style["pen"])
            if label:
                fig.text(text=label, x=label_xy[0], y=label_xy[1],
                         font="10p,Helvetica-Bold,black")
    
        # rectangle on the main map marking this window
        fig.plot(data=[[inset_region[0], inset_region[2],
                        inset_region[1], inset_region[3]]],
                 style="r+s", pen="2p,black")
    
    
    # Wellington + Marlborough  ->  top-left
    draw_inset([174.68, 175.1, -41.4, -41.1], "jTL","Wellington", (174.4, -40.75))
    #fig.plot(x=[165.0, 174.68], y=[-39.385, -41.4], pen="0.7p,black,--")
    #fig.plot(x=[172.08, 175.1], y=[-34.0, -41.1], pen="0.7p,black,--")
    
    # Kaikoura  ->  bottom-right
    draw_inset([173.51, 173.8, -42.44, -42.28], "jBR","Kaikoura", (173.7, -42.20), inset_width=5)
    draw_inset([173.25, 174.25, -41.94, -41.39], "jBR+o5.5c/0c","Marlborough", (173.6, -41.10),inset_width=5.08)
    #fig.plot(x=[173.31, 173.25], y=[-42.44, -41.94], pen="0.7p,black,--")
    #fig.plot(x=[173.95, 174.25], y=[-48, -41.39], pen="0.7p,black,--") 
    
    #fig.plot(x=[181.0, 173.51], y=[-42.44, -42.485], pen="0.7p,black,--")
    #fig.plot(x=[173.95, 173.8], y=[-48, -42.28], pen="0.7p,black,--")  

                             
    
    # ---- Region Annotations with Arrows ----

    # ---- Region Annotations with Arrows ----
    
    # region_labels = {
    #     "Palmerston North": (-40.1539, 172.4457),
    #     "West Coast": (-42.5398, 168.2808),
    #     "Napier": (-39.6685, 177.2990),
    #     "Kaikōura": (-42.4199, 173.9871),
    #     "Southland": (-47.1132, 168.4708),
    #     "Nelson-Tasman": (-40.7325, 171.3841),
    #     "Wellington": (-41.6432, 174.7506),
    #     "Canterbury": (-43.3913, 172.9954),
    # }
    
    # for name, (lat, lon) in region_labels.items():
    
    #     if name == "Napier":
    #         label_lon = 176.8516
    #         label_lat = -39.5935   
    #         fig.plot(x=[lon, label_lon], y=[lat, label_lat], pen="0.5p,black")
            
    #     elif name == "Palmerston North":
    #         label_lon = 175.4611
    #         label_lat = -40.3444  
    #         fig.plot(x=[175.0510, label_lon], y=[-40.2934, label_lat], pen="0.5p,black")
            
    #     elif name == "Nelson-Tasman":
    #         label_lon = 173.0079  
    #         label_lat = -41.3219 
    #         fig.plot(x=[173.3777, label_lon], y=[-40.8372, label_lat], pen="0.5p,black")
            
    #     elif name == "West Coast":
    #         label_lon = 170.6963
    #         label_lat = -42.9851
    #         fig.plot(x=[169.6708, label_lon], y=[-42.6808, label_lat], pen="0.5p,black")
            
    #     elif name == "Canterbury":
    #         label_lon = 172.5732
    #         label_lat = -43.2083
    #         fig.plot(x=[lon, label_lon], y=[lat, label_lat], pen="0.5p,black")
            
    #     elif name == "Kaikōura":
    #         label_lon = 173.7113
    #         label_lat = -42.4153
    #         fig.plot(x=[lon, label_lon], y=[lat, label_lat], pen="0.5p,black")
            
    #     elif name == "Wellington":
    #         label_lon = 174.8489
    #         label_lat = -41.3147
    #         fig.plot(x=[174.8679, label_lon], y=[-41.5719, label_lat], pen="0.5p,black")
            
    #     elif name == "Southland":
    #         label_lon = 168.4128
    #         label_lat = -46.3300
    #         fig.plot(x=[168.5829, label_lon], y=[-46.9877, label_lat], pen="0.5p,black")
            
    #     fig.text(
    #         x=lon,
    #         y=lat,
    #         text=name,
    #         font="12p,Helvetica,black",
    #         justify="LM"
    #     )
    fig.show()
    print(f"Saving figure to file... {output_path}")
    fig.savefig(output_path, dpi=900, anti_alias=True)


main("Basins_NZVM2p09.png")