from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import json
from pygmt_helper import plotting
import geopandas as gpd 
from shapely.geometry import Polygon
from workflow.realisations import SourceConfig
from visualisation.realisation import plot_sources

# -------------------------------
# USER INPUTS
# -------------------------------
sim_path = Path('/scratch/jobs/ati47/Ayushi_Models/Paper2_withbasin_runs/')
basin_dir = Path('/scratch/jobs/ati47/Paper2_GMTFigues/NZVM2p09')

out_dir  = Path("Figure_11")
out_dir.mkdir(exist_ok=True)

def lon360(lon):
    return lon % 360.0

def read_domain_corners(realisation_path):
    with open(realisation_path) as f:
        data = json.load(f)
    pts = data["domain"]["domain"]
    lons = [lon360(p["longitude"]) for p in pts]
    lats = [p["latitude"]          for p in pts]
    lons.append(lons[0])           # close the ring
    lats.append(lats[0])
    return lons, lats






def plot_basins(fig, basin_dir, domain_poly=None, pen="0.3p,black"):
    """Thin basin outlines, optionally clipped to the simulation domain."""
    for gj in sorted(Path(basin_dir).glob("*.geojson")):
        gdf = gpd.read_file(gj)
        if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs(4326)
        if domain_poly is not None:
            gdf = gdf[gdf.intersects(domain_poly)]   # keep only in-domain basins
            if gdf.empty:
                continue
        fig.plot(data=gdf, pen=pen)
        
IM_array = [
#    "PGA",
#    "PGV",
#    "pSA_0.2",
#    "pSA_1.5",
"pSA_2.0",
#"Ds595",
]

component = "rotd50"
main_region = (165, 180, -48, -33)


def extract_sim_IMs(dset, IM, component):

    if IM.startswith("pSA"):
        period = float(IM.split("_")[1])

        psa = dset["pSA"]

    
        data = psa.sel(
            component=component,
            period=period
        )

    else:
        data = dset[IM].sel(component=component)

    return data


# -------------------------------
# REGION FUNCTION
# -------------------------------
def auto_region_from_points(lon, lat, pad_deg=0.0, min_span_deg=1.5, clamp=None):

    xmin, xmax = np.min(lon), np.max(lon)
    ymin, ymax = np.min(lat), np.max(lat)

    if (xmax - xmin) < min_span_deg:
        mid = 0.5 * (xmin + xmax)
        xmin = mid - min_span_deg / 2
        xmax = mid + min_span_deg / 2

    if (ymax - ymin) < min_span_deg:
        mid = 0.5 * (ymin + ymax)
        ymin = mid - min_span_deg / 2
        ymax = mid + min_span_deg / 2

    xmin -= pad_deg
    xmax += pad_deg
    ymin -= pad_deg
    ymax += pad_deg

    if clamp:
        xmin = max(xmin, clamp[0])
        xmax = min(xmax, clamp[1])
        ymin = max(ymin, clamp[2])
        ymax = min(ymax, clamp[3])

    return (xmin, xmax, ymin, ymax)


# -------------------------------
# MAIN LOOP
# -------------------------------
event_list = [f.name for f in sim_path.iterdir() if f.is_dir()]
print("Event list:", event_list)

#event_list = ['2023p310616']
event_list = ['2022p901216']
#event_list = ['2015p012816']
for evid in event_list:

    print(f"\nProcessing event: {evid}")

    sim_ims = xr.open_dataset(sim_path / evid / "intensity_measures.h5")
    realisation_path = sim_path / evid / "realisation.json"   # results folder
    dom_lon, dom_lat = read_domain_corners(realisation_path)

    lats = sim_ims.latitude.values
    lons = sim_ims.longitude.values

    #outdir = out_dir / evid
    outdir = out_dir
    outdir.mkdir(exist_ok=True)

    # -------------------------------
    # EVENT REGION
    # -------------------------------
    finite_xy = np.isfinite(lons) & np.isfinite(lats)

    event_region = auto_region_from_points(
        lons[finite_xy],
        lats[finite_xy],
        pad_deg=0.5,
        min_span_deg=1.8,
        clamp=main_region,
    )

    # -------------------------------
    # LOOP IMs
    # -------------------------------
    for IM in IM_array:

        print(f"  Plotting {IM}")

        da = extract_sim_IMs(sim_ims, IM, component)

        df = (da.to_dataframe().reset_index()[["longitude", "latitude", da.name]].rename(columns={"longitude": "lon", "latitude": "lat", da.name: "value"}))
        df = df[df["value"] > 0]
        df["logval"] = np.log10(df["value"])
        grid = plotting.create_grid(df, "logval", grid_spacing="100e/100e",
                                    region=event_region, interp_method="linear",
                                    set_water_to_nan=True)
        # fixed across ALL events: 1e-3 to 1e-1 g

        
        # -------------------------------
        # CREATE GRID
        # -------------------------------
        # grid = plotting.create_grid(
        #     df,
        #     "value",
        #     grid_spacing="100e/100e",  
        #     region=event_region,
        #     interp_method="linear",
        #     set_water_to_nan=True,
        # )
        
        mask_threshold = -3.5
        grid = grid.where(grid >= mask_threshold)

        
        # -------------------------------
        # CREATE BASE MAP
        # -------------------------------
        fig = plotting.gen_region_fig(
            region=event_region,
            plot_kwargs={
                "water_color": "white",
                "topo_cmap_min": -2000,
                "topo_cmap_max": 7000,
                "topo_cmap": "gray",
            },
            plot_highways=False,
            plot_topo=True,
            high_res_topo=False,
        )
        plotting.plot_grid(fig, grid, cmap="hot", cmap_limits=(-3.5, -1.5, 0.25), cmap_limit_colors=("white", "black"), continuous_cmap = True, reverse_cmap=True, cb_label="log@-10@- SA 2.0 s (g)", transparency = 20, plot_contours=False)
        # -------------------------------
        # COLOR SCALE
        # -------------------------------
        # vmin = mask_threshold
        # #vmax = 0.01
        # #inc = (vmax - vmin) / 50

        
        # -------------------------------
        # PLOT GRID
        # -------------------------------
        # plotting.plot_grid(
        #     fig,
        #     grid,
        #     cmap="hot",
        #     cmap_limits=(vmin, vmax, inc),
        #     #cmap_limits = (vmin, vmax, 5),
        #     cmap_limit_colors=("white", "black"),
        #     cb_label="SA 2.0 s (g)",
        #     #cb_label = "Ds5-95 (s)",
        #     reverse_cmap=True,
        #     transparency=50,
        #     plot_contours=False,
        # )
        domain_poly = Polygon(zip(dom_lon, dom_lat))  
        plot_basins(fig, basin_dir, domain_poly = domain_poly, pen="0.5p,black")
        fig.plot(x=dom_lon, y=dom_lat, pen="1.5p,black")
        source_config = SourceConfig.read_from_realisation(realisation_path)
        #plot_sources(fig, source_config, pen="1.5p,black")
        fig.plot(x=[sim_ims.attrs["hypo_lon"]],y=[sim_ims.attrs["hypo_lat"]],style="a0.6c",fill="white",pen="1.5p,black")
        # -------------------------------
        # SAVE
        # -------------------------------
        outfile = outdir / f"{evid}_{IM.replace('.', 'p')}_grid_CS_EP2020_Paper2.png"
        xs_lon1, xs_lat1 = 174.444381, -39.156606  
        xs_lon2, xs_lat2 = 177.155332, -39.720282  
        
        # keep the same longitude convention as the rest of the map
        xs_lon1, xs_lon2 = lon360(xs_lon1), lon360(xs_lon2)
        
        # the line
        fig.plot(x=[xs_lon1, xs_lon2], y=[xs_lat1, xs_lat2], pen="2p,red")
        
        # end-point dots (optional)
        fig.plot(x=[xs_lon1, xs_lon2], y=[xs_lat1, xs_lat2],
                 style="c0.15c", fill="red", pen="0.5p,black")
        
        # A / A' labels nudged off the ends
        fig.text(x=xs_lon1, y=xs_lat1, text="A",  font="14p,Helvetica-Bold,red",
                 justify="BR", offset="-0.15c/0.15c")
        fig.text(x=xs_lon2, y=xs_lat2, text="A'", font="14p,Helvetica-Bold,red",
                 justify="BL", offset="0.15c/0.15c")

        fig.savefig(outfile,dpi=900)

        print(f"Saved {outfile}")