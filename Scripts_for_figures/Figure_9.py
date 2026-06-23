from pathlib import Path
import pandas as pd
import numpy as np
import pygmt
from pygmt_helper import plotting

data_ffp = Path("FINAL_GRID_TRUE_Z2p5.csv")
df = pd.read_csv(data_ffp).rename(columns={"Lon": "lon", "Lat": "lat"})
df["Z"] = df["Z2p5_wb"] 
# z1_bins = [50, 75, 100, 200, 300, 500, 1000,1500]
z1_bins = [300, 500, 1000, 1500, 3000, 5000, 7500, 10000,13000]
series_str = ",".join(str(v) for v in z1_bins)
# cpt_path = Path("USE.cpt") # For Z1.0
cpt_path = Path("USE_Z2p5.cpt") # For Z2.5
# if not cpt_path.exists():
#     pygmt.makecpt(
#         cmap="inferno",
#         series=series_str,
#         reverse=True,        
#         continuous=False,   
#         output=str(cpt_path),
#     )
region = [166.00, 179.00, -47.50, -34.00]
fig = plotting.gen_region_fig(
    plot_kwargs={
        "water_color": "white",
        "topo_cmap_min": -900,
        "topo_cmap_max": 3100,
        'topo_cmap': 'gray'
    },
    plot_highways=False,
    region = region,
    plot_topo=True,
    high_res_topo=False,
    config_options=dict(
        MAP_FRAME_TYPE="fancy",
        FORMAT_GEO_MAP="ddd",
        MAP_GRID_PEN="0.5p,gray",
        MAP_TICK_PEN_PRIMARY="1p,black",
        MAP_FRAME_PEN="1p,black",
        MAP_FRAME_AXES="WSne",
        FONT_ANNOT_PRIMARY="14p,Helvetica,black",   
        FONT_LABEL="18p,Helvetica,black",           
    ),
)

grid = plotting.create_grid(df, "Z", grid_spacing="100e/100e")

grid = grid.where(grid >= 300, np.nan)
# grid = grid.where(grid >= 50, np.nan)
fig.grdimage(
    grid=grid,
    cmap=str(cpt_path),
    nan_transparent=True,
    interpolation="c",
)
# fig.text(
#     text="(A) With explicit sedimentary basins",
#     x=0.03, y=0.97,  
#     justify="TL",
#     font="18p,Helvetica-Bold,black",
#     no_clip=True
# )

# label = "Depth to 1.0 km/s (V@-S@-) horizon, Z1.0 (m)"

# fig.colorbar(
#     cmap=str(cpt_path),
#     position="JBC+w12c/0.5c",
#     equalsize=True,              
#     frame=[f"a+l{label}"],       
# )
# fig.colorbar(
#     cmap=str(cpt_path),
#     position="JBC+w12c/0.5c",
#     equalsize=True,
#     frame=["x+lDepth to 1.0 km/s @%V@%@-S@- horizon, Z@-1@-.@-0@- (m)"]

# )


fig.savefig("z_values_pyGMT_wb_TRUE_Z2p5.png", dpi=900, anti_alias=True)