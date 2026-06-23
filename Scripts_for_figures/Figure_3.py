# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 08:59:51 2024

@author: ati47
"""

import numpy as np
import pandas as pd
from pathlib import Path
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import addcopyfighandler
import shapely
import geopandas as gpd
import pyproj

Features = pd.read_csv(Path(r"C:\Users\ati47\OneDrive - University of Canterbury\Desktop\PhD\10. Research\Objective 1\9. Model development\Correlation\Features.csv"))
Features_new = pd.read_csv(Path(r"C:\Users\ati47\OneDrive - University of Canterbury\Desktop\PhD\10. Research\Objective 2\1. Datasets\LATEST_NZGMDBv_4.3_Categorizations.csv"))
save_dir = Path(r"C:\Users\ati47\OneDrive - University of Canterbury\Desktop\PhD\10. Research\Objective 2\4. Analysis")
#%% Thomson Vs Tiwari - Lee et al. (2022)
desired_basin_order = ['Non-Basin', 'Unmodeled Basin', 'Type 1 Basin','Type 3 Basin', 'Type 4 Basin']
counts_original = Features['Basin Type'].value_counts().reindex(desired_basin_order)
counts_latest = Features['Basin Type (NZVM Latest)'].value_counts().reindex(desired_basin_order)
# Combine into a single DataFrame
counts_df = pd.concat([counts_original, counts_latest], axis=1)
counts_df.columns = ['Thomson et al. (2020)', 'This study']
counts_df = counts_df.fillna(0)  
fig,ax = plt.subplots(figsize=(11.64,7.81),constrained_layout=True)
counts_df.plot(kind='bar', width=0.8,color=['r','b'],ax=ax)
ax.set_ylabel("Number of Stations",fontsize=20)
ax.set_xlabel("Basin Type Category",fontsize=20)
ax.set_xticklabels(ax.get_xticklabels(),rotation=0,ha='center',fontsize=16)
ax.legend(fontsize=20,loc = 'upper right')
ax.tick_params(labelsize=16)
plt.savefig(os.path.join(save_dir,"ThomsonVsTiwari_Lee.png"),dpi=300)
changed = counts_df[counts_df['Thomson et al. (2020)'] != counts_df['This study']]
fig, ax = plt.subplots(figsize=(11.64,7.81), constrained_layout=True)
changed.plot(kind='bar', width=0.8, color=['r','b'], ax=ax)
ax.set_ylabel("Number of Stations", fontsize=20)
ax.set_xlabel("Basin Type Category", fontsize=20)
ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha='center', fontsize=16)
ax.legend(fontsize=20,loc = 'upper right')
ax.tick_params(labelsize=16)
plt.savefig(os.path.join(save_dir, "ThomsonVsTiwari_Lee_Changes.png"), dpi=300)
#%% Thomson Vs Tiwari - NZGMDB
desired_basin_order = ['Non-Basin', 'Unmodeled', 'Type 1','Type 2','Type 3', 'Type 4']
counts_original_NZGMDB = Features_new['BasinType_2p02'].value_counts().reindex(desired_basin_order)
counts_latest_NZGMDB = Features_new['BasinType_2p09'].value_counts().reindex(desired_basin_order)
# Combine into a single DataFrame
counts_df = pd.concat([counts_original_NZGMDB, counts_latest_NZGMDB], axis=1)
counts_df.columns = ['Thomson et al. (2020) - NZVM 2.0', 'This study - NZVM2.09']
counts_df = counts_df.fillna(0)  
fig,ax = plt.subplots(figsize=(11.64,7.81),constrained_layout=True)
counts_df.plot(kind='bar', width=0.6,color=['r','b'],ax=ax)
ax.set_ylabel("Number of Stations",fontsize=20)
ax.set_xlabel("Basin Type Category",fontsize=20)
ax.set_xticklabels(ax.get_xticklabels(),rotation=0,ha='center',fontsize=16)
ax.legend(fontsize=20,loc = 'upper right')
ax.tick_params(labelsize=16)
plt.savefig(os.path.join(save_dir,"ThomsonVsTiwari_NZGMDB.png"),dpi=300)
changed = counts_df[counts_df['Thomson et al. (2020) - NZVM 2.0'] != counts_df['This study - NZVM2.09']]
fig, ax = plt.subplots(figsize=(11.64,7.81), constrained_layout=True)
changed.plot(kind='bar', width=0.6, color=['r','b'], ax=ax)
ax.set_ylabel("Number of Stations", fontsize=20)
ax.set_xlabel("Basin Type Category", fontsize=20)
ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha='center', fontsize=16)
ax.legend(fontsize=20,loc = 'upper right')
ax.tick_params(labelsize=16)
plt.savefig(os.path.join(save_dir, "ThomsonVsTiwari_NZGMDB_Changes.png"), dpi=300)
#%% Population Datasets - Just Print
transformer = pyproj.Transformer.from_crs(4326, 2193, always_xy=True)
BASIN_PATH = Path(r'C:\Users\ati47\OneDrive - University of Canterbury\Desktop\PhD\10. Research\Journal papers\Paper 1\Figures\GMT Map Figures\NZVM2p09')
basins: dict[str, shapely.Geometry] = {}
for basin_path in BASIN_PATH.glob('*.geojson'):
    geometry_latlon = shapely.from_geojson(basin_path.read_text())
    geometry = shapely.transform(geometry_latlon, lambda coords: np.array(transformer.transform(coords[:, 0], coords[:, 1])).T)
    basins[basin_path.stem] = geometry

gdf = gpd.read_file(r'Stats NZ\new-zealand-estimated-resident-population-grid-250-metre.shp')
total = gdf['PopEst2023'].sum()
for basin_name, basin in basins.items():
    in_gdf = gdf.loc[gdf.intersects(basin)]
    in_basin_population = in_gdf["PopEst2023"].sum()
    print(f'Number of people living in {basin_name} is roughly {round(in_basin_population,2)} ({round(in_basin_population / total * 100,2)}%)')

#%% Population Datasets - CSV
transformer = pyproj.Transformer.from_crs(4326, 2193, always_xy=True)
BASIN_PATH = Path(r'C:\Users\ati47\OneDrive - University of Canterbury\Desktop\PhD\10. Research\Journal papers\Paper 1\Figures\GMT Map Figures\NZVM2p09')
basins: dict[str, list] = {}

for basin_path in BASIN_PATH.glob('*.geojson'):
    geometry_latlon = shapely.from_geojson(basin_path.read_text())
    geometry = shapely.transform(
        geometry_latlon,
        lambda coords: np.array(transformer.transform(coords[:, 0], coords[:, 1])).T
    )
    
    # Extract base name before "_outline"
    base_name = basin_path.stem.split('_outline')[0]
    
    if base_name not in basins:
        basins[base_name] = []
    basins[base_name].append(geometry)

# Load population grid
gdf = gpd.read_file(r'Stats NZ\new-zealand-estimated-resident-population-grid-250-metre.shp')

# Calculate populations
results = []
for basin_name, geometries in basins.items():
    total_pop = 0
    for geom in geometries:
        in_gdf = gdf.loc[gdf.intersects(geom)]
        total_pop += in_gdf["PopEst2023"].sum()
    
    results.append({'Basin name': basin_name, 'Population': round(total_pop, 2)})

# Save to CSV
df_results = pd.DataFrame(results)
df_results.to_csv('basin_populations.csv', index=False)
#%% Population Histogram - Numbers
pop_df = pd.read_csv(Path(r"Populations_Basins.csv"))
desired_basin_order = ['Type 1','Type 2','Type 3', 'Type 4']
pop_df.columns = pop_df.columns.str.strip()
pop_by_type = pop_df.groupby('Type')['Population'].sum().reindex(desired_basin_order)
pop_by_type = pop_by_type.fillna(0)
pop_by_type_millions = pop_by_type / 1e6
fig, ax = plt.subplots(figsize=(11.64,7.81), constrained_layout=True)
pop_by_type_millions.plot(kind='bar', width=0.2, ax=ax)
ax.set_ylabel("Population (millions)", fontsize=28)
ax.set_xlabel("Basin Type", fontsize=28)
ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha='center', fontsize=28)
ax.tick_params(labelsize=28)
plt.savefig(os.path.join(save_dir, "Population_By_BasinType.png"), dpi=300)
#%% Population Histogram - By percentage
pop_by_type_percent = (pop_by_type / pop_by_type.sum()) * 100
fig, ax = plt.subplots(figsize=(11.64,7.81), constrained_layout=True)
pop_by_type_percent.plot(kind='bar', width=0.8, color=["#191970","#4169E1","#48D1CC","#ADD8E6"], ax=ax)
ax.set_ylabel("Population (%)", fontsize=20)
ax.set_xlabel("Basin Type", fontsize=20)
ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha='center', fontsize=16)
ax.tick_params(labelsize=16)
plt.savefig(os.path.join(save_dir, "Population_By_BasinType_percent.png"), dpi=300)
#%% Population Histogram - By percentage - From country
pop_by_type_percent = (pop_by_type / total) * 100
fig, ax = plt.subplots(figsize=(11.64,7.81), constrained_layout=True)
pop_by_type_percent.plot(kind='bar', width=0.8, color=["#191970","#4169E1","#48D1CC","#ADD8E6"], ax=ax)
ax.set_ylabel("Population (%)", fontsize=20)
ax.set_xlabel("Basin Type", fontsize=20)
ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha='center', fontsize=16)
ax.tick_params(labelsize=16)
plt.savefig(os.path.join(save_dir, "Population_By_BasinType_percent_allcountry.png"), dpi=300)
non_basinpop = total - (pop_by_type['Type 1']+pop_by_type['Type 2']+pop_by_type['Type 3']+pop_by_type['Type 4'])
non_basinpercent = non_basinpop/total*100
#%% Poster Figure
desired_basin_order = ['Non-Basin', 'Unmodeled', 'Type 1','Type 2','Type 3', 'Type 4']
counts_original_NZGMDB = Features_new['BasinType_2p02'].value_counts().reindex(desired_basin_order)
counts_latest_NZGMDB = Features_new['BasinType_2p09'].value_counts().reindex(desired_basin_order)

counts_df = pd.concat([counts_original_NZGMDB, counts_latest_NZGMDB], axis=1)
counts_df.columns = ['Thomson et al. (2020) - NZVMv2.0', 'This study - NZVMv2.09']
counts_df = counts_df.fillna(0)

# total for proportion
counts_prop = counts_df.div(counts_df.sum(axis=0), axis=1)

# --- subplot (b) data ---
pop_df = pd.read_csv(Path(r"Populations_Basins.csv"))
desired_basin_order_b = ['Type 1','Type 2','Type 3', 'Type 4']
pop_df.columns = pop_df.columns.str.strip()
pop_by_type = pop_df.groupby('Type')['Population'].sum().reindex(desired_basin_order_b)
pop_by_type_millions = pop_by_type.fillna(0) / 1e6

# --- figure setup ---
cm_to_inch = 1/2.54
fig = plt.figure(figsize=(49*cm_to_inch, 25*cm_to_inch), constrained_layout=True)
gs = fig.add_gridspec(1, 2, width_ratios=[2.5, 2])  # subplot (a) wider

fontsize_labels = 28
fontsize_ticks = 24
fontsize_legend = 28

ax1 = fig.add_subplot(gs[0,0])
counts_prop.plot(kind='bar', width=0.5, color=['r','b'], ax=ax1)

ax1.set_ylabel("Proportion of stations", fontsize=fontsize_labels)
ax1.set_xlabel("Basin Type", fontsize=fontsize_labels)
ax1.set_ylim(0, 1)
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=0, ha='center', fontsize=fontsize_ticks)
ax1.tick_params(labelsize=fontsize_ticks)
ax1.legend(fontsize=fontsize_legend, loc='upper right')

# --- add counts above bars ---
for i, cat in enumerate(desired_basin_order):
    # get both counts
    c1 = counts_df.iloc[i, 0]
    c2 = counts_df.iloc[i, 1]
    
    # bar positions (center between the two bars of the cluster)
    bar1 = ax1.containers[0][i]
    bar2 = ax1.containers[1][i]
    x_center = (bar1.get_x() + bar2.get_x() + bar2.get_width())/2
    y_max = max(bar1.get_height(), bar2.get_height())

    if c1 == c2:  # equal counts → show only once
        ax1.text(x_center, y_max + 0.02, f"{int(c1)}",
                 ha='center', va='bottom', fontsize=fontsize_ticks)
    else:  # unequal counts → keep both separately
        ax1.text(bar1.get_x() + bar1.get_width()/2, bar1.get_height() + 0.02,
                 f"{int(c1)}", ha='center', va='bottom', fontsize=fontsize_ticks)
        ax1.text(bar2.get_x() + bar2.get_width()/2, bar2.get_height() + 0.02,
                 f"{int(c2)}", ha='center', va='bottom', fontsize=fontsize_ticks)

ax1.text(-0.05, 1.0, "(a)", transform=ax1.transAxes,
         fontsize=fontsize_labels, fontweight="bold", va="bottom", ha="right")

ax2 = fig.add_subplot(gs[0,1])

# x positions
x = range(len(pop_by_type_millions))
y = pop_by_type_millions.values

# draw lollipops
ax2.vlines(x, 0, y, color="black", linewidth=2)
ax2.scatter(x, y, color="black", s=100, zorder=3)

# formatting
ax2.set_xticks(x)
ax2.set_xticklabels(pop_by_type_millions.index, rotation=0, ha='center', fontsize=fontsize_ticks)
ax2.set_ylabel("Population (millions)", fontsize=fontsize_labels)
ax2.set_xlabel("Basin Type", fontsize=fontsize_labels)
ax2.tick_params(labelsize=fontsize_ticks)

ax2.text(-0.05, 1.0, "(b)", transform=ax2.transAxes,
         fontsize=fontsize_labels, fontweight="bold", va="bottom", ha="right")
ax2.text(0.2, 0.88, "49% of NZ population\nin modelled basins",
         transform=ax2.transAxes, fontsize=fontsize_labels,
         ha="left", va="bottom", fontweight="bold")
plt.savefig(os.path.join(save_dir,"Poster_Figure.png"), dpi=300)
#%% Paper 2 Figure - New
from matplotlib.lines import Line2D
desired_basin_order = ['Non-Basin', 'Unmodeled', 
                       'Type 1', 'Type 2', 'Type 3', 'Type 4']

# Station counts
counts_original = Features_new['BasinType_2p02'].value_counts().reindex(desired_basin_order)
counts_latest   = Features_new['BasinType_2p09'].value_counts().reindex(desired_basin_order)

counts_df = pd.concat([counts_original, counts_latest], axis=1)
counts_df.columns = ['Thomson et al. (2020) - NZVMv2.02',
                     'This study - NZVMv2.09']
counts_df = counts_df.fillna(0)

counts_prop = counts_df.div(counts_df.sum(axis=0), axis=1)

# Population
pop_df = pd.read_csv(Path("Populations_Basins.csv"))
pop_df.columns = pop_df.columns.str.strip()

pop_by_type = pop_df.groupby('Type')['Population'].sum()

pop_by_type = pop_by_type.reindex(desired_basin_order)
pop_millions = pop_by_type.fillna(0) / 1e6


cm_to_inch = 1/2.54
fig, ax1 = plt.subplots(figsize=(10.03,6.86),constrained_layout = True)

fontsize_labels = 18
fontsize_ticks = 14
fontsize_legend = 16

# Bar plot (station proportions)
counts_prop.plot(kind='bar',
                 width=0.6,
                 color=['red','blue'],
                 ax=ax1)

ax1.set_ylabel("Proportion of stations", fontsize=fontsize_labels)
ax1.set_xlabel("Basin Type", fontsize=fontsize_labels)
ax1.set_ylim(0, 1)
ax1.set_xticklabels(desired_basin_order, rotation=0, fontsize=fontsize_ticks)
ax1.tick_params(labelsize=fontsize_ticks)

# Add station count labels
label_offset_default = 0.015

for i, cat in enumerate(desired_basin_order):
    c1 = counts_df.iloc[i, 0]
    c2 = counts_df.iloc[i, 1]

    bar1 = ax1.containers[0][i]  # red
    bar2 = ax1.containers[1][i]  # blue

    x_center = (bar1.get_x() + bar2.get_x() + bar2.get_width()) / 2
    y_max = max(bar1.get_height(), bar2.get_height())

    # default offset
    off = label_offset_default

    # special handling for Type 2: push labels upward (and slightly apart if needed)
    if cat == "Type 2":
        off = 0.04

    if c1 == c2:
        ax1.text(x_center, y_max + off, f"{int(c1)}",
                 ha='center', va='bottom', fontsize=fontsize_ticks)
    else:
        if cat == "Type 2":
            # separate the two numbers slightly so they don't collide with the dot
            ax1.text(bar1.get_x() + bar1.get_width()/2 - 0.06,
                     bar1.get_height() + off, f"{int(c1)}",
                     ha='center', va='bottom', fontsize=fontsize_ticks)
            ax1.text(bar2.get_x() + bar2.get_width()/2 + 0.06,
                     bar2.get_height() + off, f"{int(c2)}",
                     ha='center', va='bottom', fontsize=fontsize_ticks)
        else:
            ax1.text(bar1.get_x() + bar1.get_width()/2,
                     bar1.get_height() + off, f"{int(c1)}",
                     ha='center', va='bottom', fontsize=fontsize_ticks)
            ax1.text(bar2.get_x() + bar2.get_width()/2,
                     bar2.get_height() + off, f"{int(c2)}",
                     ha='center', va='bottom', fontsize=fontsize_ticks)


ax2 = ax1.twinx()

# Get cluster centers for dot alignment
blue_bar_centers = []

for i in range(len(desired_basin_order)):
    blue_bar = ax1.containers[1][i]  # 0 = red, 1 = blue
    center = blue_bar.get_x() + blue_bar.get_width() / 2
    blue_bar_centers.append(center)

# Add vertical stems + dots
for xc, yp in zip(blue_bar_centers, pop_millions.values):
    # ax2.vlines(xc, 0, yp, color='black', linewidth=1.2)
    ax2.scatter(xc, yp, color='black', s=80, zorder=5)

ax2.set_ylabel("Population (millions)", fontsize=fontsize_labels)
ax2.tick_params(axis='y', labelsize=fontsize_ticks)
ax2.set_ylim(0, 2.0)
# -------------------------------
# Legend
# -------------------------------

dot_legend = Line2D([0], [0],
                    marker='o',
                    color='w',
                    markerfacecolor='black',
                    markersize=11,
                    label='Population (v2.09 basins)')

handles, labels = ax1.get_legend_handles_labels()
handles.append(dot_legend)
labels.append("Population (v2.09 basins)")

ax1.legend(handles, labels,
           fontsize=fontsize_legend,
           loc='upper right',
           frameon=True)

plt.savefig(os.path.join(save_dir, "Paper2DataPop_Combined.pdf"), dpi=900)
#%% Paper 2 Figure
desired_basin_order = ['Non-Basin', 'Unmodeled', 'Type 1','Type 2','Type 3', 'Type 4']
counts_original_NZGMDB = Features_new['BasinType_2p02'].value_counts().reindex(desired_basin_order)
counts_latest_NZGMDB = Features_new['BasinType_2p09'].value_counts().reindex(desired_basin_order)

counts_df = pd.concat([counts_original_NZGMDB, counts_latest_NZGMDB], axis=1)
counts_df.columns = ['Thomson et al. (2020) - NZVMv2.02', 'This study - NZVMv2.09']
counts_df = counts_df.fillna(0)


counts_prop = counts_df.div(counts_df.sum(axis=0), axis=1)


pop_df = pd.read_csv(Path(r"Populations_Basins.csv"))
desired_basin_order_b = ['Non-Basin','Unmodeled','Type 1','Type 2','Type 3', 'Type 4']
pop_df.columns = pop_df.columns.str.strip()

pop_by_type['Unmodeled'] = 1836000
total_without_hills = pop_by_type.sum()
pop_by_type['Non-Basin'] = 5122600 - total_without_hills
pop_by_type = pop_df.groupby('Type')['Population'].sum().reindex(desired_basin_order_b)

# pop_by_type_millions = pop_by_type.fillna(0) / 1e6


cm_to_inch = 1/2.54
fig = plt.figure(figsize=(49*cm_to_inch, 25*cm_to_inch), constrained_layout=True)
gs = fig.add_gridspec(1, 2, width_ratios=[3, 2])  # subplot (a) wider

fontsize_labels = 28
fontsize_ticks = 21
fontsize_legend = 28

ax1 = fig.add_subplot(gs[0,0])
counts_prop.plot(kind='bar', width=0.5, color=['r','b'], ax=ax1)

ax1.set_ylabel("Proportion of stations", fontsize=fontsize_labels)
ax1.set_xlabel("Basin Type", fontsize=fontsize_labels)
ax1.set_ylim(0, 1)
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=0, ha='center', fontsize=fontsize_ticks)
ax1.tick_params(labelsize=fontsize_ticks)
ax1.legend(fontsize=fontsize_legend, loc='upper right')

for i, cat in enumerate(desired_basin_order):
    # get both counts
    c1 = counts_df.iloc[i, 0]
    c2 = counts_df.iloc[i, 1]
    
    # bar positions (center between the two bars of the cluster)
    bar1 = ax1.containers[0][i]
    bar2 = ax1.containers[1][i]
    x_center = (bar1.get_x() + bar2.get_x() + bar2.get_width())/2
    y_max = max(bar1.get_height(), bar2.get_height())

    if c1 == c2:  # equal counts → show only once
        ax1.text(x_center, y_max + 0.02, f"{int(c1)}",
                 ha='center', va='bottom', fontsize=fontsize_ticks)
    else:  # unequal counts → keep both separately
        ax1.text(bar1.get_x() + bar1.get_width()/2, bar1.get_height() + 0.02,
                 f"{int(c1)}", ha='center', va='bottom', fontsize=fontsize_ticks)
        ax1.text(bar2.get_x() + bar2.get_width()/2, bar2.get_height() + 0.02,
                 f"{int(c2)}", ha='center', va='bottom', fontsize=fontsize_ticks)

ax1.text(-0.05, 1.0, "(a)", transform=ax1.transAxes,
         fontsize=fontsize_labels, fontweight="bold", va="bottom", ha="right")

ax2 = fig.add_subplot(gs[0,1])


x = range(len(pop_by_type_millions))
y = pop_by_type_millions.values


ax2.vlines(x, 0, y, color="black", linewidth=2)
ax2.scatter(x, y, color="black", s=100, zorder=3)

# formatting
ax2.set_xticks(x)
ax2.set_xticklabels(pop_by_type_millions.index, rotation=0, ha='center', fontsize=fontsize_ticks)
ax2.set_ylabel("Population (millions)", fontsize=fontsize_labels)
ax2.set_xlabel("Basin Type", fontsize=fontsize_labels)
ax2.tick_params(labelsize=fontsize_ticks)

ax2.text(-0.05, 1.0, "(b)", transform=ax2.transAxes,
         fontsize=fontsize_labels, fontweight="bold", va="bottom", ha="right")
# ax2.text(0.2, 0.88, "49% of NZ population\nin modelled basins",
#          transform=ax2.transAxes, fontsize=fontsize_labels,
#          ha="left", va="bottom", fontweight="bold")
plt.savefig(os.path.join(save_dir,"Paper2DataPop.pdf"))
#%% Z1.0 before vs after plots for the stations unmodelled in NZVM2p02 to NZVM2p08
Z1diff= pd.read_csv("Z1_Calculations/NZGMDB_Stats_Thresholds_SQ.csv")
x = Z1diff["Z1_nb"]*1000
y = Z1diff["Z1_wb"]*1000
geomorph_colors = {
    "Basin": "#FF5733",
    "Basin-edge": "#33FFBD",
    "Valley": "#9B59B6",
    "Hill": "maroon"
}
scatter_objects = []
fig, ax = plt.subplots(figsize=(11.64,7.81), constrained_layout=True)
for geom, color in geomorph_colors.items():
    subset = Z1diff[Z1diff["Geomorphology"] == geom]
    sc = ax.scatter(subset["Z1_nb"]*1000, subset["Z1_wb"]*1000, edgecolor="k", s=60, color=color, label=geom,picker=True, pickradius=5)
    scatter_objects.append((sc, subset["Station"].values))

# 1:1 line
lims = [min(x.min(), y.min()), max(x.max(), y.max())]
ax.plot(lims, lims, "r--", linewidth=2, label="1:1 line")

# Formatting consistent with your style
ax.set_ylabel("Z1.0 - with sedimentary basins (m)", fontsize=20)
ax.set_xlabel("Z1.0 - without sedimentary basins (m)", fontsize=20)
ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha='center', fontsize=16)
ax.set_yticklabels(ax.get_yticklabels(), fontsize=16)
ax.legend(fontsize=20, loc="best")
ax.tick_params(labelsize=16,direction='in',axis='both',which='both')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xticks([20,50,100, 200, 300, 500, 1000])
ax.set_xticklabels([20,50,100, 200, 300, 500, 1000], rotation=0, ha='center', fontsize=16)
ax.set_yticks([20,50,100, 200, 300, 500, 1000])
ax.set_yticklabels([20,50,100, 200, 300, 500, 1000], rotation=90, ha='center', fontsize=16)
ax.tick_params(axis='y', pad=10)
def onpick(event):
    artist = event.artist
    mouse_ind = event.ind  # index of clicked point

    for sc, station_ids in scatter_objects:
        if artist == sc:
            clicked_ids = station_ids[mouse_ind]
            print("Clicked station(s):", clicked_ids)

fig.canvas.mpl_connect('pick_event', onpick)

plt.savefig(os.path.join(save_dir, "Z1.0_BeforeVsAfter.png"), dpi=300)
#%% Z1.0 before vs after plots for the stations unmodelled in NZVM2p02 to NZVM2p08
Z1diff= pd.read_csv("Zdepths_NZVMv2p02.csv")
x = Z1diff["Z1.0_Before"]
y = Z1diff["Z1_2p09"]*1000
geomorph_colors = {
    "Basin": "#FF5733",
    "Basin-edge": "#33FFBD",
    "Valley": "#9B59B6"
}

fig, ax = plt.subplots(figsize=(11.64,7.81), constrained_layout=True)
for geom, color in geomorph_colors.items():
    subset = Z1diff[Z1diff["Geomorphology"] == geom]
    ax.scatter(subset["Z1.0_Before"], subset["Z1.0_After"], edgecolor="k", s=60, color=color, label=geom)

# 1:1 line
lims = [min(x.min(), y.min()), max(x.max(), y.max())]
ax.plot(lims, lims, "r--", linewidth=2, label="1:1 line")

# Formatting consistent with your style
ax.set_ylabel("Z1.0 After (m)", fontsize=20)
ax.set_xlabel("Z1.0 Before (m)", fontsize=20)
ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha='center', fontsize=16)
ax.set_yticklabels(ax.get_yticklabels(), fontsize=16)
ax.legend(fontsize=20, loc="best")
ax.tick_params(labelsize=16)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xticks([20,50,100, 200, 300, 400, 500, 1000])
ax.set_xticklabels([20,50,100, 200, 300, 400, 500, 1000], rotation=0, ha='center', fontsize=16)
ax.set_yticks([20,50,100, 200, 300, 400, 500, 1000])
ax.set_yticklabels([20,50,100, 200, 300, 400, 500, 1000], rotation=90, ha='center', fontsize=16)
# plt.savefig(os.path.join(save_dir, "Z1.0_BeforeVsAfter.png"), dpi=300)
#%% Z1.0 database Vs. NZVM plots for all the stations 
x = Features_new["Z1.0"]
y = Features_new["Z1_NZVM2p08"]
# geomorph_colors = {
#     "Basin": "#FF5733",
#     "Basin-edge": "#33FFBD",
#     "Valley": "#9B59B6"
# }
geomorph_colors = { "Non-Basin": '#800000', "Unmodeled Basin": '#FF5733', "Type 1 Basin": '#33FFBD', "Type 2 Basin": '#9B59B6', "Type 3 Basin": '#1F618D', "Type 4 Basin": '#F4D03F'}

fig, ax = plt.subplots(figsize=(11.64,7.81), constrained_layout=True)
for geom, color in geomorph_colors.items():
    subset = Features_new[Features_new["Basin Type Latest"] == geom]
    ax.scatter(subset["Z1.0"], subset["Z1_NZVM2p08"], edgecolor="k", s=60, color=color, label=geom)

# 1:1 line
lims = [min(x.min(), y.min()), max(x.max(), y.max())]
ax.plot(lims, lims, "r--", linewidth=2, label="1:1 line")

# Formatting consistent with your style
ax.set_ylabel("Z1.0 NZVM2p08 (m)", fontsize=20)
ax.set_xlabel("Z1.0 Database (m)", fontsize=20)
ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha='center', fontsize=16)
ax.set_yticklabels(ax.get_yticklabels(), fontsize=16)
ax.legend(fontsize=20, loc="best")
ax.tick_params(labelsize=16)

plt.savefig(os.path.join(save_dir, "Z1.0_DatabaseVsNZVM.png"), dpi=300)
#%% Z1 Vs. Vs30 - All NZGM4p3 stations
basin_colors = { "Non-Basin": '#800000', "Unmodeled Basin": '#FF5733', "Type 1 Basin": '#33FFBD', "Type 2 Basin": '#9B59B6', "Type 3 Basin": '#1F618D', "Type 4 Basin": '#F4D03F'}
fig, ax = plt.subplots(figsize=(11.64,7.81), constrained_layout=True)

scatter_plots = []
station_ids = []

# Plot each Basin Type separately and keep track of stations
for basin_type, color in basin_colors.items():
    subset = Features_new[Features_new["Basin Type Latest"] == basin_type]
    sc = ax.scatter(subset["Vs30"], subset["Z1_NZVM2p08"],
                    edgecolor="k", s=60, color=color, label=basin_type, picker=True)
    scatter_plots.append(sc)
    station_ids.append(subset["Station"].values)

# Add correlation models
vs30_vals = np.linspace(Features_new["Vs30"].min(), Features_new["Vs30"].max(), 300)
# Add new correlation model from image
z1_ASK = np.zeros_like(vs30_vals)
for i, vs30 in enumerate(vs30_vals):
    if vs30 <= 180:
        z1_ASK[i] = np.exp(6.745)
    elif 180 < vs30 <= 500:
        z1_ASK[i] = np.exp(6.745 - 1.35 * (vs30 / 180))
    else:  # vs30 > 500
        z1_ASK[i] = np.exp(5.394 - 4.48 * (vs30 / 500))
ax.plot(vs30_vals, z1_ASK, "r-", linewidth=2, label="Abrahamson & Silva (2008)")

z1_cy08 =  np.exp(28.5 - (3.82/8) * np.log(vs30_vals**8 + 378.7**8))
ax.plot(vs30_vals, z1_cy08, "g-", linewidth=2, label="Chiou & Youngs (2008)")

z1_cy14 = np.exp((-7.15/4) * np.log((vs30_vals**4 + 571**4) / (1360**4 + 571**4)))
ax.plot(vs30_vals, z1_cy14, "b-", linewidth=2, label="Chiou & Youngs (2014)")


# Formatting
ax.set_ylabel("Depth to $V_S$ = 1 km/s, $Z_{1.0}$ (m)", fontsize=20)
ax.set_xlabel("Average shear-wave velocity of top 30 m, $V_{S30}$ (m/s)", fontsize=20)
ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha='center', fontsize=16)
ax.set_yticklabels(ax.get_yticklabels(), fontsize=16)
ax.legend(fontsize=16, loc="best")
ax.tick_params(labelsize=16)
# ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xticks([100, 200, 300, 400, 500, 1000])
ax.set_xticklabels([100, 200, 300, 400, 500, 1000], rotation=0, ha='center', fontsize=16)
# Define what happens when clicking a point
def onpick(event):
    ind = event.ind[0]  # index of picked point
    sc = event.artist
    # find which scatter it belongs to
    for i, s in enumerate(scatter_plots):
        if s == sc:
            station = station_ids[i][ind]
            print(f"Clicked on station: {station}")
            ax.annotate(station,
                        (sc.get_offsets()[ind,0], sc.get_offsets()[ind,1]),
                        xytext=(10,10), textcoords='offset points',
                        fontsize=12, color="black",
                        bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.6))
            fig.canvas.draw()
            break

fig.canvas.mpl_connect("pick_event", onpick)
plt.savefig(os.path.join(save_dir, "Vs30_vs_Z1.0_ByBasinType.png"), dpi=300)
#%% Z1 Vs. Vs30 - Separated into basin types
basin_colors = {
    "Non-Basin": '#800000',
    "Unmodeled Basin": '#FF5733',
    "Type 1 Basin": '#33FFBD',
    "Type 2 Basin": '#9B59B6',
    "Type 3 Basin": '#1F618D',
    "Type 4 Basin": '#F4D03F'
}

fig, ax = plt.subplots(figsize=(11.64,7.81), constrained_layout=True)

scatter_plots = []
station_ids = []

# Plot all types together
for basin_type, color in basin_colors.items():
    subset = Features_new[Features_new["Basin Type Latest"] == basin_type]
    sc = ax.scatter(subset["Vs30"], subset["Z1_NZVM2p08"],
                    edgecolor="k", s=60, color=color, label=basin_type, picker=True)
    scatter_plots.append(sc)
    station_ids.append(subset["Station"].values)

# Add correlation models
vs30_vals = np.linspace(Features_new["Vs30"].min(), Features_new["Vs30"].max(), 300)

z1_ASK = np.zeros_like(vs30_vals)
for i, vs30 in enumerate(vs30_vals):
    if vs30 <= 180:
        z1_ASK[i] = np.exp(6.745)
    elif 180 < vs30 <= 500:
        z1_ASK[i] = np.exp(6.745 - 1.35 * (vs30 / 180))
    else:  # vs30 > 500
        z1_ASK[i] = np.exp(5.394 - 4.48 * (vs30 / 500))
ax.plot(vs30_vals, z1_ASK, "r-", linewidth=2, label="Abrahamson & Silva (2008)")

z1_cy08 = np.exp(28.5 - (3.82/8) * np.log(vs30_vals**8 + 378.7**8))

z1_cy14 = np.exp((-7.15/4) * np.log((vs30_vals**4 + 571**4) / (1360**4 + 571**4)))


for basin_type, color in basin_colors.items():
    fig_i, ax_i = plt.subplots(figsize=(11.64,7.81), constrained_layout=True)

    subset = Features_new[Features_new["Basin Type Latest"] == basin_type]
    ax_i.scatter(subset["Vs30"], subset["Z1_NZVM2p08"],
                 edgecolor="k", s=60, color=color, label=basin_type, picker=False)

    # Add correlation models
    ax_i.plot(vs30_vals, z1_ASK, "r-", linewidth=2, label="Abrahamson & Silva (2008)")
    ax_i.plot(vs30_vals, z1_cy08, "g-", linewidth=2, label="Chiou & Youngs (2008)")
    ax_i.plot(vs30_vals, z1_cy14, "b-", linewidth=2, label="Chiou & Youngs (2014)")

    # Formatting
    ax_i.set_ylabel("Depth to $V_S$ = 1 km/s, $Z_{1.0}$ (m)", fontsize=20)
    ax_i.set_xlabel("Average shear-wave velocity of top 30 m, $V_{S30}$ (m/s)", fontsize=20)
    ax_i.set_xticklabels(ax_i.get_xticklabels(), rotation=0, ha='center', fontsize=16)
    ax_i.set_yticklabels(ax_i.get_yticklabels(), fontsize=16)
    ax_i.legend(fontsize=16, loc="best")
    ax_i.tick_params(labelsize=16)
    ax_i.set_xscale('log')
    ax_i.set_xticks([100, 200, 300, 400, 500, 1000])
    ax_i.set_xticklabels([100, 200, 300, 400, 500, 1000], rotation=0, ha='center', fontsize=16)

    # Save with basin type in filename
    fname = f"Vs30_vs_Z1.0_{basin_type.replace(' ', '')}.png"
    plt.savefig(os.path.join(save_dir, fname), dpi=300)
    plt.close(fig_i)
#%% Z2p5 Vs. Vs30 - All NZGM4p3 stations
basin_colors = { "Non-Basin": '#800000', "Unmodeled Basin": '#FF5733', "Type 1 Basin": '#33FFBD', "Type 2 Basin": '#9B59B6', "Type 3 Basin": '#1F618D', "Type 4 Basin": '#F4D03F'}
fig, ax = plt.subplots(figsize=(11.64,7.81), constrained_layout=True)

scatter_plots = []
station_ids = []

# Plot each Basin Type separately and keep track of stations
for basin_type, color in basin_colors.items():
    subset = Features_new[Features_new["Basin Type Latest"] == basin_type]
    sc = ax.scatter(subset["Vs30"], subset["Z2p5_NZVM2p08"],
                    edgecolor="k", s=60, color=color, label=basin_type, picker=True)
    scatter_plots.append(sc)
    station_ids.append(subset["Station"].values)

# Add correlation models
vs30_vals = np.linspace(100, 1500, 300)


z2p5_CB14 =  (np.exp(7.089-1.144*np.log(vs30_vals)))*1000
ax.plot(vs30_vals, z2p5_CB14, "r-", linewidth=2, label="Campbell & Bozorgnia (2014)")



# Formatting
ax.set_ylabel("Depth to $V_S$ = 2.5 km/s, $Z_{2.5}$ (m)", fontsize=20)
ax.set_xlabel("Average shear-wave velocity of top 30 m, $V_{S30}$ (m/s)", fontsize=20)
ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha='center', fontsize=16)
ax.set_yticklabels(ax.get_yticklabels(), fontsize=16)
ax.legend(fontsize=12, loc="best")
ax.tick_params(labelsize=16)
ax.set_xscale('log')
ax.set_xticks([100, 200, 300, 400, 500, 1000])
ax.set_xticklabels([100, 200, 300, 400, 500, 1000], rotation=0, ha='center', fontsize=16)
# Define what happens when clicking a point
def onpick(event):
    ind = event.ind[0]  # index of picked point
    sc = event.artist
    # find which scatter it belongs to
    for i, s in enumerate(scatter_plots):
        if s == sc:
            station = station_ids[i][ind]
            print(f"Clicked on station: {station}")
            ax.annotate(station,
                        (sc.get_offsets()[ind,0], sc.get_offsets()[ind,1]),
                        xytext=(10,10), textcoords='offset points',
                        fontsize=12, color="black",
                        bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.6))
            fig.canvas.draw()
            break

fig.canvas.mpl_connect("pick_event", onpick)
plt.savefig(os.path.join(save_dir, "Vs30_vs_Z2.5_ByBasinType.png"), dpi=300)

