# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 09:17:19 2023

@author: ati47
"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import mplcursors as mpl
import os.path
import os
import re
import shutil
import geopandas as gpd
from shapely.geometry import Point
import addcopyfighandler
from matplotlib.gridspec import GridSpec
from scipy.interpolate import UnivariateSpline

plt.rcParams["font.family"] = "Times New Roman"

# Define the working directory
rootDir = Path(__file__).parent
outDir  = Path(__file__).parent / 'Figures_Final'
outDir.mkdir(parents=True, exist_ok=True)

# Define the folder that contains the inputs for the residual analysis
inputDir = Path(__file__).parents[1] / '2. Residual Analysis' / 'residual_input'

# Define the folder that contains the results of the residual analysis
resultsDirff = Path(__file__).parents[1] / '2. Residual Analysis' / 'residual_output' / 'ff'
resultsDirnb = Path(__file__).parents[1] / '2. Residual Analysis' / 'residual_output' / 'nb'


events = pd.read_csv(os.path.join(inputDir, 'events.csv'))
stations = pd.read_csv(os.path.join(inputDir, 'stations.csv'))
im_sim_wb   = pd.read_csv(os.path.join(inputDir, 'im_sim_withbasin.csv'))
im_sim_nb   = pd.read_csv(os.path.join(inputDir, 'im_sim_nobasin.csv'))
im_obs      = pd.read_csv(os.path.join(inputDir, 'im_obs.csv'))
event_counts = im_sim_wb['stat_id'].value_counts()

#Read all basin-runs results
bias_std_ff    = pd.read_csv(os.path.join(resultsDirff, 'bias_std_df.csv'))
siteres_ff     = pd.read_csv(os.path.join(resultsDirff, 'site_res_df.csv'))
sitereserr_ff  = pd.read_csv(os.path.join(resultsDirff, 'site_cond_std_df.csv'))
eventres_ff    = pd.read_csv(os.path.join(resultsDirff, 'event_res_df.csv'))

#Read all non-basin runs results
bias_std_nb = pd.read_csv(os.path.join(resultsDirnb, 'bias_std_df.csv'))
siteres_nb  = pd.read_csv(os.path.join(resultsDirnb, 'site_res_df.csv'))
sitereserr_nb  = pd.read_csv(os.path.join(resultsDirnb, 'site_cond_std_df.csv'))
eventres_nb = pd.read_csv(os.path.join(resultsDirnb, 'event_res_df.csv'))

#Categorization results
catresults = pd.read_csv(os.path.join(Path(__file__).parents[2], '1. Datasets','LATEST_NZGMDBv_4.3_Categorizations.csv'))


cat_sel = catresults[["Station", "Geomorphology", "BasinType_2p09","Vs30","Q_Vs30", "T0","Q_T0","Latitude","Longitude", "Z1_2p09_TRUE","Z1_2p09","Z1_nb","Basin","Basin_depth","Vs50_explicit","Vs50_implicit"]]
stations = stations.merge(
    cat_sel,
    left_on="stat_name",
    right_on="Station",
    how="left"
)
stations = stations.drop(columns=["Station"])
stations["stat_key"] = "stat_" + stations["stat_id"].astype(str)
stations_sel = stations[["stat_name","stat_key", "Geomorphology","Vs30","BasinType_2p09","Z1_2p09","Z1_nb","Z1_2p09_TRUE","Basin","Basin_depth","Vs50_explicit","Vs50_implicit"]]
stations_sel["Basin_depth"] = pd.to_numeric(stations_sel["Basin_depth"], errors="coerce").round(0)
stations_sel['Z1_ratio'] = stations_sel['Z1_2p09']/stations_sel['Z1_nb']
siteres_ff = siteres_ff.merge(stations_sel, left_on="Unnamed: 0", right_on="stat_key", how="left")
siteres_nb = siteres_nb.merge(stations_sel, left_on="Unnamed: 0", right_on="stat_key", how="left")
sitereserr_ff = sitereserr_ff.merge(stations_sel, left_on="Unnamed: 0", right_on="stat_key", how="left")
sitereserr_nb = sitereserr_nb.merge(stations_sel, left_on="Unnamed: 0", right_on="stat_key", how="left")
siteres_ff = siteres_ff.drop(columns=["stat_key"])
siteres_nb = siteres_nb.drop(columns=["stat_key"])
sitereserr_ff = sitereserr_ff.drop(columns=["stat_key"])
sitereserr_nb = sitereserr_nb.drop(columns=["stat_key"])
siteres_ff['stat_key'] = siteres_ff['Unnamed: 0']
siteres_nb['stat_key'] = siteres_nb['Unnamed: 0']
sitereserr_ff['stat_key'] = sitereserr_ff['Unnamed: 0']
sitereserr_nb['stat_key'] = sitereserr_nb['Unnamed: 0']
stations_sel_sorted = stations_sel.sort_values("stat_key").reset_index(drop=True)
siteres_ff_sorted   = siteres_ff.sort_values("stat_key").reset_index(drop=True)
siteres_nb_sorted   = siteres_nb.sort_values("stat_key").reset_index(drop=True)
sitereserr_ff_sorted   = sitereserr_ff.sort_values("stat_key").reset_index(drop=True)
sitereserr_nb_sorted   = sitereserr_nb.sort_values("stat_key").reset_index(drop=True)
assert stations_sel_sorted['stat_key'].tolist() == siteres_ff_sorted['stat_key'].tolist(), "Mismatch in FF ordering!"
assert stations_sel_sorted['stat_key'].tolist() == siteres_nb_sorted['stat_key'].tolist(), "Mismatch in NB ordering!"
assert stations_sel_sorted['stat_key'].tolist() == sitereserr_ff_sorted['stat_key'].tolist(), "Mismatch in FF ordering!"
assert stations_sel_sorted['stat_key'].tolist() == sitereserr_nb_sorted['stat_key'].tolist(), "Mismatch in NB ordering!"


nzgmdbDir = Path(r'C:\Users\ati47\OneDrive - University of Canterbury\Desktop\PhD\10. Research\Objective 2\1. Datasets\Latest GMDB\4p3\earthquake_source_table.csv')
df_nzgmdb = pd.read_csv(nzgmdbDir, dtype={'sta': str, 'evid': str})
# Extract list of events
eventList = events.iloc[:, 1].to_list()
print('Number of events:', len(eventList))
magnitudes = []
for event in eventList:
    mag = df_nzgmdb[df_nzgmdb['evid'] == event]['mag'].drop_duplicates().values
    magnitudes.append(mag)
print(magnitudes)
print('max mag:', np.max(magnitudes))
print('min mag:', np.min(magnitudes))

# Extract vibration periods (rows starting with "pSA_")
T = []
period_rows = bias_std_ff[bias_std_ff.iloc[:, 0].str.startswith("pSA_", na=False)].iloc[:, 0]
for period in period_rows:
    T.append(float(period.replace("pSA_", "")))

# Extract frequencies (rows starting with "EAS_")
f = []
freq_rows = bias_std_ff[bias_std_ff.iloc[:, 0].str.startswith("EAS_", na=False)].iloc[:, 0]
for frequency in freq_rows:
    f.append(float(frequency.replace("EAS_", "")))

# Model bias, a, for SA
sa_rows_ff = bias_std_ff[bias_std_ff.iloc[:, 0].str.startswith("pSA_", na=False)]
bias_SA_ff = sa_rows_ff['bias']
# bias_std_err_SA_ff = sa_rows_ff['bias_std_err']

sa_rows_nb = bias_std_nb[bias_std_nb.iloc[:, 0].str.startswith("pSA_", na=False)]
bias_SA_nb = sa_rows_nb['bias']
# bias_std_err_SA_nb = sa_rows_nb['bias_std_err']


# Model bias, a, for EAS
eas_rows_ff = bias_std_ff[bias_std_ff.iloc[:, 0].str.startswith("EAS_", na=False)]
bias_EAS_ff = eas_rows_ff['bias']
# bias_std_err_EAS_ff = eas_rows_ff['bias_std_err']

eas_rows_nb = bias_std_nb[bias_std_nb.iloc[:, 0].str.startswith("EAS_", na=False)]
bias_EAS_nb = eas_rows_nb['bias']
# bias_std_err_EAS_nb = eas_rows_nb['bias_std_err']

# Model bias, a, for the other IMs
bias_IMs_ff = bias_std_ff.loc[0:5, 'bias']
# bias_std_err_IMs_ff = bias_std_ff.loc[0:5, 'bias_std_err']

bias_IMs_nb= bias_std_nb.loc[0:5, 'bias']
# bias_std_err_IMs_nb = bias_std_nb.loc[0:5, 'bias_std_err']


# Total standard deviation, sigma, for SA
sigma_SA_ff = sa_rows_ff['sigma']
sigma_SA_nb = sa_rows_nb['sigma']

# Total standard deviation, sigma, for EAS (or other IMs)
sigma_EAS_ff = eas_rows_ff['sigma']
sigma_EAS_nb = eas_rows_nb['sigma']

# Total standard deviation, sigma, for the other IMs
sigma_IMs_ff = bias_std_ff.loc[0:5, 'sigma']
sigma_IMs_nb = bias_std_nb.loc[0:5, 'sigma']

# Site-to-site standard deviation, Tau, for SA
tau_SA_ff = sa_rows_ff['tau']
tau_SA_nb = sa_rows_nb['tau']

#Site-to-site standard deviation, Tau, for EAS (or other IMs)
tau_EAS_ff = eas_rows_ff['tau']
tau_EAS_nb = eas_rows_nb['tau']

#Site-to-site standard deviation, Tau, for the other IMs
tau_IMs_ff = bias_std_ff.loc[0:5, 'tau']
tau_IMs_nb = bias_std_nb.loc[0:5, 'tau']

# Site-to-site standard deviation, phi_s2s, for SA
phiS2S_SA_ff = sa_rows_ff['phi_S2S']
phiS2S_SA_nb = sa_rows_nb['phi_S2S']

# Site-to-site standard deviation, phi_s2s, for EAS (or other IMs)
phiS2S_EAS_ff = eas_rows_ff['phi_S2S']
phiS2S_EAS_nb = eas_rows_nb['phi_S2S']

# Site-to-site standard deviation, phi_s2s, for the other IMs
phiS2S_IMs_ff = bias_std_ff.loc[0:5, 'phi_S2S']
phiS2S_IMs_nb = bias_std_nb.loc[0:5, 'phi_S2S']

# Remaining standard deviation, phi_w, for SA
phiss_SA_ff = sa_rows_ff['phi_w']
phiss_SA_nb = sa_rows_nb['phi_w']

phiss_EAS_ff = eas_rows_ff['phi_w']
phiss_EAS_nb = eas_rows_nb['phi_w']

# Remaining standard deviation, phi_w, for the other IMs
phiss_IMs_ff = bias_std_ff.loc[0:5, 'phi_w']
phiss_IMs_nb = bias_std_nb.loc[0:5, 'phi_w']

stations_sel_sorted['Basin'] = stations_sel_sorted['Basin'].replace({
    'PalmerstonNorth': 'Palmerston North',
    'Nelson': 'Nelson-Tasman',
    'WestCoast': 'West Coast',
    'NorthCanterbury': 'North Canterbury'
})
#%% Plot between Z1,sim implicit and explicit for Type 1-4
stations_sel_sorted_select = stations_sel_sorted[(stations_sel_sorted['BasinType_2p09']!= "Non-Basin") & (stations_sel_sorted['BasinType_2p09']!='Unmodeled') & (stations_sel_sorted['Z1_2p09']<100)]


fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)

ax.scatter(stations_sel_sorted_select['Z1_2p09'],stations_sel_sorted_select['Z1_nb'], s=30, color='k')
ax.set_xlabel(r"$Z_{1.0,\mathrm{sim}}$ (Explicit)", fontsize=20)
ax.set_ylabel(r"$Z_{1.0,\mathrm{sim}}$ (Implicit)", fontsize=20)

lims = [
    min(ax.get_xlim()[0], ax.get_ylim()[0]),
    max(ax.get_xlim()[1], ax.get_ylim()[1])
]
ax.plot(lims, lims, "k--", linewidth=1)
ax.axvline(50, linestyle=":", linewidth=2)
ax.axvline(150, linestyle=":", linewidth=2)

ax.axhline(50, linestyle=":", linewidth=2)
ax.axhline(150, linestyle=":", linewidth=2)
ax.tick_params(direction='in',axis='both', which = 'both',labelsize=16)
ax.set_xlim([0,200])
ax.set_aspect("equal", adjustable="box")
#%% Plot between Vs50 implicit and explicit for Type 1-4
stations_sel_sorted_select = stations_sel_sorted[(stations_sel_sorted['BasinType_2p09']!= "Non-Basin") & (stations_sel_sorted['BasinType_2p09']!='Unmodeled') & (stations_sel_sorted['Z1_2p09']<100)]


fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)

ax.scatter(stations_sel_sorted_select['Vs50_explicit'],stations_sel_sorted_select['Vs50_implicit'], s=30, color='k')
ax.set_xlabel(r"$V_{S,50}$ (Explicit)", fontsize=20)
ax.set_ylabel(r"$V_{S,50}$ (Implicit)", fontsize=20)

lims = [
    min(ax.get_xlim()[0], ax.get_ylim()[0]),
    max(ax.get_xlim()[1], ax.get_ylim()[1])
]
ax.plot(lims, lims, "k--", linewidth=1)
# ax.axvline(50, linestyle=":", linewidth=2)
# ax.axvline(150, linestyle=":", linewidth=2)

# ax.axhline(50, linestyle=":", linewidth=2)
# ax.axhline(150, linestyle=":", linewidth=2)
ax.tick_params(direction='in',axis='both', which = 'both',labelsize=16)
# ax.set_xlim([0,200])
ax.set_aspect("equal", adjustable="box")
#%% Plot between Z1, true and Z1,sim
import matplotlib.pyplot as plt
geom_colors = {'Basin': '#1E90FF', 'Basin Edge' : '#66C266', 'Valley': '#9370DB', 'Hill': '#993333'}


fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)

# Plot each geomorphology group separately
for geom, group in stations_sel_sorted.groupby('Geomorphology'):
    ax.scatter(
        group["Z1_2p09_TRUE"],
        group["Z1_2p09"],
        label=geom,
        color=geom_colors.get(geom, 'gray'),
        s=60,
        # alpha=0.8,
        edgecolor='black',
        linewidth=0.4
    )

ax.set_xlabel(r"$Z_{1.0,\mathrm{true}}$ (m)", fontsize=20)
ax.set_ylabel(r"$Z_{1.0,\mathrm{sim}}$ (m)", fontsize=20)


ax.tick_params(direction='in',axis='both', which = 'both',labelsize=16)


xmin = min(stations_sel_sorted["Z1_2p09_TRUE"].min(),
           stations_sel_sorted["Z1_2p09"].min())
xmax = max(stations_sel_sorted["Z1_2p09_TRUE"].max(),
           stations_sel_sorted["Z1_2p09"].max())

ax.plot([xmin, xmax], [xmin, xmax],
        'k--', lw=1.5, label='1:1 Line')


ax.legend(
    fontsize=16,
    title=None,
    # title_fontsize=13,
    frameon=True,
    loc='best'
)

plt.savefig(os.path.join(outDir,"Z1_scatter_geomorphology.pdf"))

#%% Maximum absolute change, |delta R| = |Rwithout-basin - Rwith-basin|
pSA_cols = [c for c in siteres_ff.columns if c.startswith("pSA_")]
siteres_ff_pSA  = siteres_ff_sorted[pSA_cols]
siteres_nb_pSA  = siteres_nb_sorted[pSA_cols]
siteres_systematic_ff_pSA = siteres_ff_pSA + np.array(bias_SA_ff)
siteres_systematic_nb_pSA = siteres_nb_pSA + np.array(bias_SA_nb)
EAS_cols = [c for c in siteres_ff.columns if c.startswith("EAS_")]
siteres_ff_EAS  = siteres_ff_sorted[EAS_cols]
siteres_nb_EAS  = siteres_nb_sorted[EAS_cols]
siteres_systematic_ff_EAS = siteres_ff_EAS + np.array(bias_EAS_ff)
siteres_systematic_nb_EAS = siteres_nb_EAS + np.array(bias_EAS_nb)
R_with = siteres_systematic_ff_pSA.copy()
R_without = siteres_systematic_nb_pSA.copy()
R_with_EAS = siteres_systematic_ff_EAS.copy()
R_without_EAS = siteres_systematic_nb_EAS.copy()
delta_R_abs = (R_without - R_with).abs()
delta_R_abs_EAS = (R_without_EAS - R_with_EAS).abs()
delta_R_abs_max = delta_R_abs.max(axis=1)
delta_R_abs_max_EAS = delta_R_abs_EAS.max(axis=1)
T_delta_R_abs_max = delta_R_abs.idxmax(axis=1)
delta_R_summary = stations_sel_sorted.copy()

delta_R_summary["delta_R_abs_max"] = delta_R_abs_max
delta_R_summary["delta_R_abs_max_EAS"] = delta_R_abs_max_EAS
delta_R_summary["T_delta_R_abs_max"] = T_delta_R_abs_max
delta_R_summary["T_delta_R_abs_max"] = (
    delta_R_summary["T_delta_R_abs_max"]
    .str.replace("pSA_", "", regex=False)
    .astype(float)
)
delta_R_summary =  delta_R_summary.merge(stations[["stat_name","T0"]],on="stat_name",how="left")


basin_types_to_keep = ["Type 1", "Type 2", "Type 3", "Type 4"]

plot_df = delta_R_summary[
    delta_R_summary["BasinType_2p09"].isin(basin_types_to_keep)
].copy()


plot_df = plot_df.dropna(subset=["Z1_2p09", "delta_R_abs_max", "BasinType_2p09","delta_R_abs_max_EAS"])

plt.figure(figsize=(7, 5),constrained_layout=True)

sns.scatterplot(
    data=plot_df,
    x="Z1_2p09",
    y="delta_R_abs_max",
    hue="BasinType_2p09",
    s=55,
    alpha=0.75,
    edgecolor="k",
    linewidth=0.3
)

plt.xlabel(r"$Z_{1,\mathrm{sim}}$ (m)")
plt.ylabel(r"$|\Delta R|_{\max}$")
plt.title(r"Maximum basin-induced residual change vs. $Z_{1,\mathrm{sim}}$ - pSA")
plt.legend(frameon=False)
plt.savefig("BasinTypeVsDepth.png",dpi=300)

plt.figure(figsize=(7, 5),constrained_layout=True)

sns.scatterplot(
    data=plot_df,
    x="Z1_2p09",
    y="delta_R_abs_max_EAS",
    hue="BasinType_2p09",
    s=55,
    alpha=0.75,
    edgecolor="k",
    linewidth=0.3
)

plt.xlabel(r"$Z_{1,\mathrm{sim}}$ (m)")
plt.ylabel(r"$|\Delta R|_{\max}$")
plt.title(r"Maximum basin-induced residual change vs. $Z_{1,\mathrm{sim}}$ - EAS")
plt.legend(frameon=False)
plt.savefig("BasinTypeVsDepth_EAS.png",dpi=300)
#%% Average period vs Delta implicit - Delta explicit - Paper 2
def extract_period(col):
    return float(str(col).split("_")[-1])

pSA_cols = [c for c in siteres_ff_sorted.columns if c.startswith("pSA_")]
EAS_cols = [c for c in siteres_ff_sorted.columns if c.startswith("EAS_")]

siteres_ff_pSA = siteres_ff_sorted[pSA_cols]
siteres_nb_pSA = siteres_nb_sorted[pSA_cols]

siteres_ff_EAS = siteres_ff_sorted[EAS_cols]
siteres_nb_EAS = siteres_nb_sorted[EAS_cols]

siteres_systematic_ff_pSA = siteres_ff_pSA + np.array(bias_SA_ff)
siteres_systematic_nb_pSA = siteres_nb_pSA + np.array(bias_SA_nb)

siteres_systematic_ff_EAS = siteres_ff_EAS + np.array(bias_EAS_ff)
siteres_systematic_nb_EAS = siteres_nb_EAS + np.array(bias_EAS_nb)


R_with_pSA = siteres_systematic_ff_pSA.copy()
R_without_pSA = siteres_systematic_nb_pSA.copy()

R_with_EAS = siteres_systematic_ff_EAS.copy()
R_without_EAS = siteres_systematic_nb_EAS.copy()

delta_R_pSA = R_without_pSA - R_with_pSA
delta_R_EAS = R_without_EAS - R_with_EAS


pSA_periods = pd.Series(
    {col: extract_period(col) for col in delta_R_pSA.columns}
)

pSA_band_cols = pSA_periods[
    (pSA_periods >= 1.0) & (pSA_periods <= 3.0)
].index.tolist()

print("pSA columns in 1.0-3.0 s band:")
print(pSA_band_cols)


delta_R_pSA_1p5_2p5_mean = delta_R_pSA[pSA_band_cols].mean(axis=1)

# If EAS column suffix is frequency in Hz
EAS_freqs = pd.Series(
    {col: extract_period(col) for col in delta_R_EAS.columns}
)

EAS_band_cols = EAS_freqs[
    (EAS_freqs >= 1/3) & (EAS_freqs <= 1/1.0)
].index.tolist()

print("EAS columns corresponding to T = 1.0-3.0 s:")
print(EAS_band_cols)

delta_R_EAS_1p5_2p5_mean = delta_R_EAS[EAS_band_cols].mean(axis=1)

delta_R_summary = stations_sel_sorted.copy()

delta_R_summary["delta_R_pSA_1p5_2p5_mean"] = delta_R_pSA_1p5_2p5_mean
delta_R_summary["delta_R_EAS_1p5_2p5_mean"] = delta_R_EAS_1p5_2p5_mean

basin_types_to_keep = ["Type 1", "Type 2", "Type 3", "Type 4"]

plot_df = delta_R_summary[
    delta_R_summary["BasinType_2p09"].isin(basin_types_to_keep)
].copy()

plot_df = plot_df.dropna(
    subset=[
        "Z1_2p09",
        "BasinType_2p09",
        "delta_R_pSA_1p5_2p5_mean",
        "delta_R_EAS_1p5_2p5_mean"
    ]
)

basin_order = ["Type 1", "Type 2", "Type 3", "Type 4"]

basin_palette = {
    "Type 1": '#33FFBD',
    "Type 2": '#9B59B6',
    "Type 3": '#1F618D',
    "Type 4": '#F4D03F'
}

fig, ax = plt.subplots(figsize=(9.82, 7.12), constrained_layout=True)

sns.scatterplot(
    data=plot_df,
    x="Z1_2p09",
    y="delta_R_pSA_1p5_2p5_mean",
    hue="BasinType_2p09",
    hue_order=basin_order,
    palette=basin_palette,
    s=60,
    edgecolor="k",
    linewidth = 0.75
)

ax.axhline(0, color="k", linestyle="--", linewidth=1)

ax.set_xlabel(r"$Z_{1.0,\mathrm{sim}}$ (m)", fontsize=28)
ax.set_ylabel(
    r"Implicit - explicit difference, $\Delta(a+\delta S2S_s)$",
    fontsize=25
)
ax.set_xscale('log')
ax.tick_params(labelsize=22,direction='in', axis='both', which='both')
ax.legend(title =  None, fontsize=22, loc="best", framealpha=0.9)
fig.text(0.05, 0.955, '(b)', fontsize=26, fontweight='bold') 
ax.set_ylim([-0.5,0.9])
ax.tick_params(axis='x', which='both', pad=10)
fig.savefig(os.path.join(outDir,"Z1VsBasinType_pSA_residual.pdf"))


fig, ax = plt.subplots(figsize=(9,6.11), constrained_layout=True)

sns.scatterplot(
    data=plot_df,
    x="Z1_2p09",
    y="delta_R_EAS_1p5_2p5_mean",
    hue="BasinType_2p09",
    hue_order=basin_order,
    palette=basin_palette,
    s=60,
    edgecolor="k",
    linewidth=0.75
)

ax.axhline(0, color="k", linestyle="--", linewidth=1)
ax.set_xlabel(r"$Z_{1.0,\mathrm{sim}}$ (m)", fontsize=20)
ax.set_ylabel(
    r"Implicit - explicit difference, $\Delta(a+\delta S2S_s)$",
    fontsize=19
)
ax.set_xscale('log')
ax.set_ylim([-0.5,1])
ax.legend(title =  None, fontsize=16, loc="lower right", framealpha=0.9)
ax.tick_params(axis='x', which='both', pad=10)
ax.tick_params(labelsize=16,direction='in', axis='both', which='both')
fig.text(0.02, 0.95, '(a)', fontsize=20, fontweight='bold') 
fig.savefig(os.path.join(outDir,"Z1VsBasinType_EAS_residual.pdf"))

#%% Basin Type or Depth - pSA?
bins = [-np.inf, 100, 300, 500,  np.inf]
labels = [
    "< 100 m",
    "100-300 m",
    "300–500 m",
    "≥ 500 m"
]

plot_df["Z1_bin"] = pd.cut(
    plot_df["Z1_2p09"],
    bins=bins,
    labels=labels,
    right=False
)

plot_df = plot_df.dropna(subset=["Z1_bin"]).copy()


count_table = pd.crosstab(plot_df["Z1_bin"], plot_df["BasinType_2p09"])
print("Counts in each depth bin:")
print(count_table)

print("\nSummary statistics:")
summary_table = (
    plot_df.groupby(["Z1_bin", "BasinType_2p09"], observed=True)["delta_R_pSA_1p5_2p5_mean"]
    .agg(["count", "median", "mean", "std", "min", "max"])
    .round(3)
)
print(summary_table)

plt.figure(figsize=(10, 6))

sns.boxplot(
    data=plot_df,
    x="Z1_bin",
    y="delta_R_pSA_1p5_2p5_mean",
    hue="BasinType_2p09",
    showfliers=False
)

sns.stripplot(
    data=plot_df,
    x="Z1_bin",
    y="delta_R_pSA_1p5_2p5_mean",
    hue="BasinType_2p09",
    dodge=True,
    alpha=0.5,
    size=4,
    linewidth=0.3,
    edgecolor="k"
)

plt.xlabel(r"$Z_{1,\mathrm{sim}}$ range")
plt.ylabel(r"$|\Delta R|_{\max}$")
plt.title(r"$|\Delta R|_{\max}$ by basin type within $Z_{1,\mathrm{sim}}$ bins - pSA")

# Remove duplicate legends from boxplot + stripplot
handles, labels_legend = plt.gca().get_legend_handles_labels()
n_types = len(basin_types_to_keep)
plt.legend(
    handles[:n_types],
    labels_legend[:n_types],
    title="Basin Type",
    frameon=False
)

plt.tight_layout()
plt.show()
plt.savefig("BasinTypeVsDepth_BoxWhisker.png",dpi=300)
#%% Basin Type or Depth - EAS?
basin_order = ["Type 1", "Type 2", "Type 3", "Type 4"]

basin_palette = {
    "Type 1": '#33FFBD',
    "Type 2": '#9B59B6',
    "Type 3": '#1F618D',
    "Type 4": '#F4D03F'
}


plot_df = plot_df[plot_df["BasinType_2p09"].isin(basin_order)].copy()


plot_df["BasinType_2p09"] = pd.Categorical(
    plot_df["BasinType_2p09"],
    categories=basin_order,
    ordered=True
)


bins = [-np.inf, 100, 300, 500, np.inf]

labels = [
    "< 100 m",
    "100-300 m",
    "300-500 m",
    ">= 500 m"
]

plot_df["Z1_bin"] = pd.cut(
    plot_df["Z1_2p09"],
    bins=bins,
    labels=labels,
    right=False
)

plot_df = plot_df.dropna(subset=["Z1_bin"]).copy()


plot_df["Z1_bin"] = pd.Categorical(
    plot_df["Z1_bin"],
    categories=labels,
    ordered=True
)


count_table = pd.crosstab(
    plot_df["Z1_bin"],
    plot_df["BasinType_2p09"]
).reindex(index=labels, columns=basin_order)

print("Counts in each depth bin:")
print(count_table)

print("\nSummary statistics:")
summary_table = (
    plot_df
    .groupby(["Z1_bin", "BasinType_2p09"], observed=True)["delta_R_EAS_1p5_2p5_mean"]
    .agg(["count", "median", "mean", "std", "min", "max"])
    .round(3)
)
print(summary_table)


fig, ax = plt.subplots(figsize=(10.21, 6.36), constrained_layout=True)

sns.boxplot(
    data=plot_df,
    x="Z1_bin",
    y="delta_R_EAS_1p5_2p5_mean",
    hue="BasinType_2p09",
    order=labels,
    hue_order=basin_order,
    palette=basin_palette,
    showfliers=False,
    width=0.72,
    ax=ax
)

sns.stripplot(
    data=plot_df,
    x="Z1_bin",
    y="delta_R_EAS_1p5_2p5_mean",
    hue="BasinType_2p09",
    order=labels,
    hue_order=basin_order,
    palette=basin_palette,
    dodge=True,
    alpha=0.55,
    size=4,
    linewidth=0.3,
    edgecolor="k",
    ax=ax
)

ax.set_xlabel(r"$Z_{1.0,\mathrm{sim}}$ range", fontsize=20)

ax.set_ylabel(
    r"Implicit - explicit difference, $\Delta(a+\delta S2S_s)$",
    fontsize=20
)

ax.set_ylim([-0.5, 1])
ax.tick_params(labelsize=16, direction="in", axis="both", which="both")

handles, labels_legend = ax.get_legend_handles_labels()

ax.legend(
    handles[:len(basin_order)],
    labels_legend[:len(basin_order)],
    title=None,
    fontsize=16,
    frameon=False
)

fig.text(0.02, 0.95, "(b)", fontsize=20, fontweight="bold")

fig.savefig(
    os.path.join(outDir, "BasinTypeVsDepth_BoxWhisker_EAS.pdf"),
    dpi=300
)

fig.savefig(os.path.join(outDir,"BasinTypeVsDepth_BoxWhisker_EAS.pdf"))
#%% Correlation of Z1,sim with Basin Type - Paper 2
import seaborn as sns
import matplotlib.pyplot as plt

basin_order = ["Type 1", "Type 2", "Type 3", "Type 4"]
basin_palette = {
    "Type 1": '#33FFBD',
    "Type 2": '#9B59B6',
    "Type 3": '#1F618D',
    "Type 4": '#F4D03F'
}
df = stations_sel_sorted.dropna(subset=["BasinType_2p09", "Z1_2p09","Basin_depth"]).copy()
df = df[df["BasinType_2p09"].isin(basin_order)].copy()
df["BasinType_2p09"] = pd.Categorical(
    df["BasinType_2p09"],
    categories=basin_order,
    ordered=True
)
fig = plt.figure(figsize=(5.69, 3.59),constrained_layout=True)

ax = sns.boxplot(
    data=df,
    x="BasinType_2p09",
    y="Z1_2p09",
    order=basin_order,
    palette=basin_palette,
    width=0.35,
    showfliers=False
)

sns.stripplot(
    data=df,
    x="BasinType_2p09",
    y="Z1_2p09",
    order=basin_order,
    color="k",
    alpha=0.35,
    jitter=True,
    size=3,
    ax=ax
)
fig.text(0.008, 0.95, '(a)', fontsize=16, fontweight='bold')
plt.yscale('log')
plt.xlabel("Basin Type",fontsize=16)
plt.ylabel(r"$Z_{1.0,\mathrm{sim}}$ (m)",fontsize=16)
ax.tick_params(labelsize=14,direction='in', axis='both', which='both')
fig.savefig(os.path.join(outDir,'Z1VsBasinType_Stations.pdf'))
#%% Z1 Vs. Vs30 - All stations considered here for Paper 2 (295)....
basin_colors = {
    "Non-Basin": '#800000',
    "Unmodeled": '#FF5733',
    "Type 1": '#33FFBD',
    "Type 2": '#9B59B6',
    "Type 3": '#1F618D',
    "Type 4": '#F4D03F'
}

xcol = "Vs30"
ycol = "Z1_2p09"
tcol = "BasinType_2p09"
idcol = "stat_name"

df = stations_sel_sorted.copy()


fig = plt.figure(figsize=(9.05, 7.54), constrained_layout=True)
gs = GridSpec(
    nrows=2, ncols=2, figure=fig,
    height_ratios=[1.2, 4.5], width_ratios=[4.5, 1.2],
    hspace=0.05, wspace=0.05
)

ax_top   = fig.add_subplot(gs[0, 0])
ax_main  = fig.add_subplot(gs[1, 0], sharex=ax_top)
ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)


plt.setp(ax_top.get_xticklabels(), visible=False)
plt.setp(ax_right.get_yticklabels(), visible=False)
scatter_plots = []
station_ids = []

for basin_type, color in basin_colors.items():
    subset = df[df[tcol] == basin_type]
    sc = ax_main.scatter(
        subset[xcol], subset[ycol],
        edgecolor="k", s=60, color=color, picker=True, zorder=3
    )
    scatter_plots.append(sc)
    station_ids.append(subset[idcol].values)


vs30_vals = np.linspace(df[xcol].min(), df[xcol].max(), 400)
z1_ASK = np.zeros_like(vs30_vals)
for i, vs30 in enumerate(vs30_vals):
    if vs30 <= 180:
        z1_ASK[i] = np.exp(6.745)
    elif 180 < vs30 <= 500:
        z1_ASK[i] = np.exp(6.745 - 1.35 * (vs30 / 180))
    else:  # vs30 > 500
        z1_ASK[i] = np.exp(5.394 - 4.48 * (vs30 / 500))
z1_cy08 =  np.exp(28.5 - (3.82/8) * np.log(vs30_vals**8 + 378.7**8))
z1_cy14 = np.exp((-7.15/4) * np.log((vs30_vals**4 + 571**4) / (1360**4 + 571**4)))

z1_ba18 = np.exp((-7.67 / 4) * np.log(((vs30_vals ** 4) + (610 ** 4)) / ((1360 ** 4) + (610 ** 4))))

ax_main.plot(vs30_vals, z1_ASK, "r", linewidth=2, label="Abrahamson & Silva (2008)", zorder=2)
ax_main.plot(vs30_vals, z1_cy08, "g", linewidth=2, label="Chiou & Youngs (2008)", zorder=2)
ax_main.plot(vs30_vals, z1_cy14, "b", linewidth=2, label="Chiou & Youngs (2014)", zorder=2)
ax_main.plot(vs30_vals, z1_ba18, "k", linewidth=2, label="Bayless & Abrahamson (2018)", zorder=2)

ax_main.plot(vs30_vals, 5.0 * z1_ba18, "k--", linewidth=2, zorder=2)
ax_main.plot(vs30_vals, 0.2 * z1_ba18, "k--", linewidth=2, zorder=2)


x_anno = 600
y_anno = np.interp(x_anno, vs30_vals, 0.2 * z1_ba18)

ax_main.annotate(
    "Factor of 5 variation on\n Bayless & Abrahamson (2018)",
    xy=(850, y_anno*0.2),
    xytext=(550, y_anno*0.05),
    textcoords="data",
    arrowprops=dict(arrowstyle="->", lw=1.5),
    fontsize=16,
    ha="left",
    va="bottom"
)

x_bins = np.arange(100, 1100, 50)
y_bins = np.logspace(np.log10(1), np.log10(1400), 30)
# y_bins = np.array([10,20,40, 50, 100, 200, 300, 500, 1000,1500])


x_arrays = [df[df[tcol] == bt][xcol].dropna().values for bt in basin_colors.keys()]
ax_top.hist(
    x_arrays, bins=x_bins, stacked=True,
    color=[basin_colors[bt] for bt in basin_colors.keys()],
    edgecolor="k", linewidth=0.5
)
ax_top.set_ylabel("Count", fontsize=18)
ax_top.tick_params(axis="both", labelsize=14)


y_arrays = [df[df[tcol] == bt][ycol].dropna().values for bt in basin_colors.keys()]
ax_right.hist(
    y_arrays, bins=y_bins, stacked=True, orientation="horizontal",
    color=[basin_colors[bt] for bt in basin_colors.keys()],
    edgecolor="k", linewidth=0.5
)
ax_right.set_xlabel("Count", fontsize=18)
ax_right.tick_params(axis="both", labelsize=14)


ax_main.set_ylabel(r"Depth to $V_S$ = 1 km/s, $Z_{1.0}$ (m)", fontsize=18)
ax_main.set_xlabel(r"Average shear-wave velocity of top 30 m, $V_{S30}$ (m/s)", fontsize=18)

ax_main.set_yscale("log")
ax_main.set_xlim([120, 1100])
ax_main.set_ylim([1, None])
ax_main.set_xticks([200, 400, 600, 800, 1000])
ax_main.tick_params(labelsize=14)


ax_right.set_yscale("log")


ax_main.legend(fontsize=14, loc="best", framealpha=0.9)


def onpick(event):
    ind = event.ind[0]
    sc = event.artist
    for i, s in enumerate(scatter_plots):
        if s == sc:
            station = station_ids[i][ind]
            print(f"Clicked on station: {station}")
            ax_main.annotate(
                station,
                (sc.get_offsets()[ind, 0], sc.get_offsets()[ind, 1]),
                xytext=(10, 10), textcoords="offset points",
                fontsize=11, color="black",
                bbox=dict(boxstyle="round,pad=0.25", fc="yellow", alpha=0.6)
            )
            fig.canvas.draw_idle()
            break

fig.canvas.mpl_connect("pick_event", onpick)
from matplotlib.patches import Patch

type_handles = [
    Patch(facecolor=basin_colors[bt], edgecolor="k", label=bt)
    for bt in basin_colors.keys()
]

ax_top.legend(handles=type_handles, loc="upper right",
              fontsize=12, framealpha=0.9, ncols=2)

fig.savefig(os.path.join(outDir,'Z1VsVs30_Stations.pdf'))
#%% Manual Loess
def loess(x, y, f):
    """
    Basic LOWESS smoother with uncertainty. 
    Note:
        - Not robust (so no iteration) and
             only normally distributed errors. 
        - No higher order polynomials d=1 
            so linear smoother.
    """
    # get some paras
    xwidth = f*(x.max()-x.min()) # effective width after reduction factor
    N = len(x) # number of obs
    # Don't assume the data is sorted
    order = np.argsort(x)
    # storage
    y_sm = np.zeros_like(y)
    y_stderr = np.zeros_like(y)
    # define the weigthing function -- clipping too!
    tricube = lambda d : np.clip((1- np.abs(d)**3)**3, 0, 1)
    # run the regression for each observation i
    for i in range(N):
        dist = np.abs((x[order][i]-x[order]))/xwidth
        w = tricube(dist)
        # form linear system with the weights
        A = np.stack([w, x[order]*w]).T
        b = w * y[order]
        ATA = A.T.dot(A)
        ATb = A.T.dot(b)
        # solve the syste
        sol = np.linalg.solve(ATA, ATb)
        # predict for the observation only
        yest = A[i].dot(sol)# equiv of A.dot(yest) just for k
        place = order[i]
        y_sm[place]=yest 
        sigma2 = (np.sum((A.dot(sol) -y [order])**2)/N )
        # Calculate the standard error
        y_stderr[place] = np.sqrt(sigma2 * 
                                A[i].dot(np.linalg.inv(ATA)
                                                    ).dot(A[i]))
    return y_sm, y_stderr
#%% Z1 Vs. Vs30 - All stations considered here for Paper 2 (295)....NO BASIN figure
basin_colors = {
    "Non-Basin": '#800000',
    "Unmodeled": '#FF5733',
    "Type 1": '#33FFBD',
    "Type 2": '#9B59B6',
    "Type 3": '#1F618D',
    "Type 4": '#F4D03F'
}

xcol = "Vs30"
ycol = "Z1_nb"
tcol = "BasinType_2p09"
idcol = "stat_name"

df = stations_sel_sorted.copy()


fig = plt.figure(figsize=(9.05, 7.54), constrained_layout=True)
gs = GridSpec(
    nrows=2, ncols=2, figure=fig,
    height_ratios=[1.2, 4.5], width_ratios=[4.5, 1.2],
    hspace=0.05, wspace=0.05
)

ax_top   = fig.add_subplot(gs[0, 0])
ax_main  = fig.add_subplot(gs[1, 0], sharex=ax_top)
ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)


plt.setp(ax_top.get_xticklabels(), visible=False)
plt.setp(ax_right.get_yticklabels(), visible=False)
scatter_plots = []
station_ids = []

for basin_type, color in basin_colors.items():
    subset = df[df[tcol] == basin_type]
    sc = ax_main.scatter(
        subset[xcol], subset[ycol],
        edgecolor="k", s=60, color=color, picker=True, zorder=3
    )
    scatter_plots.append(sc)
    station_ids.append(subset[idcol].values)


loess_df = df[[xcol, ycol]].dropna().copy()

x = loess_df[xcol].values
y = loess_df[ycol].values


y_log = np.log(y)


f = 0.35

y_sm_log, y_std_log = loess(x, y_log, f=f)


y_sm = np.exp(y_sm_log)


order = np.argsort(x)
x_sorted = x[order]
y_sm_sorted = y_sm[order]

ax_main.plot(
    x_sorted, y_sm_sorted,
    color="magenta", linewidth=3, zorder=4
)
vs30_vals = np.linspace(df[xcol].min(), df[xcol].max(), 400)
z1_ASK = np.zeros_like(vs30_vals)
for i, vs30 in enumerate(vs30_vals):
    if vs30 <= 180:
        z1_ASK[i] = np.exp(6.745)
    elif 180 < vs30 <= 500:
        z1_ASK[i] = np.exp(6.745 - 1.35 * (vs30 / 180))
    else:  # vs30 > 500
        z1_ASK[i] = np.exp(5.394 - 4.48 * (vs30 / 500))
z1_cy08 =  np.exp(28.5 - (3.82/8) * np.log(vs30_vals**8 + 378.7**8))
z1_cy14 = np.exp((-7.15/4) * np.log((vs30_vals**4 + 571**4) / (1360**4 + 571**4)))

z1_ba18 = np.exp((-7.67 / 4) * np.log(((vs30_vals ** 4) + (610 ** 4)) / ((1360 ** 4) + (610 ** 4))))

ax_main.plot(vs30_vals, z1_ASK, "r", linewidth=2, label="Abrahamson & Silva (2008)", zorder=2)
ax_main.plot(vs30_vals, z1_cy08, "g", linewidth=2, label="Chiou & Youngs (2008)", zorder=2)
ax_main.plot(vs30_vals, z1_cy14, "b", linewidth=2, label="Chiou & Youngs (2014)", zorder=2)
ax_main.plot(vs30_vals, z1_ba18, "k", linewidth=2, label="Bayless & Abrahamson (2018)", zorder=2)

ax_main.plot(vs30_vals, 5.0 * z1_ba18, "k--", linewidth=2, zorder=2)
ax_main.plot(vs30_vals, 0.2 * z1_ba18, "k--", linewidth=2, zorder=2)


x_anno = 600
y_anno = np.interp(x_anno, vs30_vals, 0.2 * z1_ba18)

ax_main.annotate(
    "Factor of 5 variation on\n Bayless & Abrahamson (2018)",
    xy=(850, y_anno*0.2),
    xytext=(550, y_anno*0.05),
    textcoords="data",
    arrowprops=dict(arrowstyle="->", lw=1.5),
    fontsize=16,
    ha="left",
    va="bottom"
)

x_bins = np.arange(100, 1100, 50)
y_bins = np.logspace(np.log10(1), np.log10(1400), 30)
# y_bins = np.array([10,20,40, 50, 100, 200, 300, 500, 1000,1500])


x_arrays = [df[df[tcol] == bt][xcol].dropna().values for bt in basin_colors.keys()]
ax_top.hist(
    x_arrays, bins=x_bins, stacked=True,
    color=[basin_colors[bt] for bt in basin_colors.keys()],
    edgecolor="k", linewidth=0.5
)
ax_top.set_ylabel("Count", fontsize=18)
ax_top.tick_params(axis="both", labelsize=14)


y_arrays = [df[df[tcol] == bt][ycol].dropna().values for bt in basin_colors.keys()]
ax_right.hist(
    y_arrays, bins=y_bins, stacked=True, orientation="horizontal",
    color=[basin_colors[bt] for bt in basin_colors.keys()],
    edgecolor="k", linewidth=0.5
)
ax_right.set_xlabel("Count", fontsize=18)
ax_right.tick_params(axis="both", labelsize=14)


ax_main.set_ylabel(r"Depth to $V_S$ = 1 km/s, $Z_{1.0}$ (m)", fontsize=18)
ax_main.set_xlabel(r"Average shear-wave velocity of top 30 m, $V_{S30}$ (m/s)", fontsize=18)

ax_main.set_yscale("log")
ax_main.set_xlim([120, 1100])
ax_main.set_ylim([1, None])
ax_main.set_xticks([200, 400, 600, 800, 1000])
ax_main.tick_params(labelsize=14)


ax_right.set_yscale("log")


ax_main.legend(fontsize=14, loc="best", framealpha=0.9)


def onpick(event):
    ind = event.ind[0]
    sc = event.artist
    for i, s in enumerate(scatter_plots):
        if s == sc:
            station = station_ids[i][ind]
            print(f"Clicked on station: {station}")
            ax_main.annotate(
                station,
                (sc.get_offsets()[ind, 0], sc.get_offsets()[ind, 1]),
                xytext=(10, 10), textcoords="offset points",
                fontsize=11, color="black",
                bbox=dict(boxstyle="round,pad=0.25", fc="yellow", alpha=0.6)
            )
            fig.canvas.draw_idle()
            break

fig.canvas.mpl_connect("pick_event", onpick)
from matplotlib.patches import Patch

type_handles = [
    Patch(facecolor=basin_colors[bt], edgecolor="k", label=bt)
    for bt in basin_colors.keys()
]

ax_top.legend(handles=type_handles, loc="upper right",
              fontsize=12, framealpha=0.9, ncols=2)

fig.savefig(os.path.join(outDir,'Z1VsVs30_Stations_NB.pdf'))
#%% Z1 Vs. Vs30 - Two panel plot - (a) Vs30 Vs Z1_nb, and (b) Vs30 Vs ln(Z1_wb/Z1_nb)
fig, (ax_L, ax_R) = plt.subplots(
    1, 2, figsize=(14, 6), sharex=True, constrained_layout=True)
tcol = "BasinType_2p09"
basincolor_map = {
    "Basin": "#1E90FF",
    "Basin Edge": "#66C266",
    "Valley": "#9370DB",
    "Hill": "#993333"
}
for basin_type, color in basin_colors.items():
    subset = df[df[tcol] == basin_type]


    ax_L.scatter(
        subset[xcol], subset["Z1_nb"],
        edgecolor="k", s=60, color=color, zorder=3, label = basin_type
    )


    ax_R.scatter(
        subset[xcol], subset["Z1_ratio"],
        edgecolor="k", s=60, color=color, zorder=3
    )
ax_L.plot(
    x_sorted, y_sm_sorted,
    color="magenta", linewidth=3, zorder=4
)
ax_L.set_yscale("log")
ax_L.set_ylabel(r"$Z_{1.0}$ - no basin (m)", fontsize=16)
ax_L.set_xlabel(r"$V_{S30}$ (m/s)", fontsize=16)
ax_L.set_xlim([120, 1100])
ax_L.set_ylim([None, None])
ax_L.tick_params(labelsize=13)

ax_R.set_yscale("log")
ax_R.set_ylabel(r"$Z_{1.0}$ ratio (wb/nb)", fontsize=16)
ax_R.set_xlabel(r"$V_{S30}$ (m/s)", fontsize=16)
ax_R.set_xlim([120, 1100])
ax_R.tick_params(labelsize=13)


ax_R.axhline(1.0, color="k", linestyle="--", linewidth=2)
ax_L.legend(fontsize=12, loc="best")
#%% Z1 Vs.Z1_ratio plot
fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)

tcol = "BasinType_2p09"

basincolor_map = {
    "Basin": "#1E90FF",
    "Basin Edge": "#66C266",
    "Valley": "#9370DB",
    "Hill": "#993333"
}

for basin_type, color in basin_colors.items():
    subset = df[df[tcol] == basin_type]

    ax.scatter(
        subset["Z1_2p09"], subset["Z1_ratio"],
        edgecolor="k", s=60, color=color,
        zorder=3, label=basin_type
    )

# Reference line
ax.axhline(1.0, color="k", linestyle="--", linewidth=2)

# Axis formatting
ax.set_yscale("log")
ax.set_xlabel(r"$Z_{1.0}$ (m)", fontsize=16)
ax.set_ylabel(r"$Z_{1.0}$ ratio (wb/nb)", fontsize=16)

# ax.set_xlim([120, 1100])
ax.tick_params(labelsize=13)

ax.legend(fontsize=12, loc="best")
#%% Comparison between With Basin and Without Basin bias, standard deviation, dS2S - pSA - Paper 2
fig1 = plt.figure(figsize=(15.02,  6.24))
x = [0, 1, 2, 3, 4, 5]
# x_values = ["PGA", "PGV", "CAV", "AI", "$D_{s575}$", "$D_{s595}$", "$D_{s595,LF}$"]
x_values = ["PGA", "PGV", "CAV", "AI", "$D_{s575}$", "$D_{s595}$"]

gs1 = fig1.add_gridspec(
    nrows=1, ncols=2, width_ratios=[3, 1],
    left=0.07, right=0.46, top=0.97, bottom=0.15,
    wspace=0.05, hspace=0.25
)
ax1 = fig1.add_subplot(gs1[0,0])

ax1.semilogx(T,bias_SA_ff,'b',linewidth=3,label = 'With Basins')
ax1.semilogx(T,bias_SA_nb,'r',linewidth=3,label = 'Without Basins')

ax1.axhline(0,c='k',linestyle='--')
ax1.axvline(1,c='k',linestyle='--')
ax1.set_ylim([-0.9,0.9])
ax1.set_xlim([0.01,10])
ax1.set_ylabel('Model prediction bias, a',size=20)
ax1.set_xlabel('Vibration period, T (s)',size=20)
leg = ax1.legend(fontsize=18,loc='upper left')
leg.get_frame().set_edgecolor('k')
ax1.text(0.011,-0.5,'Overprediction',size=20,fontstyle='italic')
ax1.text(0.011,0.45,'Underprediction',size=20,fontstyle='italic')
ax1.tick_params(labelsize=16,direction='in', axis='both', which='both')
# ax1.grid(color='gray', linestyle='dashed', which='both', linewidth=0.4,alpha=0.5)
ax1.tick_params(axis='x', pad=5)
ax2 = fig1.add_subplot(gs1[0,1])
ax2.set_xticks(x, x_values, size=14)
ax2.scatter(x_values, bias_IMs_nb, s=20, c='r', marker='o')
ax2.scatter(x_values, bias_IMs_ff, s=20, c='b', marker='o')
ax2.plot([-1.0, 10.0], [0, 0], color='k', linestyle='--')
ax2.set_xlim([-1.0, 6.0])
ax2.set_xticklabels(x_values, rotation=90)
ax2.set_ylim([-0.9, 0.9])
ax2.tick_params(labelsize=16,direction='in', axis='both', which='both')
ax2.tick_params(axis='x', pad=5)
ax2.grid(color='k', linestyle=(0, (5, 10)), which='major', linewidth=0.4)
ax2.grid(color='k', linestyle=(0, (5, 10)), which='minor', linewidth=0.4)
ax2.yaxis.tick_right()


gs2 = fig1.add_gridspec(
    nrows=1, ncols=2, width_ratios=[3, 1],
    left=0.58, right=0.97, top=0.96, bottom=0.15,
    wspace=0.05, hspace=0.25
)
ax7 = fig1.add_subplot(gs2[0,0])
ax7.semilogx(T,phiS2S_SA_nb,'r',linewidth=3,label = 'Without Basins')
ax7.semilogx(T,phiS2S_SA_ff,'b',linewidth=3,label = 'With Basins')
ax7.set_ylim([0,0.92])
ax7.set_xlim([0.01,10])
ax7.axvline(1,c='k',linestyle='--')
ax7.set_ylabel('Site-to-site standard deviation, $\phi_{S2S}$',size=20)
ax7.set_xlabel('Vibration period, T (s)',size=20)
ax7.tick_params(labelsize=16,direction='in', axis='both', which='both')
# ax7.grid(color='gray', linestyle='dashed', which='both', linewidth=0.4,alpha=0.5)
ax7.tick_params(axis='x', pad=5)
ax8 = fig1.add_subplot(gs2[0,1])
ax8.set_xticks(x, x_values, size=14)
ax8.scatter(x_values, phiS2S_IMs_nb, s=20, c='r', marker='o')
ax8.scatter(x_values, phiS2S_IMs_ff, s=20, c='b', marker='o')
ax8.set_xlim([-1.0, 6.0])
ax8.set_xticklabels(x_values, rotation=90)
ax8.set_ylim([0, 0.92])
ax8.tick_params(labelsize=16,direction='in', axis='both', which='both')
ax8.tick_params(axis='x', pad=5)
ax8.grid(color='k', linestyle=(0, (5, 10)), which='major', linewidth=0.4)
ax8.grid(color='k', linestyle=(0, (5, 10)), which='minor', linewidth=0.4)
ax8.yaxis.set_label_position("right")
ax8.yaxis.tick_right()

fig1.text(0.008, 0.95, '(a)', fontsize=20, fontweight='bold') 
fig1.text(0.50, 0.95, '(b)', fontsize=20, fontweight='bold')

fig1.savefig(os.path.join(outDir,'NoBasin&AllBasin_SA.png'),dpi=300)
fig1.savefig(os.path.join(outDir,'NoBasin&AllBasin_SA.pdf'))
#%% Comparison between With Basin and Without Basin bias, standard deviation, dS2S - EAS - Paper 2
fig1 = plt.figure(figsize=(13.29,4.65))


gs1 = fig1.add_gridspec(
    nrows=1, ncols=1,
    left=0.07, right=0.46, top=0.97, bottom=0.15,
)

ax1 = fig1.add_subplot(gs1[0,0])

ax1.semilogx(f,bias_EAS_ff,'b',linewidth=3,label = 'With Basins')
ax1.semilogx(f,bias_EAS_nb,'r',linewidth=3,label = 'Without Basins')

ax1.axhline(0,c='k',linestyle='--')
ax1.axvline(1,c='k',linestyle='--')
ax1.set_ylim([-1,1])
ax1.set_xlim([0.1,20])
ax1.set_ylabel('Model prediction bias, a',size=20)
ax1.set_xlabel('Frequency, f (Hz)',size=20)
# leg = ax1.legend(fontsize=18,loc='upper right')
# leg.get_frame().set_edgecolor('k')
ax1.text(1.5,-0.6,'Overprediction',size=20,fontstyle='italic')
ax1.text(1.5,0.3,'Underprediction',size=20,fontstyle='italic')
ax1.tick_params(labelsize=16,direction='in', axis='both', which='both')
# ax1.grid(color='gray', linestyle='dashed', which='both', linewidth=0.4,alpha=0.5)
ax1.tick_params(axis='x', pad=5)



gs2 = fig1.add_gridspec(
    nrows=1, ncols=1,
    left=0.58, right=0.97, top=0.96, bottom=0.15,
)
ax7 = fig1.add_subplot(gs2[0,0])
ax7.semilogx(f,phiS2S_EAS_nb,'r',linewidth=3,label = 'Without Basins')
ax7.semilogx(f,phiS2S_EAS_ff,'b',linewidth=3,label = 'With Basins')
ax7.set_ylim([0,1])
ax7.set_xlim([0.1,20])
ax7.axvline(1,c='k',linestyle='--')
ax7.set_ylabel('Site-to-site standard deviation, $\phi_{S2S}$',size=20)
ax7.set_xlabel('Frequency, f (Hz)',size=20)
ax7.tick_params(labelsize=16,direction='in', axis='both', which='both')
# ax7.grid(color='gray', linestyle='dashed', which='both', linewidth=0.4,alpha=0.5)
ax7.tick_params(axis='x', pad=5)


fig1.text(0.008, 0.95, '(c)', fontsize=20, fontweight='bold') 
fig1.text(0.50, 0.95, '(d)', fontsize=20, fontweight='bold')


fig1.savefig(os.path.join(outDir,'NoBasin&AllBasin_EAS.pdf'))

#%% Comparison between With and Without Basins full bias and standard deviation, dS2S - pSA
fig1 = plt.figure(figsize=(24.3 , 11.55))
x = [0, 1, 2, 3, 4, 5]
x_values = ["PGA", "PGV", "CAV", "AI", "$D_{s575}$", "$D_{s595}$"]
gs1 = fig1.add_gridspec(nrows=2, ncols=2, width_ratios=[3,1], left=0.05, right=0.30,top=0.98, bottom=0.08,wspace=0.05,hspace=0.25)
ax1 = fig1.add_subplot(gs1[0,0])
ax1.semilogx(T,bias_SA_ff,'b',linewidth=3,label = 'With Basins')
ax1.semilogx(T,bias_SA_nb,'r',linewidth=3,label = 'Without Basins')
ax1.axhline(0,c='k',linestyle='--')
ax1.axvline(1,c='k',linestyle='--')
ax1.set_ylim([-0.9,0.9])
ax1.set_xlim([0.01,10])
ax1.set_ylabel('Model prediction bias, a',size=20)
ax1.set_xlabel('Vibration period, T (s)',size=20)
leg = ax1.legend(fontsize=18,loc='upper left')
leg.get_frame().set_edgecolor('k')
ax1.text(0.011,-0.46,'Overprediction',size=20,fontstyle='italic')
ax1.text(0.011,0.22,'Underprediction',size=20,fontstyle='italic')
ax1.tick_params(labelsize=16,direction='in', axis='both', which='both')
# ax1.grid(color='gray', linestyle='dashed', which='both', linewidth=0.4,alpha=0.5)
ax2 = fig1.add_subplot(gs1[0,1])
ax2.set_xticks(x, x_values, size=14)
ax2.scatter(x_values, bias_IMs_nb, s=20, c='r', marker='o')
ax2.scatter(x_values, bias_IMs_ff, s=20, c='b', marker='o')
ax2.plot([-1.0, 10.0], [0, 0], color='k', linestyle='--')
ax2.set_xlim([-1.0, 6.0])
ax2.set_xticklabels(x_values, rotation=90)
ax2.set_ylim([-0.9, 0.9])
ax2.tick_params(labelsize=16,direction='in', axis='both', which='both')
ax2.grid(color='gray', linestyle='dashed', which='both', linewidth=0.4,alpha=0.5)
ax2.grid(color='gray', linestyle='dashed', which='both', linewidth=0.4,alpha=0.5)
ax2.yaxis.set_label_position("right")
ax2.yaxis.tick_right()

gs2 = fig1.add_gridspec(nrows=2, ncols=2, width_ratios=[3,1], left=0.39, right=0.64, top=0.98, bottom=0.08, wspace=0.05,hspace=0.25)
ax3 = fig1.add_subplot(gs2[0,0])
ax3.semilogx(T,sigma_SA_ff,'b',linewidth=3,label = 'With Basins')
ax3.semilogx(T,sigma_SA_nb,'r',linewidth=3,label = 'Without Basins')
ax3.set_ylim([0,1])
ax3.set_xlim([0.01,10])
ax3.axvline(1,c='k',linestyle='--')
ax3.set_ylabel('Total standard deviation, $\sigma$',size=20)
ax3.set_xlabel('Vibration period, T (s)',size=20)
ax3.tick_params(labelsize=16,direction='in', axis='both', which='both')
# ax3.grid(color='gray', linestyle='dashed', which='both', linewidth=0.4,alpha=0.5) 
ax4 = fig1.add_subplot(gs2[0,1])
ax4.set_xticks(x, x_values, size=14)
ax4.scatter(x_values, sigma_IMs_nb, s=20, c='r', marker='o')
ax4.scatter(x_values, sigma_IMs_ff, s=20, c='b', marker='o')
ax4.set_xlim([-1.0, 6.0])
ax4.set_xticklabels(x_values, rotation=90)
ax4.set_ylim([0, 1])
ax4.tick_params(labelsize=16,direction='in', axis='both', which='both')
ax4.grid(color='gray', linestyle='dashed', which='both', linewidth=0.4,alpha=0.5)
ax4.grid(color='gray', linestyle='dashed', which='both', linewidth=0.4,alpha=0.5)
ax4.yaxis.set_label_position("right")
ax4.yaxis.tick_right()

ax5 = fig1.add_subplot(gs1[1,0], sharex=ax1)
ax5.semilogx(T,tau_SA_ff,'b',linewidth=3,label = 'With Basins')
ax5.semilogx(T,tau_SA_nb,'r',linewidth=3,label = 'Without Basins')
ax5.set_ylim([0,0.8])
ax5.set_xlim([0.01,10])
ax5.axvline(1,c='k',linestyle='--')
ax5.set_ylabel(r'Between-event standard deviation, $\tau$',size=20)
ax5.set_xlabel('Vibration period, T (s)',size=20)
ax5.tick_params(labelsize=16,direction='in', axis='both', which='both')
# ax5.grid(color='gray', linestyle='dashed', which='both', linewidth=0.4,alpha=0.5)
fig1.text(0.001, 0.98, '(a)', fontsize=20, fontweight='bold') 
fig1.text(0.34, 0.98, '(b)', fontsize=20, fontweight='bold')
fig1.text(0.001, 0.50, '(c)', fontsize=20, fontweight='bold')
fig1.text(0.34, 0.50, '(d)', fontsize=20, fontweight='bold')
fig1.text(0.68, 0.50, '(e)', fontsize=20, fontweight='bold')
leg = ax5.legend(fontsize=18,loc='lower left')
leg.get_frame().set_edgecolor('k')
ax6 = fig1.add_subplot(gs1[1,1])
ax6.set_xticks(x, x_values, size=14)
ax6.scatter(x_values, tau_IMs_nb, s=20, c='r', marker='o')
ax6.scatter(x_values, tau_IMs_ff, s=20, c='b', marker='o')
ax6.set_xlim([-1.0, 6.0])
ax6.set_xticklabels(x_values, rotation=90)
ax6.set_ylim([0, 0.8])
ax6.tick_params(labelsize=16,direction='in', axis='both', which='both')
ax6.grid(color='gray', linestyle='dashed', which='both', linewidth=0.4,alpha=0.5)
ax6.grid(color='gray', linestyle='dashed', which='both', linewidth=0.4,alpha=0.5)
ax6.yaxis.set_label_position("right")
ax6.yaxis.tick_right()

ax7 = fig1.add_subplot(gs2[1,0], sharex=ax3)
ax7.semilogx(T,phiS2S_SA_ff,'b',linewidth=3,label = 'With Basins')
ax7.semilogx(T,phiS2S_SA_nb,'r',linewidth=3,label = 'Without Basins')
ax7.set_ylim([0,0.8])
ax7.set_xlim([0.01,10])
ax7.axvline(1,c='k',linestyle='--')
ax7.set_ylabel('Site-to-site standard deviation, $\phi_{S2S}$',size=20)
ax7.set_xlabel('Vibration period, T (s)',size=20)
ax7.tick_params(labelsize=16,direction='in', axis='both', which='both')
# ax7.grid(color='gray', linestyle='dashed', which='both', linewidth=0.4,alpha=0.5)
ax8 = fig1.add_subplot(gs2[1,1])
ax8.set_xticks(x, x_values, size=14)
ax8.scatter(x_values, phiS2S_IMs_nb, s=20, c='r', marker='o')
ax8.scatter(x_values, phiS2S_IMs_ff, s=20, c='b', marker='o')
ax8.set_xlim([-1.0, 6.0])
ax8.set_xticklabels(x_values, rotation=90)
ax8.set_ylim([0, 0.8])
ax8.tick_params(labelsize=16,direction='in', axis='both', which='both')
ax8.grid(color='gray', linestyle='dashed', which='both', linewidth=0.4,alpha=0.5)
ax8.grid(color='gray', linestyle='dashed', which='both', linewidth=0.4,alpha=0.5)
ax8.yaxis.set_label_position("right")
ax8.yaxis.tick_right()

gs3 = fig1.add_gridspec(nrows=2, ncols=2, width_ratios=[3,1], left=0.73, right=0.96,top=0.98, bottom=0.08, wspace=0.05,hspace=0.25)
ax9 = fig1.add_subplot(gs3[1,0])
ax9.semilogx(T,phiss_SA_ff,'b',linewidth=3,label = 'With Basins')
ax9.semilogx(T,phiss_SA_nb,'r',linewidth=3,label = 'Without Basins')
ax9.set_ylim([0,0.8])
ax9.set_xlim([0.01,10])
ax9.axvline(1,c='k',linestyle='--')
ax9.set_ylabel('Single station standard deviation, $\phi_{SS}$',size=20)
ax9.set_xlabel('Vibration period, T (s)',size=20)
ax9.tick_params(labelsize=16,direction='in', axis='both', which='both')
# ax9.grid(color='gray', linestyle='dashed', which='both', linewidth=0.4,alpha=0.5)
ax10 = fig1.add_subplot(gs3[1,1])
ax10.set_xticks(x, x_values, size=14)
ax10.scatter(x_values, phiss_IMs_nb, s=20, c='r', marker='o')
ax10.scatter(x_values, phiss_IMs_ff, s=20, c='b', marker='o')
ax10.set_xlim([-1.0, 6.0])
ax10.set_xticklabels(x_values, rotation=90)
ax10.set_ylim([0, 0.8])
ax10.tick_params(labelsize=16,direction='in', axis='both', which='both')
ax10.grid(color='gray', linestyle='dashed', which='both', linewidth=0.4,alpha=0.5)
ax10.grid(color='gray', linestyle='dashed', which='both', linewidth=0.4,alpha=0.5)
ax10.yaxis.set_label_position("right")
ax10.yaxis.tick_right()

fig1.savefig(os.path.join(outDir,'Full_NoBasin&AllBasin_SA.pdf'))
#%% Comparison between With and Without Basins full bias and standard deviation, dS2S - EAS
fig1 = plt.figure(figsize=(19.2, 9.83))


gs1 = fig1.add_gridspec(
    nrows=2, ncols=1, width_ratios=[1],
    left=0.05, right=0.30, top=0.98, bottom=0.08,
    wspace=0.05, hspace=0.25
)


ax1 = fig1.add_subplot(gs1[0,0])
ax1.semilogx(f,bias_EAS_ff,'b',linewidth=3,label = 'With Basins')
ax1.semilogx(f,bias_EAS_nb,'r',linewidth=3,label = 'Without Basins')
ax1.axhline(0,c='k',linestyle='--')
ax1.axvline(1,c='k',linestyle='--')
ax1.set_ylim([-1,1])
ax1.set_xlim([0.1,20])
ax1.set_ylabel('Model prediction bias, a',size=20)
ax1.set_xlabel('Frequency, f (Hz)',size=20)
ax1.text(1.5,-0.6,'Overprediction',size=20,fontstyle='italic')
ax1.text(1.5,0.3,'Underprediction',size=20,fontstyle='italic')
ax1.tick_params(labelsize=16,direction='in', axis='both', which='both')
# ax1.grid(color='gray', linestyle='dashed', which='both', linewidth=0.4,alpha=0.5)
ax1.legend(fontsize=18,loc='upper right')

gs2 = fig1.add_gridspec(
    nrows=2, ncols=1, width_ratios=[1],
    left=0.39, right=0.64, top=0.98, bottom=0.08,
    wspace=0.05, hspace=0.25
)

ax3 = fig1.add_subplot(gs2[0,0])

ax3.semilogx(f,sigma_EAS_ff,'b',linewidth=3,label = 'With Basins')
ax3.semilogx(f,sigma_EAS_nb,'r',linewidth=3,label = 'Without Basins')
ax3.set_ylim([0,1.1])
ax3.set_xlim([0.1,20])
ax3.axvline(1,c='k',linestyle='--')
ax3.set_ylabel('Total standard deviation, $\sigma$',size=20)
ax3.set_xlabel('Frequency, f (Hz)',size=20)
ax3.tick_params(labelsize=16,direction='in', axis='both', which='both')
# ax3.grid(color='gray', linestyle='dashed', which='both', linewidth=0.4,alpha=0.5) 


ax5 = fig1.add_subplot(gs1[1,0], sharex=ax1)
ax5.semilogx(f,tau_EAS_ff,'b',linewidth=3,label = 'With Basins')
ax5.semilogx(f,tau_EAS_nb,'r',linewidth=3,label = 'Without Basins')
ax5.set_ylim([0,1])
ax5.set_xlim([0.1,20])
ax5.axvline(1,c='k',linestyle='--')
ax5.set_ylabel(r'Between-event standard deviation, $\tau$',size=20)
ax5.set_xlabel('Frequency, f (Hz)',size=20)
ax5.tick_params(labelsize=16,direction='in', axis='both', which='both')
# ax5.grid(color='gray', linestyle='dashed', which='both', linewidth=0.4,alpha=0.5)
ax5.legend(fontsize=18,loc='lower left')

fig1.text(0.001, 0.98, '(a)', fontsize=20, fontweight='bold')
fig1.text(0.001, 0.50, '(c)', fontsize=20, fontweight='bold')


ax7 = fig1.add_subplot(gs2[1,0], sharex=ax3)
ax7.semilogx(f,phiS2S_EAS_ff,'b',linewidth=3,label = 'With Basins')
ax7.semilogx(f,phiS2S_EAS_nb,'r',linewidth=3,label = 'Without Basins')
ax7.set_ylim([0,1])
ax7.set_xlim([0.1,20])
ax7.axvline(1,c='k',linestyle='--')
ax7.set_ylabel('Site-to-site standard deviation, $\phi_{S2S}$',size=20)
ax7.set_xlabel('Frequency, f (Hz)',size=20)
ax7.tick_params(labelsize=16,direction='in', axis='both', which='both')
# ax7.grid(color='gray', linestyle='dashed', which='both', linewidth=0.4,alpha=0.5)

fig1.text(0.34, 0.98, '(b)', fontsize=20, fontweight='bold')
fig1.text(0.34, 0.50, '(d)', fontsize=20, fontweight='bold')


gs3 = fig1.add_gridspec(
    nrows=2, ncols=1, width_ratios=[1],
    left=0.73, right=0.96, top=0.98, bottom=0.08
)


ax9 = fig1.add_subplot(gs3[1,0])
ax9.semilogx(f,phiss_EAS_ff,'b',linewidth=3,label = 'With Basins')
ax9.semilogx(f,phiss_EAS_nb,'r',linewidth=3,label = 'Without Basins')
ax9.set_ylim([0,1])
ax9.set_xlim([0.1,20])
ax9.axvline(1,c='k',linestyle='--')
ax9.set_ylabel('Single station standard deviation, $\phi_{SS}$',size=20)
ax9.set_xlabel('Frequency, f (Hz)',size=20)
ax9.tick_params(labelsize=16,direction='in', axis='both', which='both')
# ax9.grid(color='gray', linestyle='dashed', which='both', linewidth=0.4,alpha=0.5)

fig1.text(0.68, 0.50, '(e)', fontsize=20, fontweight='bold')


fig1.savefig(os.path.join(outDir,'Full_NoBasin&AllBasin_EAS.pdf'))
#%% % difference in phi_S2S - EAS
phiS2S_reduction_pct = 100.0 * (phiS2S_EAS_nb - phiS2S_EAS_ff) / phiS2S_EAS_nb
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(14, 6), sharex=True,constrained_layout=True)
axs[0].semilogx(f, phiS2S_EAS_nb, linewidth=2,color='r', label='Without Basins')
axs[0].semilogx(f, phiS2S_EAS_ff, linewidth=2,color='b', label='With Basins')
axs[0].axvline(1, linestyle='--')
axs[0].set_xlim([0.1, 20])
axs[0].set_ylim([0, 1])
axs[0].set_xlabel('Frequency, f (Hz)', fontsize=16)
axs[0].set_ylabel(r'Site-to-site standard deviation, $\phi_{S2S}$', fontsize=16)
axs[0].tick_params(labelsize=13, direction='in', which='both')
axs[0].grid(color='gray', linestyle='dashed', linewidth=0.4)
axs[0].legend(fontsize=14)
axs[1].semilogx(f, phiS2S_reduction_pct, linewidth=2,color='k')
axs[1].axhline(0, linestyle='--')
axs[1].axvline(1, linestyle='--')
axs[1].set_xlim([0.1, 20])
axs[1].set_ylim([-25, 25])
axs[1].set_xlabel('Frequency, f (Hz)', fontsize=16)
axs[1].set_ylabel(r'$\phi_{S2S}$ reduction (%)', fontsize=16)
axs[1].tick_params(labelsize=13, direction='in', which='both')
axs[1].grid(color='gray', linestyle='dashed', linewidth=0.4)

fig.savefig(os.path.join(outDir, 'PhiS2S_comparison_and_reduction_EAS.pdf'), bbox_inches='tight')
#%% % difference in phi_S2S - pSA
phiS2S_reduction_pct = 100.0 * (phiS2S_SA_nb - phiS2S_SA_ff) / phiS2S_SA_nb
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(14, 6), sharex=True,constrained_layout=True)
axs[0].semilogx(T, phiS2S_SA_nb, linewidth=2,color='r', label='Without Basins')
axs[0].semilogx(T, phiS2S_SA_ff, linewidth=2,color='b', label='With Basins')
axs[0].axvline(1, linestyle='--')
axs[0].set_xlim([0.01, 10])
axs[0].set_ylim([0, 1])
axs[0].set_xlabel('Vibration Period, T (s)', fontsize=16)
axs[0].set_ylabel(r'Site-to-site standard deviation, $\phi_{S2S}$', fontsize=16)
axs[0].tick_params(labelsize=13, direction='in', which='both')
axs[0].grid(color='gray', linestyle='dashed', linewidth=0.4)
axs[0].legend(fontsize=14)
axs[1].semilogx(T, phiS2S_reduction_pct, linewidth=2,color='k')
axs[1].axhline(0, linestyle='--')
axs[1].axvline(1, linestyle='--')
axs[1].set_xlim([0.01, 10])
axs[1].set_ylim([-25, 25])
axs[1].set_xlabel('Vibration Period, T (s)', fontsize=16)
axs[1].set_ylabel(r'$\phi_{S2S}$ reduction (%)', fontsize=16)
axs[1].tick_params(labelsize=13, direction='in', which='both')
axs[1].grid(color='gray', linestyle='dashed', linewidth=0.4)

fig.savefig(os.path.join(outDir, 'PhiS2S_comparison_and_reduction_pSA.pdf'), bbox_inches='tight')
#%% % difference in a - EAS
f = np.array(f)
bias_reduction_pct = np.abs(bias_EAS_nb) - np.abs(bias_EAS_ff)
mask = (f >= 0.2) & (f <= 1.0)     
pct_band_EAS = 100 * (1 - np.nanmean(np.abs(bias_EAS_ff)[mask]) / np.nanmean(np.abs(bias_EAS_nb)[mask]))
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(14, 6), sharex=True,constrained_layout=True)
axs[0].semilogx(f, bias_EAS_nb, linewidth=2,color='r', label='Without Basins')
axs[0].semilogx(f, bias_EAS_ff, linewidth=2,color='b', label='With Basins')
axs[0].axhline(0, color='k', linestyle='--')
axs[0].axvline(1, color='k', linestyle='--')
axs[0].set_xlim([0.1, 20])
axs[0].set_ylim([-1, 1])
axs[0].set_xlabel('Frequency, f (Hz)', fontsize=16)
axs[0].set_ylabel(r'$a$', fontsize=16)
axs[0].tick_params(labelsize=13, direction='in', which='both')
axs[0].grid(color='gray', linestyle='dashed', linewidth=0.4)
axs[0].legend(fontsize=14)
axs[1].semilogx(f, bias_reduction_pct, linewidth=2,color='k')
axs[1].axhline(0, linestyle='--')
axs[1].axvline(1, linestyle='--')
axs[1].set_xlim([0.1, 20])
axs[1].set_ylim([-0.25, 0.25])
axs[1].set_xlabel('Frequency, f (Hz)', fontsize=14)
axs[1].set_ylabel(r"Reduction in bias, $a$: $|a_{\mathrm{without\ basins}}| - |a_{\mathrm{with\ basins}}|$",fontsize=14)
axs[1].tick_params(labelsize=13, direction='in', which='both')
axs[1].grid(color='gray', linestyle='dashed', linewidth=0.4)
fig.savefig(os.path.join(outDir, 'bias_comparison_and_reduction_EAS.pdf'), bbox_inches='tight')
#%% % difference in a  - pSA
T = np.array(T)
bias_reduction_pct = np.abs(bias_SA_nb) - np.abs(bias_SA_ff)
mask = (T >= 1.0) & (T <= 5.0)     
pct_band_pSA = 100 * (1 - np.nanmean(np.abs(bias_SA_ff)[mask]) / np.nanmean(np.abs(bias_SA_nb)[mask]))
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(14, 6), sharex=True,constrained_layout=True)
axs[0].semilogx(T, bias_SA_nb, linewidth=2,color='r', label='Without Basins')
axs[0].semilogx(T, bias_SA_ff, linewidth=2,color='b', label='With Basins')
axs[0].axhline(0, color='k', linestyle='--')
axs[0].axvline(1, color='k', linestyle='--')
axs[0].set_xlim([0.01, 10])
axs[0].set_ylim([-0.5, 0.5])
axs[0].set_xlabel('Vibration Period, T (s)', fontsize=16)
axs[0].set_ylabel(r'${a}$', fontsize=16)
axs[0].tick_params(labelsize=13, direction='in', which='both')
axs[0].grid(color='gray', linestyle='dashed', linewidth=0.4)
axs[0].legend(fontsize=14)
axs[1].semilogx(T, bias_reduction_pct, linewidth=2,color='k')
axs[1].axhline(0, linestyle='--')
axs[1].axvline(1, linestyle='--')
axs[1].set_xlim([0.01, 10])
axs[1].set_ylim([-0.12, 0.12])
axs[1].set_xlabel('Vibration Period, T (s)', fontsize=16)
axs[1].set_ylabel(r"Reduction in bias, $a$: $|a_{\mathrm{without\ basins}}| - |a_{\mathrm{with\ basins}}|$",fontsize=14)

axs[1].tick_params(labelsize=13, direction='in', which='both')
axs[1].grid(color='gray', linestyle='dashed', linewidth=0.4)

fig.savefig(os.path.join(outDir, 'bias_comparison_and_reduction_pSA.pdf'), bbox_inches='tight')
#%% Which basin max improvement
from matplotlib.lines import Line2D
basin = stations_sel_sorted['Basin']
basin_labels = pd.Index(basin).unique().tolist()
colors = ['b','g','r','maroon']


EAS_cols = [c for c in siteres_ff.columns if c.startswith("EAS_")]

siteres_ff_EAS  = siteres_ff_sorted[EAS_cols]
siteres_nb_EAS  = siteres_nb_sorted[EAS_cols]

sitereserr_ff_EAS  = sitereserr_ff_sorted[EAS_cols]
sitereserr_nb_EAS  = sitereserr_nb_sorted[EAS_cols]

out_dir = "Basin_performance/EAS"
os.makedirs(out_dir, exist_ok=True)

basin = stations_sel_sorted['Basin']


siteres_systematic_ff_EAS = siteres_ff_EAS + np.array(bias_EAS_ff)
siteres_systematic_nb_EAS = siteres_nb_EAS + np.array(bias_EAS_nb)

grouped_data_ff_EAS = siteres_systematic_ff_EAS.groupby(basin)
grouped_data_nb_EAS = siteres_systematic_nb_EAS.groupby(basin)


basin_results = []
i = 0
for basin_name in basin_labels:


    if (basin_name not in grouped_data_ff_EAS.groups) or (basin_name not in grouped_data_nb_EAS.groups):
        continue

    ff_cluster = grouped_data_ff_EAS.get_group(basin_name)
    nb_cluster = grouped_data_nb_EAS.get_group(basin_name)


    ff_mean = ff_cluster.mean(axis=0)
    nb_mean = nb_cluster.mean(axis=0)

    diff = nb_mean - ff_mean

    basin_results.append({
        "Basin": basin_name,
        "diff_vector": diff,                 
        "max_diff": diff.max(),             
        "T_at_max": T[np.argmax(diff)]       
    })

    i += 1

df_basin = pd.DataFrame(basin_results)

df_basin_sorted = df_basin.sort_values(by="max_diff", ascending=False)

print(df_basin_sorted) 
#%% Box-and-whisker plot of Z1,sim for all basins - Paper 2
Z1_col = "Z1_2p09"
exclude_basins = ["BPVOutcrops"]
out_dir = "Basin_performance/Z1"
os.makedirs(out_dir, exist_ok=True)

meta = stations_sel_sorted.reset_index(drop=True).copy()
meta["Basin_clean"] = meta["Basin"].astype(str).str.strip()
meta["Z1_sim"] = pd.to_numeric(meta[Z1_col], errors="coerce")

station_mask = (
    meta["Basin"].notna()
    & ~meta["Basin_clean"].isin(exclude_basins)
    & meta["Z1_sim"].notna()
)
plot_df = meta.loc[station_mask, ["Basin_clean", "Z1_sim"]].copy()
plot_df = plot_df.rename(columns={"Basin_clean": "Basin"})

basin_count = plot_df.groupby("Basin")["Z1_sim"].size()
basin_order = sorted(
    basin_count.index.tolist(),
    key=lambda b: (-basin_count.loc[b], b.lower())
)
counts = basin_count.loc[basin_order]

summary_table = (
    plot_df.groupby("Basin")["Z1_sim"]
    .agg(["count", "median", "mean", "std", "min", "max"])
    .loc[basin_order]
    .round(1)
)
print("\nBasin-wise Z1,sim summary:")
print(summary_table)

data = [
    plot_df.loc[plot_df["Basin"] == b, "Z1_sim"].values
    for b in basin_order
]
xpos = np.arange(len(basin_order))

fig, ax = plt.subplots(
    figsize=(max(12, 0.45 * len(basin_order)), 6),
    constrained_layout=True,
)

# --- (a) conditional rendering by station count -----------------------------
n_per_basin = {b: len(d) for b, d in zip(basin_order, data)}
box_idx    = [i for i, b in enumerate(basin_order) if n_per_basin[b] >= 3]
line_idx   = [i for i, b in enumerate(basin_order) if n_per_basin[b] == 2]
# single-station basins: scatter only, handled below

# (b) lighter gray box
if box_idx:
    ax.boxplot(
        [data[i] for i in box_idx],
        positions=[xpos[i] for i in box_idx],
        widths=0.55,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color="black", linewidth=2.2),
        boxprops=dict(facecolor="#ececec", edgecolor="black", linewidth=1.2),
        whiskerprops=dict(color="black", linewidth=1.2),
        capprops=dict(color="black", linewidth=1.2),
    )

# horizontal median line for n == 2 (use log-space median since y is log)
half_w = 0.55 / 2
for i in line_idx:
    med = np.median(data[i])
    ax.hlines(med, xpos[i] - half_w, xpos[i] + half_w,
              color="black", linewidth=2.2, zorder=2)

# --- (d) single-color points (no Z1 color coding) ---------------------------
rng = np.random.default_rng(42)
point_color = "b"  # muted blue; swap to taste

for i, b in enumerate(basin_order):
    y = plot_df.loc[plot_df["Basin"] == b, "Z1_sim"].values
    x = rng.normal(loc=xpos[i], scale=0.055, size=len(y))
    ax.scatter(
        x, y,
        s=40,
        color=point_color,
        alpha=0.75,
        edgecolor="black",
        linewidth=0.35,
        zorder=3,
    )

ax.set_xticks(xpos)
ax.set_xticklabels(basin_order, rotation=90)
ax.set_yscale("log")
ax.set_ylabel(r"$Z_{1.0,\mathrm{sim}}$ (m)", fontsize=16)
ax.set_xlabel("Basins in NZVM v2.09", fontsize=20)
ax.tick_params(labelsize=14, direction="in", axis="both", which="both")
ax.grid(True, axis="y", which="both", linestyle=":", linewidth=0.8)
ax.grid(True, axis="x", linestyle=":", linewidth=0.5, alpha=0.5)

# --- (c) N labels along the top --------------------------------------------
# Extend upper y-limit in log space so labels don't collide with whiskers
y_lo, y_hi = ax.get_ylim()
ax.set_ylim(y_lo, y_hi * 1.9)  # multiplicative because log

# Place labels at a fixed axes-fraction position (robust on log axes)
for i, b in enumerate(basin_order):
    label = f"{counts[b]}" if i == 0 else f"{counts[b]}"
    ax.text(xpos[i], 0.97, label,ha="center", va="top", fontsize=16,transform=ax.get_xaxis_transform())

fig.savefig(os.path.join(outDir, "Allbasins_Z1sim_boxplot.pdf"))
#%% Shallow basins - pSA - Paper 2
from matplotlib.lines import Line2D

basin_labels = ["West Coast", "North Canterbury", "Wellington"]
colors = ['r','b','g']


Z1_col = "Z1_2p09"   # change if needed

pSA_cols = [c for c in siteres_ff.columns if c.startswith("pSA_")]

siteres_ff_pSA  = siteres_ff_sorted[pSA_cols]
siteres_nb_pSA  = siteres_nb_sorted[pSA_cols]

sitereserr_ff_pSA  = sitereserr_ff_sorted[pSA_cols]
sitereserr_nb_pSA  = sitereserr_nb_sorted[pSA_cols]

out_dir = "Basin_performance/pSA"
os.makedirs(out_dir, exist_ok=True)

# Clean basin names
basin = stations_sel_sorted["Basin"].astype(str).str.strip()

# Median Z1,sim for each basin
median_z1_by_basin = (
    stations_sel_sorted
    .assign(Basin_clean=basin)
    .groupby("Basin_clean")[Z1_col]
    .apply(lambda z: np.exp(np.log(z).mean()))
)

# Basin legend handles with median Z1,sim
basin_handles = []
for i, basin_name in enumerate(basin_labels):
    median_z1 = median_z1_by_basin.loc[basin_name]
    basin_handles.append(
        Line2D(
            [0], [0],
            color=colors[i],
            lw=2,
            label=fr"{basin_name} ($\overline{{Z}}_{{1.0,\mathrm{{sim}}}}$ = {median_z1:.0f} m)"
        )
    )

style_handles = [
    Line2D([0], [0], color='k', lw=2, linestyle='-', label='With Basins'),
    Line2D([0], [0], color='k', lw=2, linestyle='--',  label='Without Basins')
    
]

siteres_systematic_ff_pSA = siteres_ff_pSA + np.array(bias_SA_ff)
siteres_systematic_nb_pSA = siteres_nb_pSA + np.array(bias_SA_nb)

grouped_data_ff_pSA = siteres_systematic_ff_pSA.groupby(basin)
grouped_data_nb_pSA = siteres_systematic_nb_pSA.groupby(basin)

fig, axes = plt.subplots(1, 1, figsize=(7.06, 5.36), constrained_layout=True)

for i, basin_name in enumerate(basin_labels):

    if (basin_name not in grouped_data_ff_pSA.groups) or (basin_name not in grouped_data_nb_pSA.groups):
        continue

    ff_cluster = grouped_data_ff_pSA.get_group(basin_name)
    nb_cluster = grouped_data_nb_pSA.get_group(basin_name)

    ff_mean = ff_cluster.mean(axis=0)
    nb_mean = nb_cluster.mean(axis=0)

    axes.semilogx(
        T, nb_mean,
        linewidth=2,
        color=colors[i],
        linestyle='--'
    )

    axes.semilogx(
        T, ff_mean,
        linewidth=2,
        color=colors[i],
        linestyle='-'
    )

axes.set_xlim([0.01, 10])
axes.set_ylim([-1.5, 1.5])

axes.axhline(0, color='k', linestyle='--')
axes.axvline(1, color='k', linestyle='--')

axes.set_xlabel('Vibration Period, T (s)', size=18)
axes.set_ylabel(r'${ a + \delta S2S_s}$', fontsize=18)

axes.tick_params(labelsize=14, direction='in', axis='both', which='both')

# Basin legend (colors + median Z1,sim)
leg1 = axes.legend(
    handles=basin_handles,
    loc='lower left',
    fontsize=14,
    frameon=True
)
axes.add_artist(leg1)

# Style legend (line types)
axes.legend(
    handles=style_handles,
    loc='best',
    fontsize=16,
    frameon=True
)

fig.text(0.008, 0.95, '(a)', fontsize=24, fontweight='bold')
axes.text(1.11, -1.3, 'Overprediction', size=20, fontstyle='italic')
axes.text(1.05,  1.3, 'Underprediction', size=20, fontstyle='italic')

fig.savefig(os.path.join(outDir, "Shallowbasins_pSA.pdf"))
#%% Deeper basins - Paper 2
from matplotlib.lines import Line2D

basin_labels = ["Marlborough", "Canterbury", "Palmerston North"]
colors = ['r', 'b', 'g']
def gm_pos(z):
    z = z[z > 0]
    return np.exp(np.log(z).mean()) if len(z) else np.nan

# Column containing Z1,sim
Z1_col = "Z1_2p09"   # change this if your column name is different

pSA_cols = [c for c in siteres_ff.columns if c.startswith("pSA_")]

siteres_ff_pSA  = siteres_ff_sorted[pSA_cols]
siteres_nb_pSA  = siteres_nb_sorted[pSA_cols]

sitereserr_ff_pSA  = sitereserr_ff_sorted[pSA_cols]
sitereserr_nb_pSA  = sitereserr_nb_sorted[pSA_cols]

out_dir = "Basin_performance/pSA"
os.makedirs(out_dir, exist_ok=True)

# Clean basin names for consistent matching
basin = stations_sel_sorted["Basin"].astype(str).str.strip()

# Median Z1,sim for each basin
median_z1_by_basin = (stations_sel_sorted.assign(B=basin).groupby("B")[Z1_col].apply(gm_pos))

# Basin legend with median Z1,sim
basin_handles = []
for i, basin_name in enumerate(basin_labels):
    median_z1 = median_z1_by_basin.loc[basin_name]

    basin_handles.append(
        Line2D(
            [0], [0],
            color=colors[i],
            lw=2,
            label=fr"{basin_name} ($\overline{{Z}}_{{1.0,\mathrm{{sim}}}}$ = {median_z1:.0f} m)")
    )

style_handles = [
    Line2D([0], [0], color='k', lw=2, linestyle='--', label='Without basins'),
    Line2D([0], [0], color='k', lw=2, linestyle='-',  label='With basins')
]

siteres_systematic_ff_pSA = siteres_ff_pSA + np.array(bias_SA_ff)
siteres_systematic_nb_pSA = siteres_nb_pSA + np.array(bias_SA_nb)

grouped_data_ff_pSA = siteres_systematic_ff_pSA.groupby(basin)
grouped_data_nb_pSA = siteres_systematic_nb_pSA.groupby(basin)


fig, axes = plt.subplots(1, 1, figsize=(7.06, 5.36), constrained_layout=True)

for i, basin_name in enumerate(basin_labels):

    if (basin_name not in grouped_data_ff_pSA.groups) or (basin_name not in grouped_data_nb_pSA.groups):
        continue

    ff_cluster = grouped_data_ff_pSA.get_group(basin_name)
    nb_cluster = grouped_data_nb_pSA.get_group(basin_name)

    ff_mean = ff_cluster.mean(axis=0)
    nb_mean = nb_cluster.mean(axis=0)

    axes.semilogx(
        T, nb_mean,
        linewidth=2,
        color=colors[i],
        linestyle='--'
    )

    axes.semilogx(
        T, ff_mean,
        linewidth=2,
        color=colors[i],
        linestyle='-'
    )


axes.set_xlim([0.01, 10])
axes.set_ylim([-1.5, 1.5])

axes.axhline(0, color='k', linestyle='--')
axes.axvline(1, color='k', linestyle='--')

axes.set_ylabel(r'${ a + \delta S2S_s}$', fontsize=18)
axes.set_xlabel('Vibration Period, T (s)', size=18)
axes.tick_params(labelsize=14, direction='in', axis='both', which='both')

# Basin legend with median Z1,sim
leg1 = axes.legend(
    handles=basin_handles,
    loc='lower left',
    fontsize=14,
    frameon=True
)
axes.add_artist(leg1)

# Optional: line-style legend
# leg2 = axes.legend(
#     handles=style_handles,
#     loc='upper left',
#     fontsize=14,
#     frameon=True
# )

fig.text(0.008, 0.95, '(b)', fontsize=24, fontweight='bold') 

fig.savefig(os.path.join(outDir, "Deeperbasins_pSA.pdf"))
#%% Shallow basins - EAS - Paper 2
from matplotlib.lines import Line2D

basin_labels = ["West Coast", "North Canterbury", "Wellington"]
colors = ['r','b','g']


Z1_col = "Z1_2p09"   

EAS_cols = [c for c in siteres_ff.columns if c.startswith("EAS_")]

siteres_ff_EAS  = siteres_ff_sorted[EAS_cols]
siteres_nb_EAS  = siteres_nb_sorted[EAS_cols]

sitereserr_ff_EAS  = sitereserr_ff_sorted[EAS_cols]
sitereserr_nb_EAS  = sitereserr_nb_sorted[EAS_cols]

out_dir = "Basin_performance/EAS"
os.makedirs(out_dir, exist_ok=True)

# Clean basin names
basin = stations_sel_sorted["Basin"].astype(str).str.strip()

# Median Z1,sim for each basin
median_z1_by_basin = (
    stations_sel_sorted
    .assign(Basin_clean=basin)
    .groupby("Basin_clean")[Z1_col]
    .apply(lambda z: np.exp(np.log(z).mean()))
)

# Basin legend handles with median Z1,sim
basin_handles = []
for i, basin_name in enumerate(basin_labels):
    median_z1 = median_z1_by_basin.loc[basin_name]
    basin_handles.append(
        Line2D(
            [0], [0],
            color=colors[i],
            lw=2,
            label=fr"{basin_name} ($\overline{{Z}}_{{1.0,\mathrm{{sim}}}}$ = {median_z1:.0f} m)"
        )
    )

style_handles = [
    Line2D([0], [0], color='k', lw=2, linestyle='-', label='With Basins'),
    Line2D([0], [0], color='k', lw=2, linestyle='--',  label='Without Basins')
    
]

siteres_systematic_ff_EAS = siteres_ff_EAS + np.array(bias_EAS_ff)
siteres_systematic_nb_EAS = siteres_nb_EAS + np.array(bias_EAS_nb)

grouped_data_ff_EAS = siteres_systematic_ff_EAS.groupby(basin)
grouped_data_nb_EAS = siteres_systematic_nb_EAS.groupby(basin)

fig, axes = plt.subplots(1, 1, figsize=(7.06, 5.36), constrained_layout=True)

for i, basin_name in enumerate(basin_labels):

    if (basin_name not in grouped_data_ff_EAS.groups) or (basin_name not in grouped_data_nb_EAS.groups):
        continue

    ff_cluster = grouped_data_ff_EAS.get_group(basin_name)
    nb_cluster = grouped_data_nb_EAS.get_group(basin_name)

    ff_mean = ff_cluster.mean(axis=0)
    nb_mean = nb_cluster.mean(axis=0)

    axes.semilogx(
        f, nb_mean,
        linewidth=2,
        color=colors[i],
        linestyle='--'
    )

    axes.semilogx(
        f, ff_mean,
        linewidth=2,
        color=colors[i],
        linestyle='-'
    )

axes.set_xlim([0.1, 10])
axes.set_ylim([-2, 2])

axes.axhline(0, color='k', linestyle='--')
axes.axvline(1, color='k', linestyle='--')

axes.set_xlabel('Frequency, f (Hz)', size=18)
axes.set_ylabel(r'${ a + \delta S2S_s}$', fontsize=18)

axes.tick_params(labelsize=14, direction='in', axis='both', which='both')

# Basin legend (colors + median Z1,sim)
leg1 = axes.legend(
    handles=basin_handles,
    loc='lower right',
    fontsize=14,
    frameon=True
)
axes.add_artist(leg1)

# Style legend (line types)
axes.legend(
    handles=style_handles,
    loc='upper right',
    fontsize=16,
    frameon=True
)

fig.text(0.008, 0.95, '(a)', fontsize=24, fontweight='bold')
axes.text(0.11, -1.3, 'Overprediction', size=20, fontstyle='italic')
axes.text(0.11,  1.3, 'Underprediction', size=20, fontstyle='italic')

fig.savefig(os.path.join(outDir, "Shallowbasins_EAS.pdf"))
#%% Deeper basins - Paper 2
def gm_pos(z):
    z = z[z > 0]
    return np.exp(np.log(z).mean()) if len(z) else np.nan
from matplotlib.lines import Line2D

basin_labels = ["Marlborough", "Canterbury", "Palmerston North"]
colors = ['r', 'b', 'g']

# Column containing Z1,sim
Z1_col = "Z1_2p09"   # change this if your column name is different

EAS_cols = [c for c in siteres_ff.columns if c.startswith("EAS_")]

siteres_ff_EAS  = siteres_ff_sorted[EAS_cols]
siteres_nb_EAS  = siteres_nb_sorted[EAS_cols]

sitereserr_ff_EAS  = sitereserr_ff_sorted[EAS_cols]
sitereserr_nb_EAS  = sitereserr_nb_sorted[EAS_cols]

out_dir = "Basin_performance/EAS"
os.makedirs(out_dir, exist_ok=True)

# Clean basin names for consistent matching
basin = stations_sel_sorted["Basin"].astype(str).str.strip()

# Median Z1,sim for each basin
median_z1_by_basin = (stations_sel_sorted.assign(B=basin).groupby("B")[Z1_col].apply(gm_pos))

# Basin legend with median Z1,sim
basin_handles = []
for i, basin_name in enumerate(basin_labels):
    median_z1 = median_z1_by_basin.loc[basin_name]

    basin_handles.append(
        Line2D(
            [0], [0],
            color=colors[i],
            lw=2,
            label=fr"{basin_name} ($\overline{{Z}}_{{1.0,\mathrm{{sim}}}}$ = {median_z1:.0f} m)"
    )
)

style_handles = [
    Line2D([0], [0], color='k', lw=2, linestyle='--', label='Without basins'),
    Line2D([0], [0], color='k', lw=2, linestyle='-',  label='With basins')
]

siteres_systematic_ff_EAS = siteres_ff_EAS + np.array(bias_EAS_ff)
siteres_systematic_nb_EAS = siteres_nb_EAS + np.array(bias_EAS_nb)

grouped_data_ff_EAS = siteres_systematic_ff_EAS.groupby(basin)
grouped_data_nb_EAS = siteres_systematic_nb_EAS.groupby(basin)


fig, axes = plt.subplots(1, 1, figsize=(7.06, 5.36), constrained_layout=True)

for i, basin_name in enumerate(basin_labels):

    if (basin_name not in grouped_data_ff_EAS.groups) or (basin_name not in grouped_data_nb_EAS.groups):
        continue

    ff_cluster = grouped_data_ff_EAS.get_group(basin_name)
    nb_cluster = grouped_data_nb_EAS.get_group(basin_name)

    ff_mean = ff_cluster.mean(axis=0)
    nb_mean = nb_cluster.mean(axis=0)

    axes.semilogx(
        f, nb_mean,
        linewidth=2,
        color=colors[i],
        linestyle='--'
    )

    axes.semilogx(
        f, ff_mean,
        linewidth=2,
        color=colors[i],
        linestyle='-'
    )


axes.set_xlim([0.1, 10])
axes.set_ylim([-2, 2])

axes.axhline(0, color='k', linestyle='--')
axes.axvline(1, color='k', linestyle='--')

axes.set_ylabel(r'${ a + \delta S2S_s}$', fontsize=18)
axes.set_xlabel('Frequency, f (Hz)', size=18)
axes.tick_params(labelsize=14, direction='in', axis='both', which='both')

# Basin legend with median Z1,sim
leg1 = axes.legend(
    handles=basin_handles,
    loc='lower left',
    fontsize=14,
    frameon=True
)
axes.add_artist(leg1)

# Optional: line-style legend
# leg2 = axes.legend(
#     handles=style_handles,
#     loc='upper left',
#     fontsize=14,
#     frameon=True
# )

fig.text(0.008, 0.95, '(b)', fontsize=24, fontweight='bold') 

fig.savefig(os.path.join(outDir, "Deeperbasins_EAS.pdf"))
#%% Box-and-whisker improvement plot for all basins - Paper 2 - pSA
summary_mode = "band"   # "single" for 2 s, or "band" for 1-3 s average
target_T = 1.5          # used only if summary_mode = "single"
T_band = (1, 3)     # used only if summary_mode = "band"

exclude_basin_types = ["BPVOutcrops"]

def extract_period(col):
    m = re.search(r"(\d+(?:[._p]\d+)?)$", col)
    if m is None:
        return np.nan
    return float(m.group(1).replace("p", ".").replace("_", "."))


meta = stations_sel_sorted.reset_index(drop=True).copy()

pSA_cols = [c for c in siteres_ff_sorted.columns if c.startswith("pSA_")]

siteres_ff_pSA = siteres_ff_sorted[pSA_cols].reset_index(drop=True).copy()
siteres_nb_pSA = siteres_nb_sorted[pSA_cols].reset_index(drop=True).copy()
bias_ff = pd.Series(np.asarray(bias_SA_ff).ravel(), index=pSA_cols)
bias_nb = pd.Series(np.asarray(bias_SA_nb).ravel(), index=pSA_cols)

siteres_systematic_ff_pSA = siteres_ff_pSA.add(bias_ff, axis="columns")
siteres_systematic_nb_pSA = siteres_nb_pSA.add(bias_nb, axis="columns")


delta_systematic_pSA = siteres_systematic_nb_pSA - siteres_systematic_ff_pSA


basin_type_str = meta["Basin"].astype(str).str.strip()

station_mask = (
    meta["Basin"].notna()
    & ~basin_type_str.isin(exclude_basin_types)
    & meta["Basin"].notna()
)

meta_sel = meta.loc[station_mask].copy()
delta_sel = delta_systematic_pSA.loc[station_mask].copy()


periods = pd.Series(
    {c: extract_period(c) for c in pSA_cols}
).dropna()

if summary_mode == "single":
    selected_col = (periods - target_T).abs().idxmin()
    delta_one_per_site = delta_sel[selected_col]

    plot_label = rf"SA({periods[selected_col]:.2f} s)"
    print(f"Using nearest period to {target_T:.2f} s: {selected_col}, T = {periods[selected_col]:.3f} s")

elif summary_mode == "band":
    band_cols = periods[
        (periods >= T_band[0]) & (periods <= T_band[1])
    ].index.tolist()

    if len(band_cols) == 0:
        raise ValueError(f"No pSA columns found between {T_band[0]} and {T_band[1]} s.")

    delta_one_per_site = delta_sel[band_cols].mean(axis=1)

    plot_label = rf"mean over {T_band[0]:.0f}-{T_band[1]:.0f} s"
    print(f"Using {len(band_cols)} periods between {T_band[0]} and {T_band[1]} s.")

else:
    raise ValueError("summary_mode must be either 'single' or 'band'.")


plot_df = meta_sel[["Basin", "BasinType_2p09", "Z1_2p09"]].copy()
plot_df["delta_resid"] = delta_one_per_site.values
plot_df = plot_df.dropna(subset=["Basin", "delta_resid"]).copy()



basin_count = plot_df.groupby("Basin")["delta_resid"].size()

basin_order = sorted(
    basin_count.index.tolist(),
    key=lambda b: (-basin_count.loc[b], b.lower())
)

counts = basin_count.loc[basin_order]


summary_table = (
    plot_df.groupby("Basin")["delta_resid"]
    .agg(["count", "median", "mean", "std", "min", "max"])
    .loc[basin_order]
    .round(3)
)

print("\nBasin-wise summary:")
print(summary_table)


data = [
    plot_df.loc[plot_df["Basin"] == b, "delta_resid"].values
    for b in basin_order
]

xpos = np.arange(len(basin_order))

fig, ax = plt.subplots(figsize=(max(12, 0.45 * len(basin_order)), 6),constrained_layout=True)

# Split basins by station count
n_per_basin = {b: len(d) for b, d in zip(basin_order, data)}

box_idx     = [i for i, b in enumerate(basin_order) if n_per_basin[b] >= 3]
line_idx    = [i for i, b in enumerate(basin_order) if n_per_basin[b] == 2]
single_idx  = [i for i, b in enumerate(basin_order) if n_per_basin[b] == 1]

# Boxes only for n >= 3
if box_idx:
    ax.boxplot(
        [data[i] for i in box_idx],
        positions=[xpos[i] for i in box_idx],
        widths=0.55,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color="black", linewidth=2.2),
        boxprops=dict(facecolor="#ececec", edgecolor="black", linewidth=1.2),  # see (3)
        whiskerprops=dict(color="black", linewidth=1.2),
        capprops=dict(color="black", linewidth=1.2),
    )

# Horizontal median line only for n == 2
half_w = 0.55 / 2
for i in line_idx:
    med = np.median(data[i])
    ax.hlines(med, xpos[i] - half_w, xpos[i] + half_w,
              color="black", linewidth=2.2, zorder=2)

# n == 1: nothing to draw here; the scatter loop already plots the point

# Individual site points with jitter
from matplotlib import cm, colors

z_all = plot_df["Z1_2p09"].to_numpy()
z_pos = z_all[z_all > 0]
vmin  = 10
vmax  = 1000
# vmin, vmax = np.nanpercentile(z_pos, [2, 99])
norm = colors.LogNorm(vmin=vmin, vmax=vmax)
cmap = cm.jet
rng = np.random.default_rng(42)

for i, b in enumerate(basin_order):
    sub = plot_df.loc[plot_df["Basin"] == b]
    y = sub["delta_resid"].values
    z = sub["Z1_2p09"].values
    x = rng.normal(loc=xpos[i], scale=0.055, size=len(y))

    sc = ax.scatter(
        x, y,
        c=z, cmap=cmap, norm=norm,
        s=40, alpha=0.85,
        edgecolor="black", linewidth=0.35,
        zorder=3,
    )

cbar = fig.colorbar(sc, ax=ax, pad=0.01, shrink=0.9)
cbar.set_label(r"$Z_{1.0,\mathrm{sim}}$ (m)", fontsize=14)
cbar.ax.tick_params(labelsize=12)

ax.axhline(0, color="black", linestyle="--", linewidth=1.3)

ax.set_xticks(xpos)
ax.set_xticklabels(basin_order, rotation=90)

ax.set_ylabel(
    r"Implicit - explicit difference, $\Delta(a+\delta S2S_s)$",
    fontsize=16
)

ax.set_xlabel("Basins in NZVM v2.09", fontsize=20)
ax.tick_params(labelsize=14, direction='in', axis='both', which='both')
ax.grid(True, axis="y", linestyle=":", linewidth=0.8)
ax.grid(True, axis="x", linestyle=":", linewidth=0.5, alpha=0.5)
# Extend upper y-limit to leave room for N labels
y_lo, y_hi = ax.get_ylim()
ax.set_ylim(y_lo, y_hi + 0.15 * (y_hi - y_lo))

# N labels just under the top of the axes
y_lo, y_hi = ax.get_ylim()
y_text = y_hi - 0.04 * (y_hi - y_lo)

# Optional leading "N=" on the first label only
for i, b in enumerate(basin_order):
    label = (f"{counts[b]}" if i == 0 else f"{counts[b]}")
    ax.text(xpos[i], y_text, label,
            ha="center", va="top", fontsize=16, color="black")
fig.savefig(os.path.join(outDir,"Allbasinaverageimprovement_pSA.pdf"))
#%% Box-and-whisker improvement plot for all basins - Paper 2 - EAS
summary_mode = "band"   # "single" for 2 s, or "band" for 1-3 s average
target_T = 2.0          # used only if summary_mode = "single"
T_band = (1/3, 1.0)     # used only if summary_mode = "band"

exclude_basin_types = ["BPVOutcrops"]

def extract_period(col):
    m = re.search(r"(\d+(?:[._p]\d+)?)$", col)
    if m is None:
        return np.nan
    return float(m.group(1).replace("p", ".").replace("_", "."))


meta = stations_sel_sorted.reset_index(drop=True).copy()

EAS_cols = [c for c in siteres_ff_sorted.columns if c.startswith("EAS_")]

siteres_ff_EAS = siteres_ff_sorted[EAS_cols].reset_index(drop=True).copy()
siteres_nb_EAS = siteres_nb_sorted[EAS_cols].reset_index(drop=True).copy()
bias_ff = pd.Series(np.asarray(bias_EAS_ff).ravel(), index=EAS_cols)
bias_nb = pd.Series(np.asarray(bias_EAS_nb).ravel(), index=EAS_cols)

siteres_systematic_ff_EAS = siteres_ff_EAS.add(bias_ff, axis="columns")
siteres_systematic_nb_EAS = siteres_nb_EAS.add(bias_nb, axis="columns")


delta_systematic_EAS = siteres_systematic_nb_EAS - siteres_systematic_ff_EAS


basin_type_str = meta["Basin"].astype(str).str.strip()

station_mask = (
    meta["Basin"].notna()
    & ~basin_type_str.isin(exclude_basin_types)
    & meta["Basin"].notna()
)

meta_sel = meta.loc[station_mask].copy()
delta_sel = delta_systematic_EAS.loc[station_mask].copy()


periods = pd.Series(
    {c: extract_period(c) for c in EAS_cols}
).dropna()

if summary_mode == "single":
    selected_col = (periods - target_T).abs().idxmin()
    delta_one_per_site = delta_sel[selected_col]

    plot_label = rf"SA({periods[selected_col]:.2f} s)"
    print(f"Using nearest period to {target_T:.2f} s: {selected_col}, T = {periods[selected_col]:.3f} s")

elif summary_mode == "band":
    band_cols = periods[
        (periods >= T_band[0]) & (periods <= T_band[1])
    ].index.tolist()

    if len(band_cols) == 0:
        raise ValueError(f"No EAS columns found between {T_band[0]} and {T_band[1]} s.")

    delta_one_per_site = delta_sel[band_cols].mean(axis=1)

    plot_label = rf"mean over {T_band[0]:.0f}-{T_band[1]:.0f} s"
    print(f"Using {len(band_cols)} periods between {T_band[0]} and {T_band[1]} s.")

else:
    raise ValueError("summary_mode must be either 'single' or 'band'.")


plot_df = meta_sel[["Basin", "BasinType_2p09", "Z1_2p09"]].copy()
plot_df["delta_resid"] = delta_one_per_site.values

plot_df = plot_df.dropna(subset=["Basin", "delta_resid"]).copy()

basin_count = plot_df.groupby("Basin")["delta_resid"].size()

basin_order = sorted(
    basin_count.index.tolist(),
    key=lambda b: (-basin_count.loc[b], b.lower())
)

counts = basin_count.loc[basin_order]


summary_table = (
    plot_df.groupby("Basin")["delta_resid"]
    .agg(["count", "median", "mean", "std", "min", "max"])
    .loc[basin_order]
    .round(3)
)

print("\nBasin-wise summary:")
print(summary_table)


data = [
    plot_df.loc[plot_df["Basin"] == b, "delta_resid"].values
    for b in basin_order
]

xpos = np.arange(len(basin_order))

fig, ax = plt.subplots(figsize=(max(12, 0.45 * len(basin_order)), 6),constrained_layout=True)

# Split basins by station count
n_per_basin = {b: len(d) for b, d in zip(basin_order, data)}

box_idx     = [i for i, b in enumerate(basin_order) if n_per_basin[b] >= 3]
line_idx    = [i for i, b in enumerate(basin_order) if n_per_basin[b] == 2]
single_idx  = [i for i, b in enumerate(basin_order) if n_per_basin[b] == 1]

# Boxes only for n >= 3
if box_idx:
    ax.boxplot(
        [data[i] for i in box_idx],
        positions=[xpos[i] for i in box_idx],
        widths=0.55,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color="black", linewidth=2.2),
        boxprops=dict(facecolor="#ececec", edgecolor="black", linewidth=1.2),  # see (3)
        whiskerprops=dict(color="black", linewidth=1.2),
        capprops=dict(color="black", linewidth=1.2),
    )

# Horizontal median line only for n == 2
half_w = 0.55 / 2
for i in line_idx:
    med = np.median(data[i])
    ax.hlines(med, xpos[i] - half_w, xpos[i] + half_w,
              color="black", linewidth=2.2, zorder=2)

# n == 1: nothing to draw here; the scatter loop already plots the point

# Individual site points with jitter
rng = np.random.default_rng(42)
from matplotlib import cm, colors

z_all = plot_df["Z1_2p09"].to_numpy()
norm = colors.LogNorm(vmin=10, vmax=1000)
cmap = cm.jet

for i, b in enumerate(basin_order):
    sub = plot_df.loc[plot_df["Basin"] == b]
    y = sub["delta_resid"].values
    z = sub["Z1_2p09"].values
    x = rng.normal(loc=xpos[i], scale=0.055, size=len(y))

    sc = ax.scatter(
        x, y,
        c=z, cmap=cmap, norm=norm,
        s=40, alpha=0.85,
        edgecolor="black", linewidth=0.35,
        zorder=3,
    )

cbar = fig.colorbar(sc, ax=ax, pad=0.01, shrink=0.9)
cbar.set_label(r"$Z_{1.0,\mathrm{sim}}$ (m)", fontsize=14)
cbar.ax.tick_params(labelsize=12)

ax.axhline(0, color="black", linestyle="--", linewidth=1.3)

ax.set_xticks(xpos)
ax.set_xticklabels(basin_order, rotation=90)

ax.set_ylabel(
    r"Implicit - explicit difference, $\Delta(a+\delta S2S_s)$",
    fontsize=16
)

ax.set_xlabel("Basins in NZVM v2.09", fontsize=20)
ax.tick_params(labelsize=14, direction='in', axis='both', which='both')
ax.grid(True, axis="y", linestyle=":", linewidth=0.8)
ax.grid(True, axis="x", linestyle=":", linewidth=0.5, alpha=0.5)
# Extend upper y-limit to leave room for N labels
y_lo, y_hi = ax.get_ylim()
ax.set_ylim(y_lo, y_hi + 0.15 * (y_hi - y_lo))

# N labels just under the top of the axes
y_lo, y_hi = ax.get_ylim()
y_text = y_hi - 0.03 * (y_hi - y_lo)

# Optional leading "N=" on the first label only
for i, b in enumerate(basin_order):
    label = (f"{counts[b]}" if i == 0 else f"{counts[b]}")
    ax.text(xpos[i], y_text, label,
            ha="center", va="top", fontsize=16, color="black")
fig.savefig(os.path.join(outDir,"Allbasinaverageimprovement_EAS.pdf"))
#%% Basin performance - pSA & all IMs
pSA_cols = [c for c in siteres_ff.columns if c.startswith("pSA_")]
siteres_ff_pSA  = siteres_ff_sorted[pSA_cols]
siteres_nb_pSA  = siteres_nb_sorted[pSA_cols]
sitereserr_ff_pSA  = sitereserr_ff_sorted[pSA_cols]
sitereserr_nb_pSA  = sitereserr_nb_sorted[pSA_cols]

siteres_ff_PGA   = siteres_ff_sorted['PGA'] + bias_IMs_ff[0]
siteres_ff_PGV   = siteres_ff_sorted['PGV'] + bias_IMs_ff[1]
siteres_ff_CAV   = siteres_ff_sorted['CAV'] + bias_IMs_ff[2]
siteres_ff_AI    = siteres_ff_sorted['AI'] + bias_IMs_ff[3]
siteres_ff_D575  = siteres_ff_sorted['Ds575'] + bias_IMs_ff[4]
siteres_ff_D595  = siteres_ff_sorted['Ds595'] + bias_IMs_ff[5]
# siteres_ff_D595_LF  = siteres_ff_sorted['Ds595_LF'] + bias_IMs_ff[6]

siteres_nb_PGA   = siteres_nb_sorted['PGA'] + bias_IMs_nb[0]
siteres_nb_PGV   = siteres_nb_sorted['PGV'] + bias_IMs_nb[1]
siteres_nb_CAV   = siteres_nb_sorted['CAV'] + bias_IMs_nb[2]
siteres_nb_AI    = siteres_nb_sorted['AI'] + bias_IMs_nb[3]
siteres_nb_D575  = siteres_nb_sorted['Ds575'] + bias_IMs_nb[4]
siteres_nb_D595  = siteres_nb_sorted['Ds595'] + bias_IMs_nb[5]
# siteres_nb_D595_LF  = siteres_nb_sorted['Ds595_LF'] + bias_IMs_nb[6]

out_dir = "Basin_performance/pSA"
os.makedirs(out_dir, exist_ok=True)
basin = stations_sel_sorted['Basin']
siteres_systematic_ff_pSA = siteres_ff_pSA + np.array(bias_SA_ff)
siteres_systematic_nb_pSA = siteres_nb_pSA + np.array(bias_SA_nb)
grouped_data_ff_pSA = siteres_systematic_ff_pSA.groupby(basin)
grouped_data_nb_pSA = siteres_systematic_nb_pSA.groupby(basin)
grouped_data_fferr_pSA = sitereserr_ff_pSA.groupby(basin)
grouped_data_nberr_pSA = sitereserr_nb_pSA.groupby(basin)

basin_labels = pd.Index(basin).unique().tolist()
# basin_labels = ["Palmerston North"]
for basin_name in basin_labels:
    # skip if basin not present in both datasets
    if (basin_name not in grouped_data_ff_pSA.groups) or (basin_name not in grouped_data_nb_pSA.groups):
        continue
    z = stations_sel_sorted.loc[stations_sel_sorted["Basin"] == basin_name, "Z1_2p09"]
    z = z[z > 0]
    Z1_ratio = np.exp(np.log(z).mean()) if len(z) else np.nan
    ff_cluster = grouped_data_ff_pSA.get_group(basin_name)
    nb_cluster = grouped_data_nb_pSA.get_group(basin_name)
    fferr_cluster = grouped_data_fferr_pSA.get_group(basin_name)
    nberr_cluster = grouped_data_nberr_pSA.get_group(basin_name)
    
    sys_IMs_ff = np.array([siteres_ff_PGA.loc[ff_cluster.index].mean(),siteres_ff_PGV.loc[ff_cluster.index].mean(),siteres_ff_CAV.loc[ff_cluster.index].mean(),siteres_ff_AI.loc[ff_cluster.index].mean(),siteres_ff_D575.loc[ff_cluster.index].mean(), siteres_ff_D595.loc[ff_cluster.index].mean()])

    sys_IMs_nb = np.array([siteres_nb_PGA.loc[nb_cluster.index].mean(),siteres_nb_PGV.loc[nb_cluster.index].mean(),siteres_nb_CAV.loc[nb_cluster.index].mean(),siteres_nb_AI.loc[nb_cluster.index].mean(),siteres_nb_D575.loc[nb_cluster.index].mean(),siteres_nb_D595.loc[nb_cluster.index].mean()])


    assert len(ff_cluster) == len(siteres_ff_PGA.loc[ff_cluster.index])

    ff_mean = ff_cluster.values.T.mean(axis=1)
    ff_std  = ff_cluster.values.T.std(axis=1)

    nb_mean = nb_cluster.values.T.mean(axis=1)
    nb_std  = nb_cluster.values.T.std(axis=1)

    fig  = plt.figure(figsize=(8.85, 6.07), constrained_layout=True)
    gs = fig.add_gridspec(nrows=1, ncols=2,width_ratios=[3, 1],left=0.05, right=0.95, bottom=0.08, top=0.98,wspace=0.01)
    ax_psa = fig.add_subplot(gs[0, 0])
    ax_im  = fig.add_subplot(gs[0, 1])
    ax_psa.text(0.011, -1.3, 'Overprediction', size=18, fontstyle='italic')
    ax_psa.text(0.011,  1.0, 'Underprediction', size=18, fontstyle='italic')
    ax_psa.semilogx(T, ff_cluster.values.T, linewidth=1, color='gray', alpha=0.3)
    ax_psa.semilogx(T, ff_mean, 'b', linewidth=3, label='With Basins')
    ax_psa.semilogx(T, nb_mean, 'r', linewidth=3, label='Without Basins')
    # ax_psa[0].fill_between(T, nb_mean-nb_std, nb_mean+nb_std, color='r', alpha=0.15)
    
    
    # ax_psa[0].fill_between(T, ff_mean-ff_std, ff_mean+ff_std, color='b', alpha=0.15)



    ax_psa.set_xlim([0.01, 10])
    ax_psa.set_ylim([-1.5, 1.5])
    ax_psa.axhline(0, color='k', linestyle='--')
    ax_psa.axvline(1, color='k', linestyle='--')
    ax_psa.set_ylabel(r'${ a + \delta S2S_s}$', size=18)
    ax_psa.set_xlabel('Vibration Period, T (s)', size=18)
    ax_psa.tick_params(labelsize=14, direction='in', axis='both', which='both')
    ax_psa.grid(color='gray', linestyle='dashed', which='both', linewidth=0.4)

    leg = ax_psa.legend(fontsize=13, loc='lower right')
    leg.get_frame().set_edgecolor('k')

    ax_psa.text(0.015,1.2,rf"$\overline{{Z}}_{{1.0,\mathrm{{sim}}}}$ = {Z1_ratio:.0f} m", fontsize=16)
    ax_psa.text(0.5,1.2,f"{basin_name} (N={len(ff_cluster)})", fontsize=16, fontweight='bold')
    
    x = np.arange(6)
    x_labels = ["PGA", "PGV", "CAV", "AI", r"$D_{s575}$", r"$D_{s595}$"]

    ax_im.scatter(x, sys_IMs_nb, s=50, c='r', marker='o', label='Without Basins')
    ax_im.scatter(x, sys_IMs_ff, s=50, c='b', marker='o', label='With Basins')

    ax_im.axhline(0, color='k', linestyle='--')
    ax_im.set_xlim([-0.5, 5.5])
    ax_im.set_ylim([-1.5, 1.5])

    ax_im.set_xticks(x)
    ax_im.set_xticklabels(x_labels, rotation=90)

    ax_im.tick_params(labelsize=14, direction='in', axis='both', which='both')
    ax_im.grid(color='gray', linestyle='dashed', which='both', linewidth=0.4, alpha=0.5)

    ax_im.yaxis.set_label_position("right")
    ax_im.yaxis.tick_right()

    # safe filename
    safe_name = str(basin_name).strip().replace(" ", "_").replace("/", "-")
    fig.savefig(os.path.join(out_dir, f"{safe_name}_performance.pdf"))
    # fig.savefig(os.path.join(out_dir, f"{safe_name}_performance.png"))
    plt.close(fig)

    
#%% ONE FIGURE for a + dS2S across ALL basins (excluding BPVOutcrops) ---pSA
pSA_cols = [c for c in siteres_ff.columns if c.startswith("pSA_")]
siteres_ff_pSA  = siteres_ff_sorted[pSA_cols]
siteres_nb_pSA  = siteres_nb_sorted[pSA_cols]
sitereserr_ff_pSA  = sitereserr_ff_sorted[pSA_cols]
sitereserr_nb_pSA  = sitereserr_nb_sorted[pSA_cols]
out_dir = "Basin_performance/pSA"
os.makedirs(out_dir, exist_ok=True)
basin = stations_sel_sorted['Basin']
siteres_systematic_ff_pSA = siteres_ff_pSA + np.array(bias_SA_ff)
siteres_systematic_nb_pSA = siteres_nb_pSA + np.array(bias_SA_nb)
grouped_data_ff_pSA = siteres_systematic_ff_pSA.groupby(basin)
grouped_data_nb_pSA = siteres_systematic_nb_pSA.groupby(basin)
grouped_data_fferr_pSA = sitereserr_ff_pSA.groupby(basin)
grouped_data_nberr_pSA = sitereserr_nb_pSA.groupby(basin)
basin_labels = pd.Index(basin).dropna().astype(str).unique().tolist()
basin_labels = [b.strip() for b in basin_labels if b.strip() != "BPVOutcrops"]


basin_labels = [b for b in basin_labels
                if (b in grouped_data_ff_pSA.groups) and (b in grouped_data_nb_pSA.groups)]

basin_labels = sorted(
    basin_labels,
    key=lambda b: (-len(grouped_data_ff_pSA.get_group(b)), b.lower())
)


n = len(basin_labels)
ncols = 4  
nrows = int(np.ceil(n / ncols))

fig, axes = plt.subplots(nrows, ncols, figsize=(4.6*ncols, 3.6*nrows), constrained_layout=True)
axes = np.atleast_1d(axes).flatten()

for i, basin_name in enumerate(basin_labels):
    ax = axes[i]

    ff_cluster = grouped_data_ff_pSA.get_group(basin_name)
    nb_cluster = grouped_data_nb_pSA.get_group(basin_name)


    ff_mean = ff_cluster.values.T.mean(axis=1)
    nb_mean = nb_cluster.values.T.mean(axis=1)


    ax.semilogx(T, ff_cluster.values.T, linewidth=1, color='gray', alpha=0.30)


    ax.semilogx(T, ff_mean, 'b', linewidth=1.5)
    ax.semilogx(T, nb_mean, 'r', linewidth=1.5)


    ax.set_xlim([0.01, 10])
    ax.set_ylim([-1.5, 1.5])
    ax.axhline(0, color='k', linestyle='--')
    ax.axvline(1, color='k', linestyle='--')
    ax.grid(color='gray', linestyle='dashed', which='both', linewidth=0.4)
    ax.tick_params(labelsize=12, direction='in', axis='both', which='both')

    ax.text(0.012, -1.0, f"{basin_name} (N={len(ff_cluster)})",
            fontsize=13, fontweight='bold')

    if i % ncols == 0:
        ax.set_ylabel(r'${ a + \delta S2S_s}$', fontsize=14)
    if i >= (nrows-1) * ncols:
        ax.set_xlabel('Vibration Period, T (s)', fontsize=14)


for j in range(n, len(axes)):
    axes[j].axis('off')

axes[0].semilogx([], [], 'b', linewidth=3, label='With Basins')
axes[0].semilogx([], [], 'r', linewidth=3, label='Without Basins')
leg = axes[0].legend(fontsize=12, loc='upper left')
leg.get_frame().set_edgecolor('k')
fig.savefig(os.path.join(outDir, "AllBasins_aPlusdS2S_excluding_BPVOutcrops.pdf"))
# plt.close(fig)
#%% a + dS2S - Separate basins separate figures - pSA
import re
out_dir = "Basin_performance/pSA/Separate_basins"
os.makedirs(out_dir, exist_ok=True)

def safe_filename(name):
    """
    Make basin names safe for file names.
    """
    return re.sub(r'[\\/*?:"<>|]', "_", str(name).strip())


for basin_name in basin_labels:

    ff_cluster = grouped_data_ff_pSA.get_group(basin_name)
    nb_cluster = grouped_data_nb_pSA.get_group(basin_name)

    ff_mean = ff_cluster.values.T.mean(axis=1)
    nb_mean = nb_cluster.values.T.mean(axis=1)

    fig, ax = plt.subplots(figsize=(6.5, 4.8), constrained_layout=True)

    # Individual station curves
    ax.semilogx(
        T,
        ff_cluster.values.T,
        linewidth=1,
        color="gray",
        alpha=0.30
    )

    # Basin means
    ax.semilogx(T, ff_mean, "b", linewidth=2.0, label="With Basins")
    ax.semilogx(T, nb_mean, "r", linewidth=2.0, label="Without Basins")

    ax.set_xlim([0.01, 10])
    ax.set_ylim([-1.5, 1.5])

    ax.axhline(0, color="k", linestyle="--")
    ax.axvline(1, color="k", linestyle="--")

    ax.grid(
        color="gray",
        linestyle="dashed",
        which="both",
        linewidth=0.4
    )

    ax.tick_params(
        labelsize=12,
        direction="in",
        axis="both",
        which="both"
    )

    ax.set_xlabel("Vibration Period, T (s)", fontsize=14)
    ax.set_ylabel(r"${ a + \delta S2S_s}$", fontsize=14)

    ax.set_title(
        f"{basin_name} (N={len(ff_cluster)})",
        fontsize=14,
        fontweight="bold"
    )

    leg = ax.legend(fontsize=12, loc="upper left")
    leg.get_frame().set_edgecolor("k")

    # Save figure
    fname = safe_filename(basin_name)
    fig.savefig(
        os.path.join(out_dir, f"{fname}_pSA.png"),
        dpi=300,
        bbox_inches="tight"
    )


    plt.close(fig)
#%%Basin performance - a & phi_S2S - EAS
EAS_cols = [c for c in siteres_ff.columns if c.startswith("EAS_")]
siteres_ff_EAS  = siteres_ff_sorted[EAS_cols]
siteres_nb_EAS  = siteres_nb_sorted[EAS_cols]
sitereserr_ff_EAS  = sitereserr_ff_sorted[EAS_cols]
sitereserr_nb_EAS  = sitereserr_nb_sorted[EAS_cols]
out_dir = "Basin_performance/EAS"
os.makedirs(out_dir, exist_ok=True)
basin = stations_sel_sorted['Basin']
siteres_systematic_ff_EAS = siteres_ff_EAS + np.array(bias_EAS_ff)
siteres_systematic_nb_EAS = siteres_nb_EAS + np.array(bias_EAS_nb)
grouped_data_ff_EAS = siteres_systematic_ff_EAS.groupby(basin)
grouped_data_nb_EAS = siteres_systematic_nb_EAS.groupby(basin)
grouped_data_fferr_EAS = sitereserr_ff_EAS.groupby(basin)
grouped_data_nberr_EAS = sitereserr_nb_EAS.groupby(basin)


basin_labels = pd.Index(basin).unique().tolist()
# basin_labels = ["Hanmer"]
for basin_name in basin_labels:
    # skip if basin not present in both datasets
    if (basin_name not in grouped_data_ff_EAS.groups) or (basin_name not in grouped_data_nb_EAS.groups):
        continue

    ff_cluster = grouped_data_ff_EAS.get_group(basin_name)
    nb_cluster = grouped_data_nb_EAS.get_group(basin_name)
    fferr_cluster = grouped_data_fferr_EAS.get_group(basin_name)
    nberr_cluster = grouped_data_nberr_EAS.get_group(basin_name)


    # stats per period
    ff_mean = ff_cluster.values.T.mean(axis=1)
    ff_std  = ff_cluster.values.T.std(axis=1)

    nb_mean = nb_cluster.values.T.mean(axis=1)
    nb_std  = nb_cluster.values.T.std(axis=1)

    fig,axes  = plt.subplots(figsize=(6.63, 5.08), constrained_layout=True)

    ff_phiS2S =  np.sqrt(np.mean((ff_cluster-np.mean(ff_cluster, axis=0))**2,axis=0)+np.mean((fferr_cluster)**2,axis=0))
    nb_phiS2S =  np.sqrt(np.mean((nb_cluster-np.mean(nb_cluster, axis=0))**2,axis=0)+np.mean((nberr_cluster)**2,axis=0))
    axes.text(0.11, -1.3, 'Overprediction', size=18, fontstyle='italic')
    axes.text(0.11,  1.0, 'Underprediction', size=18, fontstyle='italic')
    axes.semilogx(f, ff_cluster.values.T, linewidth=1, color='gray', alpha=0.3)
    axes.semilogx(f, nb_mean, 'r', linewidth=3, label='Without Basins')
    # axes.fill_between(T, nb_mean-nb_std, nb_mean+nb_std, color='r', alpha=0.15)
    
    axes.semilogx(f, ff_mean, 'b', linewidth=3, label='With Basins')
    # axes.fill_between(T, ff_mean-ff_std, ff_mean+ff_std, color='b', alpha=0.15)



    axes.set_xlim([0.1, 20])
    axes.set_ylim([-1.5, 1.5])
    axes.axhline(0, color='k', linestyle='--')
    axes.axvline(1, color='k', linestyle='--')
    axes.set_ylabel(r'${ a + \delta S2S_s}$', size=18)
    axes.set_xlabel('Frequency, f (Hz)', size=18)
    axes.tick_params(labelsize=14, direction='in', axis='both', which='both')
    axes.grid(color='gray', linestyle='dashed', which='both', linewidth=0.4)

    leg = axes.legend(fontsize=13, loc='lower right')
    leg.get_frame().set_edgecolor('k')

    axes.text(1.1,1.2,f"{basin_name} (N={len(ff_cluster)})", fontsize=16, fontweight='bold')

    # safe filename
    safe_name = str(basin_name).strip().replace(" ", "_").replace("/", "-")
    fig.savefig(os.path.join(out_dir, f"{safe_name}_performance_EAS.pdf"))
    # fig.savefig(os.path.join(out_dir, f"{safe_name}_performance.png"))
    plt.close(fig)
#%%Basin performance - a & phi_S2S - EAS - Only phiS2S
from matplotlib.lines import Line2D
colors = ['r','g','b','maroon']
basin_labels = ["PalmerstonNorth","Marlborough","Nelson","Canterbury"]
basin_handles = [Line2D([0], [0], color=colors[i], lw=2, label=basin_labels[i])for i in range(len(basin_labels))]
style_handles = [Line2D([0], [0], color='k', lw=2, linestyle='-',  label='Without basins'),Line2D([0], [0], color='k', lw=2, linestyle='--', label='With basins')]

EAS_cols = [c for c in siteres_ff.columns if c.startswith("EAS_")]
siteres_ff_EAS  = siteres_ff_sorted[EAS_cols]
siteres_nb_EAS  = siteres_nb_sorted[EAS_cols]
sitereserr_ff_EAS  = sitereserr_ff_sorted[EAS_cols]
sitereserr_nb_EAS  = sitereserr_nb_sorted[EAS_cols]
out_dir = "Basin_performance/EAS"
os.makedirs(out_dir, exist_ok=True)
basin = stations_sel_sorted['Basin']
siteres_systematic_ff_EAS = siteres_ff_EAS + np.array(bias_EAS_ff)
siteres_systematic_nb_EAS = siteres_nb_EAS + np.array(bias_EAS_nb)
grouped_data_ff_EAS = siteres_systematic_ff_EAS.groupby(basin)
grouped_data_nb_EAS = siteres_systematic_nb_EAS.groupby(basin)
grouped_data_fferr_EAS = sitereserr_ff_EAS.groupby(basin)
grouped_data_nberr_EAS = sitereserr_nb_EAS.groupby(basin)

# basin_labels = pd.Index(basin).unique().tolist()
# basin_labels = ["PalmerstonNorth","Marlborough","Nelson","Canterbury"]

i =0 
fig, axes = plt.subplots(1, 1, figsize=(7.06, 5.36), constrained_layout=True)
for basin_name in basin_labels:
    # skip if basin not present in both datasets
    if (basin_name not in grouped_data_ff_EAS.groups) or (basin_name not in grouped_data_nb_EAS.groups):
        continue

    ff_cluster = grouped_data_ff_EAS.get_group(basin_name)
    nb_cluster = grouped_data_nb_EAS.get_group(basin_name)
    fferr_cluster = grouped_data_fferr_EAS.get_group(basin_name)
    nberr_cluster = grouped_data_nberr_EAS.get_group(basin_name)


    

    ff_phiS2S =  np.sqrt(np.mean((ff_cluster-np.mean(ff_cluster, axis=0))**2,axis=0)+np.mean((fferr_cluster)**2,axis=0))
    nb_phiS2S =  np.sqrt(np.mean((nb_cluster-np.mean(nb_cluster, axis=0))**2,axis=0)+np.mean((nberr_cluster)**2,axis=0))
    
    axes.semilogx(f, nb_phiS2S, linewidth=2, color = colors[i],linestyle = '-')
    axes.semilogx(f, ff_phiS2S, linewidth=2, color = colors[i],linestyle = '--')

    i = i+1
    
axes.set_xlim([0.1, 20])
axes.set_ylim([0, 1])
# Basin legend (colors)
leg1 = axes.legend(
    handles=basin_handles,
    loc='best',
    fontsize=16,
    # title='Basins'
)
axes.add_artist(leg1) 

# Line-style legend
axes.legend(
    handles=style_handles,
    loc='upper left',
    fontsize=16,
)

axes.axvline(1, color='k', linestyle='--')
axes.set_xlabel('Frequency, f (Hz)', size=18)
axes.tick_params(labelsize=14, direction='in', axis='both', which='both')
axes.grid(color='gray', linestyle='dashed', which='both', linewidth=0.4)
axes.set_ylabel(r'$\phi_{S2S}$', fontsize=18, fontweight='bold')
fig.savefig(os.path.join(out_dir, "phiS2S_all_EAS.pdf"),dpi=600)

#%% ONE FIGURE for a + dS2S across ALL basins (excluding BPVOutcrops) ---EAS
EAS_cols = [c for c in siteres_ff.columns if c.startswith("EAS_")]
siteres_ff_EAS  = siteres_ff_sorted[EAS_cols]
siteres_nb_EAS  = siteres_nb_sorted[EAS_cols]
sitereserr_ff_EAS  = sitereserr_ff_sorted[EAS_cols]
sitereserr_nb_EAS  = sitereserr_nb_sorted[EAS_cols]
out_dir = "Basin_performance/EAS"
os.makedirs(out_dir, exist_ok=True)
basin = stations_sel_sorted['Basin']
siteres_systematic_ff_EAS = siteres_ff_EAS + np.array(bias_EAS_ff)
siteres_systematic_nb_EAS = siteres_nb_EAS + np.array(bias_EAS_nb)
grouped_data_ff_EAS = siteres_systematic_ff_EAS.groupby(basin)
grouped_data_nb_EAS = siteres_systematic_nb_EAS.groupby(basin)
grouped_data_fferr_EAS = sitereserr_ff_EAS.groupby(basin)
grouped_data_nberr_EAS = sitereserr_nb_EAS.groupby(basin)
basin_labels = pd.Index(basin).dropna().astype(str).unique().tolist()
basin_labels = [b.strip() for b in basin_labels if b.strip() != "BPVOutcrops"]


basin_labels = [b for b in basin_labels
                if (b in grouped_data_ff_EAS.groups) and (b in grouped_data_nb_EAS.groups)]

basin_labels = sorted(basin_labels,key=lambda b: len(grouped_data_ff_EAS.get_group(b)),reverse=True)


n = len(basin_labels)
ncols = 4  
nrows = int(np.ceil(n / ncols))

fig, axes = plt.subplots(nrows, ncols, figsize=(4.6*ncols, 3.6*nrows), constrained_layout=True)
axes = np.atleast_1d(axes).flatten()

for i, basin_name in enumerate(basin_labels):
    ax = axes[i]

    ff_cluster = grouped_data_ff_EAS.get_group(basin_name)
    nb_cluster = grouped_data_nb_EAS.get_group(basin_name)


    ff_mean = ff_cluster.values.T.mean(axis=1)
    nb_mean = nb_cluster.values.T.mean(axis=1)


    ax.semilogx(f, ff_cluster.values.T, linewidth=1, color='gray', alpha=0.30)


    ax.semilogx(f, ff_mean, 'b', linewidth=1.5)
    ax.semilogx(f, nb_mean, 'r', linewidth=1.5)


    ax.set_xlim([0.1, 20])
    ax.set_ylim([-2.0, 2.0])
    ax.axhline(0, color='k', linestyle='--')
    ax.axvline(1, color='k', linestyle='--')
    ax.grid(color='gray', linestyle='dashed', which='both', linewidth=0.4)
    ax.tick_params(labelsize=12, direction='in', axis='both', which='both')

    ax.text(0.12, -1.0, f"{basin_name} (N={len(ff_cluster)})",
            fontsize=13, fontweight='bold')

    if i % ncols == 0:
        ax.set_ylabel(r'${ a + \delta S2S_s}$', fontsize=14)
    if i >= (nrows-1) * ncols:
        ax.set_xlabel('Frequency, f (Hz)', fontsize=14)


for j in range(n, len(axes)):
    axes[j].axis('off')

axes[0].semilogx([], [], 'b', linewidth=3, label='With Basins')
axes[0].semilogx([], [], 'r', linewidth=3, label='Without Basins')
leg = axes[0].legend(fontsize=12, loc='upper right')
leg.get_frame().set_edgecolor('k')
fig.savefig(os.path.join(out_dir, "AllBasins_aPlusdS2S_EAS.png"), dpi=300)
plt.close(fig)
#%% a + dS2S - Separate basins separate figures - EAS
import re
out_dir = "Basin_performance/EAS/Separate_basins"
os.makedirs(out_dir, exist_ok=True)

def safe_filename(name):
    """
    Make basin names safe for file names.
    """
    return re.sub(r'[\\/*?:"<>|]', "_", str(name).strip())


for basin_name in basin_labels:

    ff_cluster = grouped_data_ff_EAS.get_group(basin_name)
    nb_cluster = grouped_data_nb_EAS.get_group(basin_name)

    ff_mean = ff_cluster.values.T.mean(axis=1)
    nb_mean = nb_cluster.values.T.mean(axis=1)

    fig, ax = plt.subplots(figsize=(6.5, 4.8), constrained_layout=True)

    # Individual station curves
    ax.semilogx(
        f,
        ff_cluster.values.T,
        linewidth=1,
        color="gray",
        alpha=0.30
    )

    # Basin means
    ax.semilogx(f, ff_mean, "b", linewidth=2.0, label="With Basins")
    ax.semilogx(f, nb_mean, "r", linewidth=2.0, label="Without Basins")

    ax.set_xlim([0.1, 10])
    ax.set_ylim([-1.5, 1.5])

    ax.axhline(0, color="k", linestyle="--")
    ax.axvline(1, color="k", linestyle="--")

    ax.grid(
        color="gray",
        linestyle="dashed",
        which="both",
        linewidth=0.4
    )

    ax.tick_params(
        labelsize=12,
        direction="in",
        axis="both",
        which="both"
    )

    ax.set_xlabel("Vibration Period, T (s)", fontsize=14)
    ax.set_ylabel(r"${ a + \delta S2S_s}$", fontsize=14)

    ax.set_title(
        f"{basin_name} (N={len(ff_cluster)})",
        fontsize=14,
        fontweight="bold"
    )

    leg = ax.legend(fontsize=12, loc="upper left")
    leg.get_frame().set_edgecolor("k")

    # Save figure
    fname = safe_filename(basin_name)
    fig.savefig(
        os.path.join(out_dir, f"{fname}_EAS.png"),
        dpi=300,
        bbox_inches="tight"
    )


    plt.close(fig)
#%% Basin performance - Z1 bins (a + dS2S, NOT delta)

from matplotlib.lines import Line2D

# ----------------------------------------------------------------------
# BIN DEFINITIONS
# ----------------------------------------------------------------------
bins = [-np.inf, 100, 300, 500, np.inf]
labels = [
    r"$Shallow\ (Z_1 < 100\ m)$",
    r"$Moderate\ (100–300\ m)$",
    r"$Deep\ (300–500\ m)$",
    r"$Very\ Deep\ (>500\ m)$"
]

colors = ['r','b','g','k']

# ----------------------------------------------------------------------
# PREP DATA (same as before)
# ----------------------------------------------------------------------
pSA_cols = [c for c in siteres_ff.columns if c.startswith("pSA_")]

siteres_ff_pSA  = siteres_ff_sorted[pSA_cols]
siteres_nb_pSA  = siteres_nb_sorted[pSA_cols]

out_dir = "Basin_performance/pSA"
os.makedirs(out_dir, exist_ok=True)

# systematic residuals (a + dS2S)
siteres_systematic_ff_pSA = siteres_ff_pSA + np.array(bias_SA_ff)
siteres_systematic_nb_pSA = siteres_nb_pSA + np.array(bias_SA_nb)

# ----------------------------------------------------------------------
# BUILD STATION DF (keep your cleaning)
# ----------------------------------------------------------------------
df_station = pd.DataFrame(index=siteres_systematic_ff_pSA.index)

df_station["Z1"] = stations_sel_sorted["Z1_2p09"].values
df_station["Basin"] = stations_sel_sorted["Basin"].astype(str).str.strip()

mask = np.isfinite(df_station["Z1"])
mask &= df_station["Basin"].ne("BPVOutcrops")
mask &= df_station["Basin"].ne("")
mask &= ~df_station["Basin"].str.lower().eq("nan")

df_station = df_station[mask]

# align
siteres_systematic_ff_pSA = siteres_systematic_ff_pSA.loc[df_station.index]
siteres_systematic_nb_pSA = siteres_systematic_nb_pSA.loc[df_station.index]

# ----------------------------------------------------------------------
# BINNING
# ----------------------------------------------------------------------
df_station["Z1_bin"] = pd.cut(df_station["Z1"], bins=bins, labels=labels)

# ----------------------------------------------------------------------
# PLOTTING
# ----------------------------------------------------------------------
fig, ax = plt.subplots(1, 1, figsize=(7.06, 5.36), constrained_layout=True)

i = 0
for lab in labels:

    idx = (df_station["Z1_bin"] == lab).values

    if idx.sum() == 0:
        continue

    ff_cluster = siteres_systematic_ff_pSA.values[idx, :]
    nb_cluster = siteres_systematic_nb_pSA.values[idx, :]

    # --- CORE: MEAN (a + dS2S) ---
    ff_mean = np.nanmean(ff_cluster, axis=0)
    nb_mean = np.nanmean(nb_cluster, axis=0)

    # --- PLOT ---
    ax.plot(T, nb_mean,
                color=colors[i],
                linestyle='-',
                linewidth=2)

    ax.plot(T, ff_mean,
                color=colors[i],
                linestyle='--',
                linewidth=2,
                label=f"{lab} (N={idx.sum()})")

    i += 1

# ----------------------------------------------------------------------
# FORMATTING
# ----------------------------------------------------------------------
ax.set_xlim([1, 10])
# ax.set_ylim([-1.5, 1.5])

ax.axhline(0, color='k', linestyle='--')
ax.axvline(1, color='k', linestyle='--')

ax.set_xlabel('Vibration Period, T (s)', size=18)
ax.set_ylabel(r'${ a + \delta S2S_s}$', fontsize=18)

ax.tick_params(labelsize=14, direction='in', axis='both', which='both')
# ax.grid(color='gray', linestyle='dashed', linewidth=0.4)

# ----------------------------------------------------------------------
# LEGENDS
# ----------------------------------------------------------------------
# Color legend (bins)
leg1 = ax.legend(loc='lower left', fontsize=13)
ax.add_artist(leg1)

# Style legend
style_handles = [
    Line2D([0], [0], color='k', lw=2, linestyle='-',  label='Without basins'),
    Line2D([0], [0], color='k', lw=2, linestyle='--', label='With basins')
]

ax.legend(handles=style_handles, loc='upper left', fontsize=13)


fig.savefig(os.path.join(out_dir, "a_dS2S_Z1bins_pSA.pdf"), dpi=600)
#%% a + dS2S difference against Z1 - pSA + IMs - Paper 2

IM_cols = ["PGA","PGV","CAV","AI","Ds575","Ds595"]


pSA_cols = [c for c in siteres_ff.columns if c.startswith("pSA_")]

siteres_ff_pSA = siteres_ff_sorted[pSA_cols]
siteres_nb_pSA = siteres_nb_sorted[pSA_cols]

siteres_ff_IMs = siteres_ff_sorted[IM_cols]
siteres_nb_IMs = siteres_nb_sorted[IM_cols]


common_index = siteres_ff_pSA.index.intersection(siteres_nb_pSA.index)
common_index = common_index.intersection(stations_sel_sorted.index)

siteres_ff_pSA = siteres_ff_pSA.loc[common_index]
siteres_nb_pSA = siteres_nb_pSA.loc[common_index]

siteres_ff_IMs = siteres_ff_IMs.loc[common_index]
siteres_nb_IMs = siteres_nb_IMs.loc[common_index]

stations_sel_sorted = stations_sel_sorted.loc[common_index]


bias_SA_ff_arr = np.array(bias_SA_ff)[None, :]
bias_SA_nb_arr = np.array(bias_SA_nb)[None, :]

bias_IMs_ff_arr = np.array(bias_IMs_ff)[None, :]
bias_IMs_nb_arr = np.array(bias_IMs_nb)[None, :]

siteres_systematic_ff_pSA = siteres_ff_pSA + bias_SA_ff_arr
siteres_systematic_nb_pSA = siteres_nb_pSA + bias_SA_nb_arr

siteres_systematic_ff_IMs = siteres_ff_IMs + bias_IMs_ff_arr
siteres_systematic_nb_IMs = siteres_nb_IMs + bias_IMs_nb_arr


delta_abs_pSA = siteres_systematic_nb_pSA - siteres_systematic_ff_pSA
delta_abs_IM  = siteres_systematic_nb_IMs - siteres_systematic_ff_IMs


df_station = pd.DataFrame(index=common_index)

df_station["Basin"] = stations_sel_sorted["Basin"].astype(str).str.strip()
df_station["Type"]  = stations_sel_sorted["BasinType_2p09"].astype(str).str.strip()
df_station["Geomorphology"] = stations_sel_sorted["Geomorphology"].astype(str).str.strip()

df_station["Z1"] = stations_sel_sorted["Z1_2p09"] 

# --- Filtering ---
# mask = np.isfinite(df_station["Z1"])
# mask &= df_station["Basin"].ne("BPVOutcrops")
# mask &= df_station["Basin"].ne("")
# mask &= ~df_station["Basin"].str.lower().eq("nan")
mask = ~stations_sel_sorted["BasinType_2p09"].isin(["Non-Basin", "Unmodeled"])
df_station = df_station[mask]
delta_abs_pSA = delta_abs_pSA.loc[df_station.index]
delta_abs_IM  = delta_abs_IM.loc[df_station.index]

print("Stations kept:", len(df_station))

# --- Z1 binning ---
bins = [-np.inf, 100, 300, 500, np.inf]

labels = [
    r"Shallow ($Z_{1.0,\mathrm{sim}} < 100$ m)",
    r"Moderate ($100 \leq Z_{1.0,\mathrm{sim}} < 300$ m)",
    r"Deep ($300 \leq Z_{1.0,\mathrm{sim}} < 500$ m)",
    r"Very deep ($Z_{1.0,\mathrm{sim}} \geq 500$ m)"
]
df_station["Z1_bin"] = pd.cut(df_station["Z1"], bins=bins, labels=labels)

# --- Compute medians ---
median_pSA = {}
median_IM  = {}
N_by_bin   = {}
iqr_low_pSA = {}
iqr_high_pSA = {}
for lab in labels:
    idx = (df_station["Z1_bin"] == lab).values
    N_by_bin[lab] = idx.sum()

    if N_by_bin[lab] == 0:
        continue

    data = delta_abs_pSA.values[idx, :]

    q25 = np.nanpercentile(data, 25, axis=0)
    q50 = np.nanpercentile(data, 50, axis=0)
    q75 = np.nanpercentile(data, 75, axis=0)

    median_pSA[lab] = np.nanmean(data, axis=0)
    iqr_low_pSA[lab] = q25
    iqr_high_pSA[lab] = q75
    median_IM[lab]  = np.nanmean(delta_abs_IM.values[idx, :], axis=0)

print("N per bin:", N_by_bin)

# --- Plot ---
fig = plt.figure(figsize=(9.62,6.46), constrained_layout=True)

gs = fig.add_gridspec(
    nrows=1, ncols=2,
    width_ratios=[3,1],
    wspace=0.05, hspace = 0.2
)

ax_psa = fig.add_subplot(gs[0,0])
ax_im  = fig.add_subplot(gs[0,1])

colors = ['r','b','g','k','orange','maroon']

# --- pSA curves ---
for i, lab in enumerate(labels):
    if lab in median_pSA:
        ax_psa.semilogx(
            T,
            median_pSA[lab],
            linewidth=2,
            color=colors[i],
            label=f"{lab} ($N={N_by_bin[lab]}$)"
        )
        # ax_psa.fill_between(T,iqr_low_pSA[lab],iqr_high_pSA[lab],color=colors[i],alpha=0.2)
ax_psa.axhline(0, color="k", linestyle="--")
ax_psa.axvline(1, color="k", linestyle="--")

ax_psa.set_xlim(min(T), max(T))
ax_psa.set_xlabel("Vibration Period, T (s)", size=22)
ax_psa.set_ylabel(
    r"Implicit - explicit difference, $\Delta(a+\delta S2S_s)$",
    fontsize=20
)

ax_psa.tick_params(labelsize=18, direction='in', axis='both', which='both')
# ax_psa.grid(True, linestyle="--", linewidth=0.4)
ax_psa.set_ylim([-0.2, 0.8])
ax_psa.legend(fontsize=13, loc = 'upper left')

# --- IM panel ---
x = np.arange(len(IM_cols))
x_labels = ["PGA","PGV","CAV","AI",r"$D_{s575}$",r"$D_{s595}$"]

for i, lab in enumerate(labels):
    if lab in median_IM:
        ax_im.scatter(
            x,
            median_IM[lab],
            s=30,
            color=colors[i]
        )

ax_im.axhline(0, color="k", linestyle="--")

ax_im.set_xlim([-0.5, len(IM_cols)-0.5])
ax_im.set_ylim([-0.2, 0.2])

ax_im.set_xticks(x)
ax_im.set_xticklabels(x_labels, rotation=90)

ax_im.tick_params(labelsize=18, direction='in')
ax_im.grid(True, linestyle="--", linewidth=0.4)

ax_im.yaxis.tick_right()

# --- Save ---
out_dir = "Basin_performance/pSA"
os.makedirs(out_dir, exist_ok=True)
# fig.text(0.008, 0.95, '(a)', fontsize=26, fontweight='bold') 

plt.savefig(os.path.join(outDir, "Median_dAbs_vs_T_binned_by_Z1_with_IMs.pdf"))
#%% a + dS2S difference against Z_2p09 - EAS - Paper 2
EAS_cols = [c for c in siteres_ff.columns if c.startswith("EAS_")]
siteres_ff_EAS  = siteres_ff_sorted[EAS_cols]
siteres_nb_EAS  = siteres_nb_sorted[EAS_cols]
sitereserr_ff_EAS  = sitereserr_ff_sorted[EAS_cols]
sitereserr_nb_EAS  = sitereserr_nb_sorted[EAS_cols]
out_dir = "Basin_performance/EAS"
os.makedirs(out_dir, exist_ok=True)
basin = stations_sel_sorted['Basin']
siteres_systematic_ff_EAS = siteres_ff_EAS + np.array(bias_EAS_ff)
siteres_systematic_nb_EAS = siteres_nb_EAS + np.array(bias_EAS_nb)
abs_ff = siteres_systematic_ff_EAS
abs_nb = siteres_systematic_nb_EAS
delta_abs = abs_nb - abs_ff 
df_station = pd.DataFrame(index=delta_abs.index)
df_station["Basin"] = stations_sel_sorted["Basin"].astype(str).str.strip().values
df_station["Z1_with"] = stations_sel_sorted["Z1_2p09"].values  
df_station["Z1_without"] = stations_sel_sorted["Z1_nb"].values  


# df_station["dZ1"] = df_station["Z1_with"] - df_station["Z1_without"]
df_station["dZ1"] = df_station["Z1_with"]

basin_clean = df_station["Basin"]
mask &= basin_clean.ne("BPVOutcrops")
mask &= basin_clean.ne("")               
mask &= ~basin_clean.str.lower().eq("nan") 
df_station = df_station[mask]
delta_abs = delta_abs.loc[df_station.index]

print("Stations kept:", len(df_station), "of", len(stations_sel_sorted))
print("Periods:", delta_abs.shape[1])


bins = [-np.inf, 100, 300, 500, np.inf]

labels = [
    r"Shallow ($Z_{1.0,\mathrm{sim}} < 100$ m)",
    r"Moderate ($100 \leq Z_{1.0,\mathrm{sim}} < 300$ m)",
    r"Deep ($300 \leq Z_{1.0,\mathrm{sim}} < 500$ m)",
    r"Very deep ($Z_{1.0,\mathrm{sim}} \geq 500$ m)"
]
df_station = df_station.copy()
df_station["dZ1_bin"] = pd.cut(df_station["dZ1"], bins=bins, labels=labels)


delta_abs_df = delta_abs.loc[df_station.index]


median_by_bin = {}
fracpos_by_bin = {}   
  
N_by_bin = {}

for lab in labels:
    idx = (df_station["dZ1_bin"] == lab).values
    N_by_bin[lab] = idx.sum()
    if N_by_bin[lab] == 0:
        continue

    arr = delta_abs_df.values[idx, :]  
    eps = 1e-2
    fracpos_by_bin[lab] = np.mean(arr > eps,axis=0)
    median_by_bin[lab] = np.nanmean(arr, axis=0)


print("N per bin:", N_by_bin)

fig, ax = plt.subplots(figsize=(7.54,5.68), constrained_layout=True)
colors = ['r','b','g','k','orange','maroon']
i = 0
for lab in labels:
    if lab in median_by_bin:
        ax.semilogx(f, median_by_bin[lab], linewidth=2, color = colors[i],label=f"{lab} ($N={N_by_bin[lab]}$)")
        i =i+1

ax.axhline(0, color="k", linestyle="--", linewidth=1)
ax.axvline(1, color="k", linestyle="--", linewidth=1)
ax.set_xlim([0.1,20])
ax.set_ylim([-0.2,1.0])
ax.set_xlabel("Frequency, f (Hz)",size=20)
ax.set_ylabel(
    r"Implicit - explicit difference, $\Delta(a+\delta S2S_s)$",
    fontsize=20
)
# ax.grid(True, which="both", linestyle="--", linewidth=0.4)
ax.tick_params(labelsize=16,direction='in', axis='both', which='both')
# fig.text(0.008, 0.95, '(b)', fontsize=20, fontweight='bold') 
ax.legend(fontsize=11, loc = 'best')
plt.savefig(os.path.join(outDir, "Median_dAbs_vs_T_binned_by_Z1_EAS.pdf"))
# plt.close(fig)
#%% TESTS
T_sel = 'pSA_2.000000000000'

df_station["Z1_with"] = stations_sel_sorted["Z1_2p09"]
df_station["Z1_without"] = stations_sel_sorted["Z1_nb"]

# safer definition (avoid log of negative)
df_station["dZ1"] = df_station["Z1_with"] - df_station["Z1_without"]

x = df_station["dZ1"]
y = delta_abs_df[T_sel]

fig, ax = plt.subplots(figsize=(7.95, 5.66), constrained_layout=True)

sc = ax.scatter(
    x,
    y,
    s=50,
    edgecolor="k"
)

ax.axhline(0, color="k", linestyle="--", linewidth=1)

ax.tick_params(labelsize=16, direction="in", axis="both", which="both")

ax.set_xlim(x.min(), x.max())
ax.set_ylim(-abs(y).max(), abs(y).max())

ax.set_xlabel(r"$\Delta Z_{1.0}$ (with - without)", size=20)
ax.set_ylabel(r"$|r_{no\ basins}|-|r_{with\ basins}|$", size=20)

T_val = float(T_sel.split("_")[1])
ax.text(0.02, 0.95, rf"$T = {T_val:g}\,\mathrm{{s}}$",
        transform=ax.transAxes, fontsize=18, fontweight='bold',
        verticalalignment="top", horizontalalignment="left")
#%% Scatter plot version of the above 
T_sel = 'pSA_2.000000000000'
df_station["Z1_with"] = stations_sel_sorted["Z1_2p09"] 
df_station["Z1_without"] = stations_sel_sorted["Z1_nb"] 
df_station["dZ1_bin"] = np.log(df_station["Z1_with"] - df_station["Z1_without"])
fig, ax = plt.subplots(figsize=(7.95, 5.66), constrained_layout=True)
out_dir = "delta_residual_vs_Z1_Tgt1s_Z1bin"
x = df_station["dZ1_bin"]
y = delta_abs_df[T_sel]
bins = df_station["Z1_bin"]
T_val = float(T_sel.split("_")[1])

bin_order = [
    "$Shallow\ (Z_1 < 100\ m)$",
    "$Moderate\ (Z_1: 100–300\ m)$",
    "$Deep\ (Z_1:300-500\ m)$",
    "$Very\ Deep\ (Z_1>500\ m)$"
]



color_map = {
    "$Shallow\ (Z_1 < 100\ m)$": "red",
    "$Moderate\ (Z_1: 100–300\ m)$": "blue",
    "$Deep\ (Z_1:300-500\ m)$": "green",
    "$Very\ Deep\ (Z_1>500\ m)$": "black",
}
scatter_plots = []
for b in bin_order:
    idx = bins == b
    sc = ax.scatter(
        x[idx],
        y[idx],
        s=50,
        color=color_map[b],
        label=b,
        edgecolor="k",
        picker=True
    )
    sc.df_indices = df_station.index[idx].to_numpy()
    scatter_plots.append(sc)
ax.axhline(0, color="k", linestyle="--", linewidth=1)

ax.tick_params(labelsize=16, direction="in", axis="both", which="both")
ax.set_xlim(min(x), max(x))
ax.set_ylim(-max(y), max(y))

ax.set_xlabel(r"Depth to the 1.0 km/s $V_S$ horizon ($Z_{1.0}$)", size=20)
ax.set_ylabel(r"$|r_{no\ basins}|-|r_{with\ basins}|$", size=20)
ax.text(0.02, 0.95, rf"$T = {T_val:g}\,\mathrm{{s}}$",transform=ax.transAxes, fontsize=18,fontweight='bold',verticalalignment="top",horizontalalignment="left")
ax.legend(fontsize=14, frameon=True)
def onpick(event):
    sc = event.artist
    ind = event.ind[0]  # index within this scatter

    df_ind = sc.df_indices[ind]
    basin = df_station.loc[df_ind, "Basin"]

    print(f"Clicked basin: {basin}")

    # Remove old annotations (optional but recommended)
    for txt in ax.texts:
        txt.remove()

    ax.annotate(
        basin,
        (sc.get_offsets()[ind, 0], sc.get_offsets()[ind, 1]),
        xytext=(10, 10),
        textcoords="offset points",
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.25", fc="yellow", alpha=0.7)
    )

    fig.canvas.draw_idle()


fig.canvas.mpl_connect("pick_event", onpick)
fig.savefig(os.path.join(out_dir, "Paper_2_Z1bin"), dpi=300)
# #%% Loop for all T's for above - Z1 bin
# # import os
# pSA_cols = [c for c in delta_abs_df.columns if c.startswith("pSA_")]
# T_vals = {c: float(c.split("_")[1]) for c in pSA_cols}
# pSA_cols_sel = [c for c, T in T_vals.items() if T >= 1.0]
# pSA_cols_sel = sorted(pSA_cols_sel, key=lambda c: T_vals[c])

# out_dir = "delta_residual_vs_Z1_Tgt1s_Z1bin"
# os.makedirs(out_dir, exist_ok=True)
# def plot_delta_vs_Z1(T_sel, df_station, delta_abs_df, out_dir):
#     T_val = float(T_sel.split("_")[1])

#     fig, ax = plt.subplots(figsize=(7.95, 5.66), constrained_layout=True)

#     x = df_station["dZ1"]
#     y = delta_abs_df[T_sel]
#     bins = df_station["dZ1_bin"]

#     bin_order = [
#         "$Shallow\\ (Z_1 < 100\\ m)$",
#         "$Moderate\\ (Z_1: 100–300\\ m)$",
#         "$Deep\\ (Z_1:300-500\\ m)$",
#         "$Very\\ Deep\\ (Z_1>500\\ m)$"
#     ]

#     color_map = {
#         bin_order[0]: "red",
#         bin_order[1]: "blue",
#         bin_order[2]: "green",
#         bin_order[3]: "black",
#     }

#     for b in bin_order:
#         idx = bins == b
#         ax.scatter(
#             x[idx],
#             y[idx],
#             s=50,
#             color=color_map[b],
#             label=b,
#             edgecolor="k"
#         )

#     ax.axhline(0, color="k", linestyle="--", linewidth=1)

#     ax.tick_params(labelsize=16, direction="in", axis="both", which="both")
#     ax.set_xlim(min(x), max(x))
#     ax.set_ylim(-max(y), max(y))

#     ax.set_xlabel(r"Depth to the 1.0 km/s $V_S$ horizon ($Z_{1.0}$)", size=20)
#     ax.set_ylabel(r"Median $|r_{no\ basins}|-|r_{with\ basins}|$", size=20)

#     ax.text(
#         0.02, 0.95,
#         rf"$T = {T_val:g}\,\mathrm{{s}}$",
#         transform=ax.transAxes,
#         fontsize=18,
#         fontweight="bold",
#         va="top", ha="left"
#     )

#     ax.legend(fontsize=14, frameon=True)

#     # Save
#     fname = f"delta_vs_Z1_T_{T_val:g}s.png"
#     fig.savefig(os.path.join(out_dir, fname), dpi=300)
#     plt.close(fig)
    
# for T_sel in pSA_cols_sel:
#     plot_delta_vs_Z1(T_sel, df_station, delta_abs_df, out_dir)


# #%% Scatter plot version of the above using Type categories
# T_sel = 'pSA_2.000000000000'

# fig, ax = plt.subplots(figsize=(7.95, 5.66), constrained_layout=True)
# out_dir = "delta_residual_vs_Z1_Tgt1s_Type"
# x = df_station["dZ1"]
# y = delta_abs_df[T_sel]
# bins = df_station["Type"]


# bin_order = [
#     "Non-Basin",
#     "Type 1",
#     "Type 2",
#     "Type 3",
#     "Type 4"
# ]


# color_map = {
#     "Non-Basin": "#800000",
#     "Type 1": "#33FFBD",
#     "Type 2": "#9B59B6",
#     "Type 3": "#1F618D",
#     "Type 4": "#F4D03F"
# }
# scatter_plots = []
# for b in bin_order:
#     idx = bins == b
#     sc = ax.scatter(
#         x[idx],
#         y[idx],
#         s=50,
#         color=color_map[b],
#         label=b,
#         edgecolor="k",
#         picker=True
#     )
#     sc.df_indices = df_station.index[idx].to_numpy()
#     scatter_plots.append(sc)
# ax.axhline(0, color="k", linestyle="--", linewidth=1)

# ax.tick_params(labelsize=16, direction="in", axis="both", which="both")
# ax.set_xlim(min(x), max(x))
# ax.set_ylim(-max(y), max(y))
# ax.text(0.02, 0.95, rf"$T = {T_val:g}\,\mathrm{{s}}$",transform=ax.transAxes, fontsize=18,fontweight='bold',verticalalignment="top",horizontalalignment="left")
# ax.set_xlabel(r"Depth to the 1.0 km/s $V_S$ horizon ($Z_{1.0}$)", size=20)
# ax.set_ylabel(r"$|r_{no\ basins}|-|r_{with\ basins}|$", size=20)
# ax.legend(fontsize=14, frameon=True)
# def onpick(event):
#     sc = event.artist
#     ind = event.ind[0]  # index within this scatter

#     df_ind = sc.df_indices[ind]
#     basin = df_station.loc[df_ind, "stat_name"]

#     print(f"Clicked basin: {basin}")

#     # Remove old annotations (optional but recommended)
#     for txt in ax.texts:
#         txt.remove()

#     ax.annotate(
#         basin,
#         (sc.get_offsets()[ind, 0], sc.get_offsets()[ind, 1]),
#         xytext=(10, 10),
#         textcoords="offset points",
#         fontsize=11,
#         bbox=dict(boxstyle="round,pad=0.25", fc="yellow", alpha=0.7)
#     )

#     fig.canvas.draw_idle()


# fig.canvas.mpl_connect("pick_event", onpick)
# fig.savefig(os.path.join(out_dir, "Paper_2_Type"), dpi=300)
# #%% Loop for all T's for above - Type
# pSA_cols = [c for c in delta_abs_df.columns if c.startswith("pSA_")]

# T_vals = {c: float(c.split("_")[1]) for c in pSA_cols}
# pSA_cols_sel = [c for c, T in T_vals.items() if T > 1.0]
# pSA_cols_sel = sorted(pSA_cols_sel, key=lambda c: T_vals[c])

# out_dir = "delta_residual_vs_Z1_Tgt1s_Type"
# os.makedirs(out_dir, exist_ok=True)

# bin_order = [
#     "Non-Basin",
#     "Type 1",
#     "Type 2",
#     "Type 3",
#     "Type 4"
# ]

# color_map = {
#     "Non-Basin": "#800000",
#     "Type 1": "#33FFBD",
#     "Type 2": "#9B59B6",
#     "Type 3": "#1F618D",
#     "Type 4": "#F4D03F"
# }

# def plot_delta_vs_Z1_by_type(T_sel, df_station, delta_abs_df, out_dir):

#     T_val = float(T_sel.split("_")[1])

#     fig, ax = plt.subplots(figsize=(7.95, 5.66), constrained_layout=True)

#     x = df_station["dZ1"]
#     y = delta_abs_df[T_sel]
#     bins = df_station["Type"]

#     for b in bin_order:
#         idx = bins == b
#         ax.scatter(
#             x[idx],
#             y[idx],
#             s=50,
#             color=color_map[b],
#             label=b,
#             edgecolor="k"
#         )

#     ax.axhline(0, color="k", linestyle="--", linewidth=1)

#     ax.tick_params(labelsize=16, direction="in", axis="both", which="both")
#     ax.set_xlim(min(x), max(x))
#     ax.set_ylim(-max(y), max(y))

#     ax.set_xlabel(
#         r"Depth to the 1.0 km/s $V_S$ horizon ($Z_{1.0}$)",
#         size=20
#     )
#     ax.set_ylabel(
#         r"Median $|r_{no\ basins}|-|r_{with\ basins}|$",
#         size=20
#     )

#     # Period annotation
#     ax.text(
#         0.02, 0.95,
#         rf"$T = {T_val:g}\,\mathrm{{s}}$",
#         transform=ax.transAxes,
#         fontsize=18,
#         fontweight="bold",
#         va="top", ha="left"
#     )

#     ax.legend(fontsize=14, frameon=True)

#     # Save figure
#     fname = f"delta_vs_Z1_Type_T_{T_val:g}s.png"
#     fig.savefig(os.path.join(out_dir, fname), dpi=300)
#     plt.close(fig)


# for T_sel in pSA_cols_sel:
#     plot_delta_vs_Z1_by_type(
#         T_sel,
#         df_station,
#         delta_abs_df,
#         out_dir
#     )

# #%% Scatter using geom category
# T_sel = 'pSA_2.000000000000'

# fig, ax = plt.subplots(figsize=(7.95, 5.66), constrained_layout=True)
# out_dir = "delta_vs_Z1_by_Geom_Tgt1s"
# x = df_station["dZ1"]
# y = delta_abs_df[T_sel]
# bins = df_station["Geomorphology"]


# bin_order = [
#     "Basin",
#     "Basin Edge",
#     "Valley",
#     "Hill"
# ]


# color_map = {
#     "Basin": "#1E90FF",
#     "Basin Edge": "#66C266",
#     "Valley": "#9370DB",
#     "Hill": "#993333"
# }
# scatter_plots = []
# for b in bin_order:
#     idx = bins == b
#     sc = ax.scatter(
#         x[idx],
#         y[idx],
#         s=50,
#         color=color_map[b],
#         label=b,
#         edgecolor="k",
#         picker=True
#     )
#     sc.df_indices = df_station.index[idx].to_numpy()
#     scatter_plots.append(sc)
# ax.axhline(0, color="k", linestyle="--", linewidth=1)

# ax.tick_params(labelsize=16, direction="in", axis="both", which="both")
# ax.set_xlim(min(x), max(x))
# ax.set_ylim(-max(y), max(y))
# ax.text(0.02, 0.95, rf"$T = {T_val:g}\,\mathrm{{s}}$",transform=ax.transAxes, fontsize=18,fontweight='bold',verticalalignment="top",horizontalalignment="left")
# ax.set_xlabel(r"Depth to the 1.0 km/s $V_S$ horizon ($Z_{1.0}$)", size=20)
# ax.set_ylabel(r"$|r_{no\ basins}|-|r_{with\ basins}|$", size=20)
# ax.legend(fontsize=14, frameon=True)
# def onpick(event):
#     sc = event.artist
#     ind = event.ind[0]  # index within this scatter

#     df_ind = sc.df_indices[ind]
#     basin = df_station.loc[df_ind, "stat_name"]

#     print(f"Clicked basin: {basin}")

#     # Remove old annotations (optional but recommended)
#     for txt in ax.texts:
#         txt.remove()

#     ax.annotate(
#         basin,
#         (sc.get_offsets()[ind, 0], sc.get_offsets()[ind, 1]),
#         xytext=(10, 10),
#         textcoords="offset points",
#         fontsize=11,
#         bbox=dict(boxstyle="round,pad=0.25", fc="yellow", alpha=0.7)
#     )

#     fig.canvas.draw_idle()


# fig.canvas.mpl_connect("pick_event", onpick)
# fig.savefig(os.path.join(out_dir, "Paper_2_Geom"), dpi=300)
# #%% Loop over above - Geom
# pSA_cols = [c for c in delta_abs_df.columns if c.startswith("pSA_")]
# T_vals = {c: float(c.split("_")[1]) for c in pSA_cols}
# pSA_cols_sel = sorted([c for c, T in T_vals.items() if T > 1.0], key=lambda c: T_vals[c])

# out_dir = "delta_vs_Z1_by_Geom_Tgt1s"
# os.makedirs(out_dir, exist_ok=True)

# bin_order = [
#     "Basin",
#     "Basin Edge",
#     "Valley",
#     "Hill"
# ]

# color_map = {
#     "Basin": "#1E90FF",
#     "Basin Edge": "#66C266",
#     "Valley": "#9370DB",
#     "Hill": "#993333"
# }

# def plot_delta_vs_Z1_by_geom(T_sel, df_station, delta_abs_df, out_dir):
#     T_val = float(T_sel.split("_")[1])

#     fig, ax = plt.subplots(figsize=(7.95, 5.66), constrained_layout=True)

#     x = df_station["dZ1"]
#     y = delta_abs_df[T_sel]
#     bins = df_station["Geomorphology"]

#     for b in bin_order:
#         idx = bins == b
#         ax.scatter(
#             x[idx],
#             y[idx],
#             s=50,
#             color=color_map[b],
#             label=b,
#             edgecolor="k"
#         )

#     ax.axhline(0, color="k", linestyle="--", linewidth=1)

#     ax.tick_params(labelsize=16, direction="in", axis="both", which="both")
#     ax.set_xlim(min(x), max(x))
#     ax.set_ylim(-max(y), max(y))

#     ax.set_xlabel(r"Depth to the 1.0 km/s $V_S$ horizon ($Z_{1.0}$)", size=20)
#     ax.set_ylabel(r"$|r_{no\ basins}|-|r_{with\ basins}|$", size=20)

#     ax.text(
#         0.02, 0.95,
#         rf"$T = {T_val:g}\,\mathrm{{s}}$",
#         transform=ax.transAxes,
#         fontsize=18,
#         fontweight="bold",
#         va="top",
#         ha="left"
#     )

#     ax.legend(fontsize=14, frameon=True)

#     fname = f"delta_vs_Z1_Geom_T_{T_val:g}s.png"
#     fig.savefig(os.path.join(out_dir, fname), dpi=300)
#     # plt.close(fig)

# for T_sel in pSA_cols_sel:
#     plot_delta_vs_Z1_by_geom(T_sel, df_station, delta_abs_df, out_dir)

#%% FK's Paper 3 - pSA
T = np.array(T)
pSA_cols = [c for c in siteres_ff.columns if c.startswith("pSA_")]
siteres_ff_pSA  = siteres_ff_sorted[pSA_cols]
siteres_nb_pSA  = siteres_nb_sorted[pSA_cols]
sitereserr_ff_pSA  = sitereserr_ff_sorted[pSA_cols]
sitereserr_nb_pSA  = sitereserr_nb_sorted[pSA_cols]
out_dir = "Basin_performance/pSA"
os.makedirs(out_dir, exist_ok=True)
basin = stations_sel_sorted['Basin']
siteres_systematic_ff_pSA = siteres_ff_pSA + np.array(bias_SA_ff)
siteres_systematic_nb_pSA = siteres_nb_pSA + np.array(bias_SA_nb)
abs_ff = siteres_systematic_ff_pSA.abs()
abs_nb = siteres_systematic_nb_pSA.abs()
delta_abs_df = abs_nb - abs_ff 
band_idx = (T >= 1.0) & (T <= 5.0)
score = delta_abs_df.loc[:, band_idx].median(axis=1)
tmp = pd.DataFrame(index=score.index)
tmp["BasinType"] = stations_sel_sorted.loc[score.index, "BasinType_2p09"].astype(str).str.strip()
tmp["Geomorphology"] = stations_sel_sorted.loc[score.index, "Geomorphology"].astype(str).str.strip()
tmp["score"] = score



tmp["class"] = np.where(tmp["score"] <= 0, "Decrease in performance (≤ 0)", np.where(tmp["score"] < 0.1, "Slight increase in performance (0 – 0.1)", "Considerable increase in performance (≥ 0.1)"))

classes = ["Decrease in performance (≤ 0)", "Slight increase in performance (0 – 0.1)", "Considerable increase in performance (≥ 0.1)"]
order = tmp["BasinType"].value_counts().index.tolist() 
order = ["Non-Basin","Unmodeled","Type 1","Type 2","Type 3","Type 4"]
ct = (tmp.groupby(["BasinType", "class"]).size()
        .unstack(fill_value=0)
        .reindex(index=order)
        .reindex(columns=classes, fill_value=0))

color = ['r','b','g']


def plot_stacked(ax, ct_in, xlabels, classes, colors, add_percent=False):
    bottom = np.zeros(len(ct_in))
    i = 0
    for c in classes:
        ax.bar(xlabels, ct_in[c].values, bottom=bottom, color=colors[i], label=c)
        bottom += ct_in[c].values
        i += 1

    if add_percent:
        total = ct_in.sum(axis=1).values
        considerable = ct_in["Considerable increase in performance (≥ 0.1)"].values
        for i, (xlab, tot, cons) in enumerate(zip(ct_in.index, total, considerable)):
            if tot > 0 and cons > 0:
                ax.text(i, ct_in.loc[xlab].sum() - cons/2, f"{100*cons/tot:.1f}%",
                        ha="center", va="center", fontsize=12, color="white", fontweight="bold")

    ax.set_ylabel("Count", fontsize=20)
    ax.tick_params(axis="both", labelrotation=35, labelsize=20)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.4)

# Type 1
type1 = tmp[tmp["BasinType"] == "Type 1"].copy()

geom_order = ["Basin", "Basin Edge", "Valley"]

geom_order = [g for g in geom_order if g in type1["Geomorphology"].unique()] + \
             [g for g in type1["Geomorphology"].value_counts().index.tolist() if g not in geom_order]

ct_g = (type1.groupby(["Geomorphology", "class"]).size()
          .unstack(fill_value=0)
          .reindex(index=geom_order)
          .reindex(columns=classes, fill_value=0))

# Type 3
type3 = tmp[tmp["BasinType"] == "Type 3"].copy()
geom_order3 = ["Basin", "Basin Edge", "Valley"]
geom_order3 = [g for g in geom_order3 if g in type3["Geomorphology"].unique()] + \
              [g for g in type3["Geomorphology"].value_counts().index.tolist() if g not in geom_order3]

ct_g3 = (type3.groupby(["Geomorphology", "class"]).size()
           .unstack(fill_value=0)
           .reindex(index=geom_order3)
           .reindex(columns=classes, fill_value=0))

# Type 4
type4 = tmp[tmp["BasinType"] == "Type 4"].copy()
geom_order4 = ["Basin", "Basin Edge", "Valley"]
geom_order4 = [g for g in geom_order4 if g in type4["Geomorphology"].unique()] + \
              [g for g in type4["Geomorphology"].value_counts().index.tolist() if g not in geom_order4]

ct_g4 = (type4.groupby(["Geomorphology", "class"]).size()
           .unstack(fill_value=0)
           .reindex(index=geom_order4)
           .reindex(columns=classes, fill_value=0))



fig, axes = plt.subplots(2, 2, figsize=(14.5, 9.5), constrained_layout=True)

# (a) All sites (BasinType)
plot_stacked(axes[0, 0], ct, ct.index, classes, color, add_percent=True)
axes[0, 0].set_title("All sites", fontsize=18, fontweight="bold")

# (b) Type 1 sites (Geomorphology)
plot_stacked(axes[0, 1], ct_g, ct_g.index, classes, color, add_percent=False)
axes[0, 1].set_title("Type 1 sites", fontsize=18, fontweight="bold")

# (c) Type 3 sites (Geomorphology)
plot_stacked(axes[1, 0], ct_g3, ct_g3.index, classes, color, add_percent=False)
axes[1, 0].set_title("Type 3 sites", fontsize=18, fontweight="bold")

# (d) Type 4 sites (Geomorphology)
plot_stacked(axes[1, 1], ct_g4, ct_g4.index, classes, color, add_percent=False)
axes[1, 1].set_title("Type 4 sites", fontsize=18, fontweight="bold")

axes[0,0].legend(fontsize=14, frameon=True,loc = 'best')
panel_labels = ["(a)", "(b)", "(c)", "(d)"]
for lab, ax in zip(panel_labels, axes.flatten()):
    ax.text(-0.07, 1.05, lab, transform=ax.transAxes,
            fontsize=18, fontweight="bold", va="top", ha="left")

plt.savefig(os.path.join(out_dir, "All_Type1_Type3_Type4_T1-5s.pdf"))
#%% FK's Paper 3 - EAS
f = np.array(f)
EAS_cols = [c for c in siteres_ff.columns if c.startswith("EAS_")]
siteres_ff_EAS  = siteres_ff_sorted[EAS_cols]
siteres_nb_EAS  = siteres_nb_sorted[EAS_cols]
sitereserr_ff_EAS  = sitereserr_ff_sorted[EAS_cols]
sitereserr_nb_EAS  = sitereserr_nb_sorted[EAS_cols]
out_dir = "Basin_performance/EAS"
os.makedirs(out_dir, exist_ok=True)
basin = stations_sel_sorted['Basin']
siteres_systematic_ff_EAS = siteres_ff_EAS + np.array(bias_EAS_ff)
siteres_systematic_nb_EAS = siteres_nb_EAS + np.array(bias_EAS_nb)
abs_ff = siteres_systematic_ff_EAS.abs()
abs_nb = siteres_systematic_nb_EAS.abs()
delta_abs_df = abs_nb - abs_ff 
band_idx = (f >= 0.2) & (f <= 1.0)
score = delta_abs_df.loc[:, band_idx].median(axis=1)
tmp = pd.DataFrame(index=score.index)
tmp["BasinType"] = stations_sel_sorted.loc[score.index, "BasinType_2p09"].astype(str).str.strip()
tmp["Geomorphology"] = stations_sel_sorted.loc[score.index, "Geomorphology"].astype(str).str.strip()
tmp["score"] = score



tmp["class"] = np.where(tmp["score"] <= 0, "Decrease in performance (≤ 0)", np.where(tmp["score"] < 0.1, "Slight increase in performance (0 – 0.1)", "Considerable increase in performance (≥ 0.1)"))

classes = ["Decrease in performance (≤ 0)", "Slight increase in performance (0 – 0.1)", "Considerable increase in performance (≥ 0.1)"]
order = tmp["BasinType"].value_counts().index.tolist() 
order = ["Non-Basin","Unmodeled","Type 1","Type 2","Type 3","Type 4"]
ct = (tmp.groupby(["BasinType", "class"]).size()
        .unstack(fill_value=0)
        .reindex(index=order)
        .reindex(columns=classes, fill_value=0))

color = ['r','b','g']


def plot_stacked(ax, ct_in, xlabels, classes, colors, add_percent=False):
    bottom = np.zeros(len(ct_in))
    i = 0
    for c in classes:
        ax.bar(xlabels, ct_in[c].values, bottom=bottom, color=colors[i], label=c)
        bottom += ct_in[c].values
        i += 1

    if add_percent:
        total = ct_in.sum(axis=1).values
        considerable = ct_in["Considerable increase in performance (≥ 0.1)"].values
        for i, (xlab, tot, cons) in enumerate(zip(ct_in.index, total, considerable)):
            if tot > 0 and cons > 0:
                ax.text(i, ct_in.loc[xlab].sum() - cons/2, f"{100*cons/tot:.1f}%",
                        ha="center", va="center", fontsize=12, color="white", fontweight="bold")

    ax.set_ylabel("Count", fontsize=20)
    ax.tick_params(axis="both", labelrotation=35, labelsize=20)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.4)

# Type 1
type1 = tmp[tmp["BasinType"] == "Type 1"].copy()

geom_order = ["Basin", "Basin Edge", "Valley"]

geom_order = [g for g in geom_order if g in type1["Geomorphology"].unique()] + \
             [g for g in type1["Geomorphology"].value_counts().index.tolist() if g not in geom_order]

ct_g = (type1.groupby(["Geomorphology", "class"]).size()
          .unstack(fill_value=0)
          .reindex(index=geom_order)
          .reindex(columns=classes, fill_value=0))

# Type 3
type3 = tmp[tmp["BasinType"] == "Type 3"].copy()
geom_order3 = ["Basin", "Basin Edge", "Valley"]
geom_order3 = [g for g in geom_order3 if g in type3["Geomorphology"].unique()] + \
              [g for g in type3["Geomorphology"].value_counts().index.tolist() if g not in geom_order3]

ct_g3 = (type3.groupby(["Geomorphology", "class"]).size()
           .unstack(fill_value=0)
           .reindex(index=geom_order3)
           .reindex(columns=classes, fill_value=0))

# Type 4
type4 = tmp[tmp["BasinType"] == "Type 4"].copy()
geom_order4 = ["Basin", "Basin Edge", "Valley"]
geom_order4 = [g for g in geom_order4 if g in type4["Geomorphology"].unique()] + \
              [g for g in type4["Geomorphology"].value_counts().index.tolist() if g not in geom_order4]

ct_g4 = (type4.groupby(["Geomorphology", "class"]).size()
           .unstack(fill_value=0)
           .reindex(index=geom_order4)
           .reindex(columns=classes, fill_value=0))



fig, axes = plt.subplots(2, 2, figsize=(14.5, 9.5), constrained_layout=True)

# (a) All sites (BasinType)
plot_stacked(axes[0, 0], ct, ct.index, classes, color, add_percent=True)
axes[0, 0].set_title("All sites", fontsize=18, fontweight="bold")

# (b) Type 1 sites (Geomorphology)
plot_stacked(axes[0, 1], ct_g, ct_g.index, classes, color, add_percent=False)
axes[0, 1].set_title("Type 1 sites", fontsize=18, fontweight="bold")

# (c) Type 3 sites (Geomorphology)
plot_stacked(axes[1, 0], ct_g3, ct_g3.index, classes, color, add_percent=False)
axes[1, 0].set_title("Type 3 sites", fontsize=18, fontweight="bold")

# (d) Type 4 sites (Geomorphology)
plot_stacked(axes[1, 1], ct_g4, ct_g4.index, classes, color, add_percent=False)
axes[1, 1].set_title("Type 4 sites", fontsize=18, fontweight="bold")

axes[0,0].legend(fontsize=14, frameon=True,loc = 'best')
panel_labels = ["(a)", "(b)", "(c)", "(d)"]
for lab, ax in zip(panel_labels, axes.flatten()):
    ax.text(-0.07, 1.05, lab, transform=ax.transAxes,
            fontsize=18, fontweight="bold", va="top", ha="left")

plt.savefig(os.path.join(out_dir, "All_Type1_Type3_Type4_T1-5s_EAS.pdf"))
#%% Z1_2p09 categories analysis - pSA - Paper 2
pSA_cols = [c for c in siteres_ff.columns if c.startswith("pSA_")]

# Use the filtered station/residual datasets
# These should already exclude Non-Basin and Unmodeled sites
mask = ~stations_sel_sorted["BasinType_2p09"].isin(["Non-Basin", "Unmodeled"])
stations_selected = stations_sel_sorted.loc[mask]
siteres_ff_selected = siteres_ff_sorted.loc[mask]
siteres_nb_selected = siteres_nb_sorted.loc[mask]

siteres_ff_pSA = siteres_ff_selected[pSA_cols]
siteres_nb_pSA = siteres_nb_selected[pSA_cols]

IM_cols = ["PGA", "PGV", "CAV", "AI", "Ds575", "Ds595"]

siteres_ff_IMs = siteres_ff_selected[IM_cols]
siteres_nb_IMs = siteres_nb_selected[IM_cols]


# ------------------------------------------------------------
# Define Z1_2p09 categories
# ------------------------------------------------------------

z1_bins = [-np.inf, 100, 300, 500, np.inf]

z1_labels = [
    r"Shallow ($Z_{1.0,\mathrm{sim}} < 100$ m)",
    r"Moderate ($100 \leq Z_{1.0,\mathrm{sim}} < 300$ m)",
    r"Deep ($300 \leq Z_{1.0,\mathrm{sim}} < 500$ m)",
    r"Very deep ($Z_{1.0,\mathrm{sim}} \geq 500$ m)"
]

z1_category = pd.cut(
    stations_selected["Z1_2p09"],
    bins=z1_bins,
    labels=z1_labels,
    right=False
)

print(z1_category.value_counts(dropna=False).sort_index())


# ------------------------------------------------------------
# Add systematic bias
# ------------------------------------------------------------

siteres_systematic_ff_pSA = siteres_ff_pSA + np.array(bias_SA_ff)
siteres_systematic_nb_pSA = siteres_nb_pSA + np.array(bias_SA_nb)

siteres_systematic_ff_IMs = siteres_ff_IMs + np.array(bias_IMs_ff)
siteres_systematic_nb_IMs = siteres_nb_IMs + np.array(bias_IMs_nb)


# ------------------------------------------------------------
# Group residuals by Z1_2p09 category
# ------------------------------------------------------------

grouped_data_ff_pSA = siteres_systematic_ff_pSA.groupby(z1_category)
grouped_data_nb_pSA = siteres_systematic_nb_pSA.groupby(z1_category)

grouped_data_ff_IMs = siteres_systematic_ff_IMs.groupby(z1_category)
grouped_data_nb_IMs = siteres_systematic_nb_IMs.groupby(z1_category)

desired_order = z1_labels


# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------
x = np.arange(len(IM_cols))
x_values = ["PGA", "PGV", "CAV", "AI", "$D_{s575}$", "$D_{s595}$"]

fig = plt.figure(figsize=(12.72, 8.15), constrained_layout=True)

gs = fig.add_gridspec(
    nrows=2,
    ncols=4,
    width_ratios=[3, 1, 3, 1],
    hspace=0.03,
    wspace=0.01
)

ax_sa_list = []
ax_im_list = []

for i, cluster_id in enumerate(desired_order):

    # Skip empty categories, if any
    if cluster_id not in grouped_data_ff_pSA.groups:
        continue

    row = i // 2
    col = (i % 2) * 2

    ax_sa = fig.add_subplot(gs[row, col])
    ax_im = fig.add_subplot(gs[row, col + 1])

    ax_sa_list.append(ax_sa)
    ax_im_list.append(ax_im)

    ff_cluster = grouped_data_ff_pSA.get_group(cluster_id)
    nb_cluster = grouped_data_nb_pSA.get_group(cluster_id)

    ff_cluster_IM = grouped_data_ff_IMs.get_group(cluster_id)
    nb_cluster_IM = grouped_data_nb_IMs.get_group(cluster_id)

    ff_mean_IM = ff_cluster_IM.mean(axis=0)
    nb_mean_IM = nb_cluster_IM.mean(axis=0)

    # --------------------------------------------------------
    # pSA residuals
    # --------------------------------------------------------

    ax_sa.semilogx(
        T,
        ff_cluster.values.T,
        linewidth=1,
        color="gray",
        alpha=0.3
    )

    ff_mean = ff_cluster.values.T.mean(axis=1)
    ff_std = ff_cluster.values.T.std(axis=1)

    nb_mean = nb_cluster.values.T.mean(axis=1)

    ax_sa.semilogx(T, ff_mean, "b", linewidth=3)

    ax_sa.fill_between(
        T,
        ff_mean - ff_std,
        ff_mean + ff_std,
        facecolor="gray",
        edgecolor="None",
        linestyle="dashed",
        linewidth=1,
        alpha=0.3
    )

    ax_sa.semilogx(T, nb_mean, "r", linewidth=3)

    ax_sa.set_xlim([0.01, 10])
    ax_sa.set_ylim([-1.5, 1.5])

    ax_sa.axhline(0, color="k", linestyle="--")
    ax_sa.axvline(1, color="k", linestyle="--")

    # y-label only on first category column
    if i % 2 == 0:
        ax_sa.set_ylabel("${ a +  \\delta S2S_s}$", size=20)

    ax_sa.text(
        0.011,
        1.05,
        f"{cluster_id}\n(N = {len(ff_cluster)})",
        size=16,
        fontweight="bold"
    )

    ax_sa.tick_params(
        labelsize=16,
        direction="in",
        axis="both",
        which="both"
    )

    # --------------------------------------------------------
    # IM residuals
    # --------------------------------------------------------

    ax_im.set_xticks(x, x_values)

    ax_im.scatter(x_values, nb_mean_IM, s=20, c="r", marker="o")
    ax_im.scatter(x_values, ff_mean_IM, s=20, c="b", marker="o")

    ax_im.plot([-1, 10], [0, 0], color="k", linestyle="--")

    ax_im.set_xlim([-1, 6])
    ax_im.set_ylim([-1.5, 1.5])

    ax_im.set_xticklabels(x_values, rotation=90)

    ax_im.tick_params(labelsize=14, direction="in")
    ax_im.grid(color="k", linestyle=(0, (5, 10)), linewidth=0.4)
    ax_im.yaxis.tick_right()


# ------------------------------------------------------------
# Annotation, labels, and legend
# ------------------------------------------------------------

if len(ax_sa_list) > 0:

    # ax_sa_list[0].annotate(
    #     "Category mean",
    #     xy=(1.7, 0.57),
    #     xytext=(0.4, 0.95),
    #     fontsize=18,
    #     arrowprops=dict(arrowstyle="->", lw=1, color="black"),
    #     ha="left",
    #     va="center"
    # )

    ax_sa_list[0].text(
        0.011,
        -1.0,
        "Overprediction",
        size=20,
        fontstyle="italic"
    )

    ax_sa_list[0].text(
        0.011,
        0.7,
        "Underprediction",
        size=20,
        fontstyle="italic"
    )

    ax_sa_list[0].semilogx([], [], "b", linewidth=3, label="With Basins")
    ax_sa_list[0].semilogx([], [], "r", linewidth=3, label="Without Basins")

    leg = ax_sa_list[0].legend(fontsize=14, loc="lower right")
    leg.get_frame().set_edgecolor("k")

    # x-labels only for bottom row pSA panels
    for ax in ax_sa_list[2:]:
        ax.set_xlabel("Vibration Period, T (s)", size=20)
fig.savefig(os.path.join(outDir,"All Z1_Systematic.pdf"))
#%% Z1_2p09 categories analysis - EAS - Paper 2
EAS_cols = [c for c in siteres_ff.columns if c.startswith("EAS_")]

# Use filtered station/residual datasets
# These should already exclude "Non-Basin" and "Unmodeled"
mask = ~stations_sel_sorted["BasinType_2p09"].isin(["Non-Basin", "Unmodeled"])
stations_selected = stations_sel_sorted.loc[mask]
siteres_ff_selected = siteres_ff_sorted.loc[mask]
siteres_nb_selected = siteres_nb_sorted.loc[mask]

siteres_ff_EAS = siteres_ff_selected[EAS_cols]
siteres_nb_EAS = siteres_nb_selected[EAS_cols]


# ------------------------------------------------------------
# Define Z1_2p09 categories
# ------------------------------------------------------------

z1_bins = [-np.inf, 100, 300, 500, np.inf]

z1_labels = [
    r"Shallow ($Z_{1.0,\mathrm{sim}} < 100$ m)",
    r"Moderate ($100 \leq Z_{1.0,\mathrm{sim}} < 300$ m)",
    r"Deep ($300 \leq Z_{1.0,\mathrm{sim}} < 500$ m)",
    r"Very deep ($Z_{1.0,\mathrm{sim}} \geq 500$ m)"
]


z1_category = pd.cut(
    stations_selected["Z1_2p09"],
    bins=z1_bins,
    labels=z1_labels,
    right=False
)

print(z1_category.value_counts(dropna=False).sort_index())


# ------------------------------------------------------------
# Add systematic bias
# ------------------------------------------------------------

siteres_systematic_ff_EAS = siteres_ff_EAS + np.array(bias_EAS_ff)
siteres_systematic_nb_EAS = siteres_nb_EAS + np.array(bias_EAS_nb)


# ------------------------------------------------------------
# Group EAS residuals by Z1_2p09 category
# ------------------------------------------------------------

grouped_data_ff_EAS = siteres_systematic_ff_EAS.groupby(z1_category)
grouped_data_nb_EAS = siteres_systematic_nb_EAS.groupby(z1_category)

desired_order = z1_labels


# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------

fig, axes = plt.subplots(
    2,
    2,
    figsize=(9.89,  7.92),
    constrained_layout=True
)

axes_flat = axes.flatten()

axes_flat[0].text(
    1.11,
    -0.7,
    "Overprediction",
    size=20,
    fontstyle="italic"
)

axes_flat[0].text(
    1.11,
    0.7,
    "Underprediction",
    size=20,
    fontstyle="italic"
)


for i, (cluster_id, ax) in enumerate(zip(desired_order, axes_flat)):

    # Skip empty categories, if any
    if cluster_id not in grouped_data_ff_EAS.groups:
        ax.set_visible(False)
        continue

    ff_cluster = grouped_data_ff_EAS.get_group(cluster_id)
    nb_cluster = grouped_data_nb_EAS.get_group(cluster_id)

    ax.semilogx(
        f,
        ff_cluster.values.T,
        linewidth=1,
        color="gray",
        alpha=0.3
    )

    ff_mean = ff_cluster.values.T.mean(axis=1)
    ff_std = ff_cluster.values.T.std(axis=1)

    ax.semilogx(f, ff_mean, "b", linewidth=3)

    ax.fill_between(
        f,
        ff_mean - ff_std,
        ff_mean + ff_std,
        facecolor="gray",
        edgecolor="None",
        linestyle="dashed",
        linewidth=1,
        alpha=0.3
    )

    nb_mean = nb_cluster.values.T.mean(axis=1)

    ax.semilogx(f, nb_mean, "r", linewidth=3)

    ax.set_xlim([0.1, 20])
    ax.set_ylim([-1.5, 1.5])

    ax.axhline(0, color="k", linestyle="--")
    ax.axvline(1, color="k", linestyle="--")

    if i % 2 == 0:
        ax.set_ylabel("${ a +  \\delta S2S_s}$", size=20)

    ax.text(
        0.11,
        1.05,
        f"{cluster_id}\n(N = {len(ff_cluster)})",
        size=16,
        fontweight="bold"
    )

    ax.tick_params(
        labelsize=16,
        direction="in",
        axis="both",
        which="both"
    )


# ------------------------------------------------------------
# Legend and axis labels
# ------------------------------------------------------------

axes_flat[0].semilogx([], [], "b", linewidth=3, label="With Basins")
axes_flat[0].semilogx([], [], "r", linewidth=3, label="Without Basins")

leg = axes_flat[0].legend(fontsize=14, loc="lower right")
leg.get_frame().set_edgecolor("k")

for ax in axes[-1,:]:
    ax.set_xlabel("Frequency, f (Hz)", size=20)

fig.savefig(os.path.join(outDir,"All Z1_Systematic_EAS.pdf"))
#%% Total residual difference analysis by geomorphology
obs_df = im_obs.copy()
sim_ff_df = im_sim_wb.copy()   
sim_nb_df = im_sim_nb.copy()   


stations_df = stations_sel.copy()


id_cols = ["gm_id", "event_id", "stat_id"]


IM_cols = ["PGA", "PGV", "CAV", "AI", "Ds575", "Ds595"]


pSA_cols = [c for c in obs_df.columns if c.startswith("pSA_")]
EAS_cols = [c for c in obs_df.columns if c.startswith("EAS_")]

all_IM_cols = IM_cols + pSA_cols + EAS_cols




obs_sorted = (
    obs_df[id_cols + all_IM_cols]
    .sort_values(id_cols)
    .reset_index(drop=True)
)

sim_ff_sorted = (
    sim_ff_df[id_cols + all_IM_cols]
    .sort_values(id_cols)
    .reset_index(drop=True)
)

sim_nb_sorted = (
    sim_nb_df[id_cols + all_IM_cols]
    .sort_values(id_cols)
    .reset_index(drop=True)
)


if not obs_sorted[id_cols].equals(sim_ff_sorted[id_cols]):
    raise ValueError("Observation and with-basin simulation dataframes are not aligned.")

if not obs_sorted[id_cols].equals(sim_nb_sorted[id_cols]):
    raise ValueError("Observation and without-basin simulation dataframes are not aligned.")



total_res_ff = np.log(obs_sorted[all_IM_cols]) - np.log(sim_ff_sorted[all_IM_cols])
total_res_nb = np.log(obs_sorted[all_IM_cols]) - np.log(sim_nb_sorted[all_IM_cols])


total_res_ff = pd.concat([obs_sorted[id_cols], total_res_ff], axis=1)
total_res_nb = pd.concat([obs_sorted[id_cols], total_res_nb], axis=1)


diff_total_res = total_res_nb[all_IM_cols] - total_res_ff[all_IM_cols]

diff_total_res = pd.concat(
    [obs_sorted[id_cols], diff_total_res],
    axis=1
)


stations_meta = stations[["stat_id", "Geomorphology"]].drop_duplicates("stat_id")

diff_total_res = diff_total_res.merge(
    stations_meta,
    on="stat_id",
    how="left"
)

missing_geomorph = diff_total_res["Geomorphology"].isna().sum()

if missing_geomorph > 0:
    print(f"Warning: {missing_geomorph} rows have missing Geomorphology labels.")



diff_total_res_pSA = diff_total_res[pSA_cols]
diff_total_res_IMs = diff_total_res[IM_cols]

geomorph = diff_total_res["Geomorphology"]

grouped_diff_pSA = diff_total_res_pSA.groupby(geomorph)
grouped_diff_IMs = diff_total_res_IMs.groupby(geomorph)



summary_diff_pSA_mean = grouped_diff_pSA.mean()
summary_diff_pSA_std  = grouped_diff_pSA.std()

summary_diff_IMs_mean = grouped_diff_IMs.mean()
summary_diff_IMs_std  = grouped_diff_IMs.std()



desired_order = ["Basin", "Basin Edge", "Valley", "Hill"]

x = np.arange(len(IM_cols))
x_values = ["PGA", "PGV", "CAV", "AI", "$D_{s575}$", "$D_{s595}$"]

fig = plt.figure(figsize=(12, 8), constrained_layout=True)

gs = fig.add_gridspec(
    nrows=2,
    ncols=4,
    width_ratios=[3, 1, 3, 1],
    hspace=0.03,
    wspace=0.01
)

ax_sa_list = []

for i, cluster_id in enumerate(desired_order):

    if cluster_id not in grouped_diff_pSA.groups:
        print(f"Skipping {cluster_id}: no records found.")
        continue

    row = i // 2
    col = (i % 2) * 2

    ax_sa = fig.add_subplot(gs[row, col])
    ax_im = fig.add_subplot(gs[row, col + 1])
    ax_sa_list.append(ax_sa)

    # Get grouped data
    diff_cluster_pSA = grouped_diff_pSA.get_group(cluster_id)
    diff_cluster_IMs = grouped_diff_IMs.get_group(cluster_id)

    # Mean and std
    diff_mean_pSA = diff_cluster_pSA.values.T.mean(axis=1)
    diff_std_pSA  = diff_cluster_pSA.values.T.std(axis=1)

    diff_mean_IMs = diff_cluster_IMs.mean(axis=0)

    # pSA plot
    ax_sa.semilogx(
        T,
        diff_cluster_pSA.values.T,
        linewidth=1,
        color="gray",
        alpha=0.3
    )

    ax_sa.semilogx(
        T,
        diff_mean_pSA,
        color="k",
        linewidth=3,
        label="Mean difference"
    )

    ax_sa.fill_between(
        T,
        diff_mean_pSA - diff_std_pSA,
        diff_mean_pSA + diff_std_pSA,
        facecolor="gray",
        edgecolor="None",
        linewidth=1,
        alpha=0.3
    )

    ax_sa.axhline(0, color="k", linestyle="--")
    ax_sa.axvline(1, color="k", linestyle="--")

    ax_sa.set_xlim([0.01, 10])
    ax_sa.set_ylim([-1.5, 1.5])

    if i % 2 == 0:
        ax_sa.set_ylabel(r'$\Delta_{\mathrm{implicit}} - \Delta_{\mathrm{explicit}}$', size=20)

    ax_sa.text(
        0.011,
        1.28,
        f"{cluster_id} (N = {len(diff_cluster_pSA)})",
        size=17,
        fontweight="bold"
    )

    ax_sa.tick_params(
        labelsize=16,
        direction="in",
        axis="both",
        which="both"
    )

    # Scalar IM plot
    ax_im.scatter(
        x,
        diff_mean_IMs.values,
        s=30,
        c="k",
        marker="o"
    )

    ax_im.plot(
        [-1, 10],
        [0, 0],
        color="k",
        linestyle="--"
    )

    ax_im.set_xticks(x)
    ax_im.set_xticklabels(x_values, rotation=90)

    ax_im.set_xlim([-1, 6])
    ax_im.set_ylim([-0.1, 0.1])

    ax_im.tick_params(
        labelsize=14,
        direction="in"
    )

    ax_im.grid(
        color="k",
        linestyle=(0, (5, 10)),
        linewidth=0.4
    )

    ax_im.yaxis.tick_right()


# Add labels to bottom pSA axes
for ax in ax_sa_list[2:]:
    ax.set_xlabel("Vibration Period, T (s)", size=20)
    ax_sa_list[0].legend(fontsize=14, loc="lower right")


# Save figure
fig.savefig(os.path.join(outDir, "Geomorphology_Difference_TotalResiduals.pdf"))
EAS_fmax = 24.77076356000

EAS_cols = [
    c for c in obs_df.columns
    if c.startswith("EAS_") and float(c.replace("EAS_", "")) <= EAS_fmax
]

diff_total_res_EAS = diff_total_res[EAS_cols]

geomorph = diff_total_res["Geomorphology"]

grouped_diff_EAS = diff_total_res_EAS.groupby(geomorph)


summary_diff_EAS_mean = grouped_diff_EAS.mean()
summary_diff_EAS_std  = grouped_diff_EAS.std()


desired_order = ["Basin", "Basin Edge", "Valley", "Hill"]

fig, axes = plt.subplots(
    2,
    2,
    figsize=(9.89, 7.92),
    constrained_layout=True
)

axes_flat = axes.flatten()

for i, (cluster_id, ax) in enumerate(zip(desired_order, axes_flat)):

    if cluster_id not in grouped_diff_EAS.groups:
        print(f"Skipping {cluster_id}: no records found.")
        continue

    # Get EAS residual-difference curves for this geomorphic category
    diff_cluster_EAS = grouped_diff_EAS.get_group(cluster_id)

    # Mean and standard deviation
    diff_mean_EAS = diff_cluster_EAS.values.T.mean(axis=1)
    diff_std_EAS  = diff_cluster_EAS.values.T.std(axis=1)

    # Plot individual curves
    ax.semilogx(
        f,
        diff_cluster_EAS.values.T,
        linewidth=1,
        color="gray",
        alpha=0.3
    )

    # Plot mean curve
    ax.semilogx(
        f,
        diff_mean_EAS,
        color="k",
        linewidth=3,
    )

    # Plot mean +/- std
    ax.fill_between(
        f,
        diff_mean_EAS - diff_std_EAS,
        diff_mean_EAS + diff_std_EAS,
        facecolor="gray",
        edgecolor="None",
        linewidth=1,
        alpha=0.3
    )

    # Reference lines
    ax.axhline(0, color="k", linestyle="--")
    ax.axvline(1, color="k", linestyle="--")

    # Limits
    ax.set_xlim([0.1, 20])
    ax.set_ylim([-1.5, 1.5])

    # Labels
    if i % 2 == 0:
        ax.set_ylabel(
            r"$\Delta_{\mathrm{implicit}} - \Delta_{\mathrm{explicit}}$",
            size=20
        )

    ax.text(
        0.11,
        1.28,
        f"{cluster_id} (N = {len(diff_cluster_EAS)})",
        size=17,
        fontweight="bold"
    )

    ax.tick_params(
        labelsize=16,
        direction="in",
        axis="both",
        which="both"
    )


# Bottom-row x labels
for ax in axes[1]:
    ax.set_xlabel("Frequency, f (Hz)", size=20)


# Legend
axes[0][0].semilogx([], [], color="k", linewidth=3, label="Mean difference")

leg = axes[0][0].legend(
    fontsize=14,
    loc="lower right"
)
leg.get_frame().set_edgecolor("k")

fig.savefig(os.path.join(outDir, "Geomorphology_Difference_TotalResiduals_EAS.pdf"))
#%% Geomorphic categories analysis - pSA - Paper 2
pSA_cols = [c for c in siteres_ff.columns if c.startswith("pSA_")]
siteres_ff_pSA  = siteres_ff_sorted[pSA_cols]
siteres_nb_pSA  = siteres_nb_sorted[pSA_cols]
IM_cols = ["PGA","PGV","CAV","AI","Ds575","Ds595"]

siteres_ff_IMs = siteres_ff_sorted[IM_cols]
siteres_nb_IMs = siteres_nb_sorted[IM_cols]




geomorph = stations_sel_sorted['Geomorphology']
siteres_systematic_ff_pSA = siteres_ff_pSA + np.array(bias_SA_ff)
siteres_systematic_nb_pSA = siteres_nb_pSA + np.array(bias_SA_nb)
siteres_systematic_ff_IMs = siteres_ff_IMs + np.array(bias_IMs_ff)
siteres_systematic_nb_IMs = siteres_nb_IMs + np.array(bias_IMs_nb)
grouped_data_ff_pSA = siteres_systematic_ff_pSA.groupby(geomorph)
grouped_data_nb_pSA = siteres_systematic_nb_pSA.groupby(geomorph)
grouped_data_ff_IMs = siteres_systematic_ff_IMs.groupby(geomorph)
grouped_data_nb_IMs = siteres_systematic_nb_IMs.groupby(geomorph)
desired_order = ['Basin']
x = np.arange(len(IM_cols))
x_values = ["PGA","PGV","CAV","AI","$D_{s575}$","$D_{s595}$"]
fig = plt.figure(figsize=(12,8),constrained_layout=True)

gs = fig.add_gridspec(
    nrows=2, ncols=4,
    width_ratios=[3,1,3,1],
    hspace=0.03,
    wspace=0.01
)
ax_sa_list = []
for i, cluster_id in enumerate(desired_order):

    row = i // 2
    col = (i % 2) * 2

    ax_sa = fig.add_subplot(gs[row, col])
    ax_im = fig.add_subplot(gs[row, col+1])
    ax_sa_list.append(ax_sa)

    ff_cluster = grouped_data_ff_pSA.get_group(cluster_id)
    nb_cluster = grouped_data_nb_pSA.get_group(cluster_id)
    ff_cluster_IM = grouped_data_ff_IMs.get_group(cluster_id)
    nb_cluster_IM = grouped_data_nb_IMs.get_group(cluster_id)

    ff_mean_IM = ff_cluster_IM.mean(axis=0)
    nb_mean_IM = nb_cluster_IM.mean(axis=0)


    ax_sa.semilogx(T, ff_cluster.values.T, linewidth=1, color='gray', alpha=0.3)
    ff_mean = ff_cluster.values.T.mean(axis=1)
    ff_std  = ff_cluster.values.T.std(axis=1)
    ax_sa.semilogx(T, ff_mean, 'b', linewidth=3)
    ax_sa.fill_between(T, ff_mean-ff_std, ff_mean+ff_std, facecolor='gray', edgecolor='None',linestyle='dashed', linewidth=1, alpha=0.3)
    nb_mean = nb_cluster.values.T.mean(axis=1)
    ax_sa.semilogx(T, nb_mean, 'r', linewidth=3)
    ax_sa.set_xlim([0.01, 10])
    ax_sa.set_ylim([-1.5, 1.5])
    ax_sa.axhline(0, color='k', linestyle='--')
    ax_sa.axvline(1, color='k', linestyle='--')

    if i % 2 == 0:
        ax_sa.set_ylabel('${ a +  \delta S2S_s}$', size=20)

    ax_sa.text(0.011, 1.28,
            f"{cluster_id} (N = {len(ff_cluster)})",
            size=17, fontweight='bold')

    ax_sa.tick_params(labelsize=16, direction='in',
                   axis='both', which='both')




    ax_im.set_xticks(x, x_values)

    ax_im.scatter(x_values, nb_mean_IM, s=20, c='r', marker='o')
    ax_im.scatter(x_values, ff_mean_IM, s=20, c='b', marker='o')

    ax_im.plot([-1,10],[0,0],color='k',linestyle='--')

    ax_im.set_xlim([-1,6])
    ax_im.set_ylim([-1.5,1.5])

    ax_im.set_xticklabels(x_values, rotation=90)

    ax_im.tick_params(labelsize=14,direction='in')
    ax_im.grid(color='k', linestyle=(0,(5,10)), linewidth=0.4)
    ax_im.yaxis.tick_right()

ax_sa_list[0].annotate("Category mean", xy=(1.7, 0.57), xytext=(0.4, 0.95), fontsize=18,arrowprops=dict(arrowstyle="->", lw=1, color="black"), ha="left", va="center")

ax_sa_list[0].text(0.011, -1.3, 'Overprediction', size=20, fontstyle='italic')
ax_sa_list[0].text(0.011,  1.0, 'Underprediction', size=20, fontstyle='italic')
ax_sa_list[0].semilogx([], [], 'b', linewidth=3, label='With Basins')
ax_sa_list[0].semilogx([], [], 'r', linewidth=3, label='Without Basins')

leg = ax_sa_list[0].legend(fontsize=14, loc='lower right')
leg.get_frame().set_edgecolor('k')

for ax in ax_sa_list[2:]:
    ax.set_xlabel('Vibration Period, T (s)', size=20)
#%% Basin geomorphic category % reduction - pSA
T = np.array(T)
bias_reduction_pct = np.abs(nb_mean) - np.abs(ff_mean)
mask = (T >= 1.0) & (T <= 5.0)     
pct_band_pSA = 100 * (1 - np.nanmean(np.abs(ff_mean)[mask]) / np.nanmean(np.abs(nb_mean)[mask]))
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(14, 6), sharex=True,constrained_layout=True)
axs[0].semilogx(T, nb_mean, linewidth=2,color='r', label='Without Basins')
axs[0].semilogx(T, ff_mean, linewidth=2,color='b', label='With Basins')
axs[0].axhline(0, color='k', linestyle='--')
axs[0].axvline(1, color='k', linestyle='--')
axs[0].set_xlim([0.01, 10])
axs[0].set_ylim([-0.5, 0.5])
axs[0].set_xlabel('Vibration Period, T (s)', fontsize=16)
axs[0].set_ylabel(r'${a}$', fontsize=16)
axs[0].tick_params(labelsize=13, direction='in', which='both')
axs[0].grid(color='gray', linestyle='dashed', linewidth=0.4)
axs[0].legend(fontsize=14)
axs[1].semilogx(T, bias_reduction_pct, linewidth=2,color='k')
axs[1].axhline(0, linestyle='--')
axs[1].axvline(1, linestyle='--')
axs[1].set_xlim([0.01, 10])
# axs[1].set_ylim([-0.12, 0.12])
axs[1].set_xlabel('Vibration Period, T (s)', fontsize=16)
axs[1].set_ylabel(r"Reduction in bias, $a$: $|a_{\mathrm{without\ basins}}| - |a_{\mathrm{with\ basins}}|$",fontsize=14)

axs[1].tick_params(labelsize=13, direction='in', which='both')
axs[1].grid(color='gray', linestyle='dashed', linewidth=0.4)

# fig.savefig(os.path.join(outDir,"All Geomorphology_Systematic.pdf"))
#%% Geomorphic categories analysis - dS2S only - WITHOUT BASINS

pSA_cols = [c for c in siteres_nb.columns if c.startswith("pSA_")]
siteres_nb_pSA  = siteres_nb_sorted[pSA_cols]

IM_cols = ["PGA","PGV","CAV","AI","Ds575","Ds595","Ds595_LF"]
siteres_nb_IMs = siteres_nb_sorted[IM_cols]

geomorph = stations_sel_sorted['Geomorphology']

# --- NO bias addition ---
grouped_data_nb_pSA = siteres_nb_pSA.groupby(geomorph)
grouped_data_nb_IMs = siteres_nb_IMs.groupby(geomorph)

desired_order = ['Basin','Basin Edge','Valley','Hill']

x = np.arange(len(IM_cols))
x_values = ["PGA","PGV","CAV","AI","$D_{s575}$","$D_{s595}$","$D_{s595,LF}$"]

fig = plt.figure(figsize=(12,8), constrained_layout=True)

gs = fig.add_gridspec(
    nrows=2, ncols=4,
    width_ratios=[3,1,3,1],
    hspace=0.03,
    wspace=0.01
)

ax_sa_list = []

for i, cluster_id in enumerate(desired_order):

    row = i // 2
    col = (i % 2) * 2

    ax_sa = fig.add_subplot(gs[row, col])
    ax_im = fig.add_subplot(gs[row, col+1])
    ax_sa_list.append(ax_sa)

    nb_cluster = grouped_data_nb_pSA.get_group(cluster_id)
    nb_cluster_IM = grouped_data_nb_IMs.get_group(cluster_id)

    nb_mean_IM = nb_cluster_IM.mean(axis=0)

    # ---- pSA plots ----
    ax_sa.semilogx(T, nb_cluster.values.T, linewidth=1, color='gray', alpha=0.3)

    nb_mean = nb_cluster.values.T.mean(axis=1)
    nb_std  = nb_cluster.values.T.std(axis=1)

    ax_sa.semilogx(T, nb_mean, 'r', linewidth=3)

    ax_sa.fill_between(
        T,
        nb_mean-nb_std,
        nb_mean+nb_std,
        facecolor='gray',
        edgecolor='None',
        linestyle='dashed',
        linewidth=1,
        alpha=0.3
    )

    ax_sa.set_xlim([0.01, 10])
    ax_sa.set_ylim([-1.5, 1.5])

    ax_sa.axhline(0, color='k', linestyle='--')
    ax_sa.axvline(1, color='k', linestyle='--')

    if i % 2 == 0:
        ax_sa.set_ylabel('$\\delta S2S_s$', size=20)

    ax_sa.text(
        0.011, 1.28,
        f"{cluster_id} (N = {len(nb_cluster)})",
        size=17,
        fontweight='bold'
    )

    ax_sa.tick_params(
        labelsize=16,
        direction='in',
        axis='both',
        which='both'
    )

    # ---- Non-SA IM plots ----
    ax_im.set_xticks(x, x_values)

    ax_im.scatter(x_values, nb_mean_IM, s=20, c='r', marker='o')

    ax_im.plot([-1,10],[0,0],color='k',linestyle='--')

    ax_im.set_xlim([-1,7])
    ax_im.set_ylim([-1.5,1.5])

    ax_im.set_xticklabels(x_values, rotation=90)

    ax_im.tick_params(labelsize=14, direction='in')
    ax_im.grid(color='k', linestyle=(0,(5,10)), linewidth=0.4)

    ax_im.yaxis.tick_right()


ax_sa_list[0].text(0.011, -1.3, 'Overprediction', size=20, fontstyle='italic')
ax_sa_list[0].text(0.011,  1.0, 'Underprediction', size=20, fontstyle='italic')

ax_sa_list[0].semilogx([], [], 'r', linewidth=3, label='Without Basins')

leg = ax_sa_list[0].legend(fontsize=14, loc='lower right')
leg.get_frame().set_edgecolor('k')

for ax in ax_sa_list[2:]:
    ax.set_xlabel('Vibration Period, T (s)', size=20)

fig.savefig(os.path.join(outDir,"All_Geomorphology_dS2S_NoBasins.pdf"))
#%% Geomorphic categories analysis - only dS2S - basins
pSA_cols = [c for c in siteres_ff.columns if c.startswith("pSA_")]
siteres_ff_pSA  = siteres_ff_sorted[pSA_cols]
siteres_nb_pSA  = siteres_nb_sorted[pSA_cols]
IM_cols = ["PGA","PGV","CAV","AI","Ds575","Ds595","Ds595_LF"]

siteres_ff_IMs = siteres_ff_sorted[IM_cols]
siteres_nb_IMs = siteres_nb_sorted[IM_cols]




geomorph = stations_sel_sorted['Geomorphology']
siteres_systematic_ff_pSA = siteres_ff_pSA 
siteres_systematic_ff_IMs = siteres_ff_IMs

grouped_data_ff_pSA = siteres_systematic_ff_pSA.groupby(geomorph)
grouped_data_ff_IMs = siteres_systematic_ff_IMs.groupby(geomorph)

desired_order = ['Basin','Basin Edge','Valley','Hill']
x = np.arange(len(IM_cols))
x_values = ["PGA","PGV","CAV","AI","$D_{s575}$","$D_{s595}$","$D_{s595,LF}$"]
fig = plt.figure(figsize=(12,8),constrained_layout=True)

gs = fig.add_gridspec(
    nrows=2, ncols=4,
    width_ratios=[3,1,3,1],
    hspace=0.03,
    wspace=0.01
)
ax_sa_list = []
for i, cluster_id in enumerate(desired_order):

    row = i // 2
    col = (i % 2) * 2

    ax_sa = fig.add_subplot(gs[row, col])
    ax_im = fig.add_subplot(gs[row, col+1])
    ax_sa_list.append(ax_sa)

    ff_cluster = grouped_data_ff_pSA.get_group(cluster_id)
    nb_cluster = grouped_data_nb_pSA.get_group(cluster_id)
    ff_cluster_IM = grouped_data_ff_IMs.get_group(cluster_id)
    nb_cluster_IM = grouped_data_nb_IMs.get_group(cluster_id)

    ff_mean_IM = ff_cluster_IM.mean(axis=0)
    nb_mean_IM = nb_cluster_IM.mean(axis=0)


    ax_sa.semilogx(T, ff_cluster.values.T, linewidth=1, color='gray', alpha=0.3)
    ff_mean = ff_cluster.values.T.mean(axis=1)
    ff_std  = ff_cluster.values.T.std(axis=1)
    nb_mean = nb_cluster.values.T.mean(axis=1)
    nb_std  = nb_cluster.values.T.std(axis=1)
    ax_sa.semilogx(T, ff_mean, 'b', linewidth=3)
    ax_sa.fill_between(T, ff_mean-ff_std, ff_mean+ff_std, facecolor='gray', edgecolor='None',linestyle='dashed', linewidth=1, alpha=0.3)

    ax_sa.set_xlim([0.01, 10])
    ax_sa.set_ylim([-1.5, 1.5])
    ax_sa.axhline(0, color='k', linestyle='--')
    ax_sa.axvline(1, color='k', linestyle='--')

    if i % 2 == 0:
        ax_sa.set_ylabel('${ \delta S2S_s}$', size=20)

    ax_sa.text(0.011, 1.28,
            f"{cluster_id} (N = {len(ff_cluster)})",
            size=17, fontweight='bold')

    ax_sa.tick_params(labelsize=16, direction='in',
                   axis='both', which='both')




    ax_im.set_xticks(x, x_values)

    # ax_im.scatter(x_values, nb_mean_IM, s=20, c='r', marker='o')
    ax_im.scatter(x_values, ff_mean_IM, s=20, c='b', marker='o')

    ax_im.plot([-1,10],[0,0],color='k',linestyle='--')

    ax_im.set_xlim([-1,7])
    ax_im.set_ylim([-1.5,1.5])

    ax_im.set_xticklabels(x_values, rotation=90)

    ax_im.tick_params(labelsize=14,direction='in')
    ax_im.grid(color='k', linestyle=(0,(5,10)), linewidth=0.4)
    ax_im.yaxis.tick_right()

ax_sa_list[0].text(0.011, -1.3, 'Overprediction', size=20, fontstyle='italic')
ax_sa_list[0].text(0.011,  1.0, 'Underprediction', size=20, fontstyle='italic')
ax_sa_list[0].semilogx([], [], 'b', linewidth=3, label='With Basins')
# ax_sa_list[0].semilogx([], [], 'r', linewidth=3, label='Without Basins')

leg = ax_sa_list[0].legend(fontsize=14, loc='lower right')
leg.get_frame().set_edgecolor('k')

for ax in ax_sa_list[2:]:
    ax.set_xlabel('Vibration Period, T (s)', size=20)

fig.savefig(os.path.join(outDir,"All Geomorphology_onlydS2S_basins.pdf"))
#%% Geomorphic categories analysis - EAS
EAS_cols = [c for c in siteres_ff.columns if c.startswith("EAS_")]
siteres_ff_EAS  = siteres_ff_sorted[EAS_cols]
siteres_nb_EAS  = siteres_nb_sorted[EAS_cols]
geomorph = stations_sel_sorted['Geomorphology']
siteres_systematic_ff_EAS = siteres_ff_EAS + np.array(bias_EAS_ff)
siteres_systematic_nb_EAS = siteres_nb_EAS + np.array(bias_EAS_nb)



grouped_data_ff_EAS = siteres_systematic_ff_EAS.groupby(geomorph)
grouped_data_nb_EAS = siteres_systematic_nb_EAS.groupby(geomorph)


desired_order = ['Basin']


fig, axes = plt.subplots(2, 2, figsize=(9.89, 7.92), constrained_layout=True)


axes[0][0].text(0.11, -1.3, 'Overprediction', size=20, fontstyle='italic')
axes[0][0].text(0.11,  1.0, 'Underprediction', size=20, fontstyle='italic')


for i, (cluster_id, ax) in enumerate(zip(desired_order, axes.flatten())):


    ff_cluster = grouped_data_ff_EAS.get_group(cluster_id)
    nb_cluster = grouped_data_nb_EAS.get_group(cluster_id)
    ax.semilogx(f, ff_cluster.values.T, linewidth=1, color='gray', alpha=0.3)
    ff_mean = ff_cluster.values.T.mean(axis=1)
    ff_std  = ff_cluster.values.T.std(axis=1)
    ax.semilogx(f, ff_mean, 'b', linewidth=3)
    ax.fill_between(f, ff_mean-ff_std, ff_mean+ff_std, facecolor='gray', edgecolor='None',linestyle='dashed', linewidth=1, alpha=0.3)
    nb_mean = nb_cluster.values.T.mean(axis=1)
    ax.semilogx(f, nb_mean, 'r', linewidth=3)
    ax.set_xlim([0.1, 20])
    ax.set_ylim([-1.5, 1.5])
    ax.axhline(0, color='k', linestyle='--')
    ax.axvline(1, color='k', linestyle='--')
    if i % 2 == 0:
        ax.set_ylabel('${ a +  \delta S2S_s}$', size=20)

    ax.text(0.11, 1.28,
            f"{cluster_id} (N = {len(ff_cluster)})",
            size=17, fontweight='bold')

    ax.tick_params(labelsize=16, direction='in', axis='both', which='both')
    # ax.grid(color='gray', linestyle='dashed', which='both', linewidth=0.4,alpha=0.5)


axes[0][0].semilogx([], [], 'b', linewidth=3, label='With Basins')
axes[0][0].semilogx([], [], 'r', linewidth=3, label='Without Basins')
leg = axes[0][0].legend(fontsize=14, loc='lower right')
leg.get_frame().set_edgecolor('k')


for ax in axes[1]:
    ax.set_xlabel('Frequency, f (Hz)', size=20)

# fig.savefig(os.path.join(outDir,"All Geomorphology_Systematic_EAS.pdf"))
#%% Basin geomorphic category % reduction - EAS
f = np.array(f)
bias_reduction_pct = np.abs(nb_mean) - np.abs(ff_mean)
mask = (f >= 0.2) & (f <= 1.0)     
pct_band_EAS = 100 * (1 - np.nanmean(np.abs(ff_mean)[mask]) / np.nanmean(np.abs(nb_mean)[mask]))
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(14, 6), sharex=True,constrained_layout=True)
axs[0].semilogx(f, nb_mean, linewidth=2,color='r', label='Without Basins')
axs[0].semilogx(f, ff_mean, linewidth=2,color='b', label='With Basins')
axs[0].axhline(0, color='k', linestyle='--')
axs[0].axvline(1, color='k', linestyle='--')
axs[0].set_xlim([0.1, 20])
axs[0].set_ylim([-1, 1])
axs[0].set_xlabel('Frequency, f (Hz)', fontsize=16)
axs[0].set_ylabel(r'$a$', fontsize=16)
axs[0].tick_params(labelsize=13, direction='in', which='both')
axs[0].grid(color='gray', linestyle='dashed', linewidth=0.4)
axs[0].legend(fontsize=14)
axs[1].semilogx(f, bias_reduction_pct, linewidth=2,color='k')
axs[1].axhline(0, linestyle='--')
axs[1].axvline(1, linestyle='--')
axs[1].set_xlim([0.1, 20])
# axs[1].set_ylim([-0.25, 0.25])
axs[1].set_xlabel('Frequency, f (Hz)', fontsize=14)
axs[1].set_ylabel(r"Reduction in bias, $a$: $|a_{\mathrm{without\ basins}}| - |a_{\mathrm{with\ basins}}|$",fontsize=14)
axs[1].tick_params(labelsize=13, direction='in', which='both')
axs[1].grid(color='gray', linestyle='dashed', linewidth=0.4)
#%% Basin Type categories analysis - pSA + non-SA IMs - Paper 2

pSA_cols = [c for c in siteres_ff.columns if c.startswith("pSA_")]
siteres_ff_pSA  = siteres_ff_sorted[pSA_cols]
siteres_nb_pSA  = siteres_nb_sorted[pSA_cols]

IM_cols = ["PGA","PGV","CAV","AI","Ds575","Ds595"]
siteres_ff_IMs = siteres_ff_sorted[IM_cols]
siteres_nb_IMs = siteres_nb_sorted[IM_cols]

bt = stations_sel_sorted['BasinType_2p09']

siteres_systematic_ff_pSA = siteres_ff_pSA + np.array(bias_SA_ff)
siteres_systematic_nb_pSA = siteres_nb_pSA + np.array(bias_SA_nb)

siteres_systematic_ff_IMs = siteres_ff_IMs + np.array(bias_IMs_ff)
siteres_systematic_nb_IMs = siteres_nb_IMs + np.array(bias_IMs_nb)

grouped_data_ff_pSA = siteres_systematic_ff_pSA.groupby(bt)
grouped_data_nb_pSA = siteres_systematic_nb_pSA.groupby(bt)

grouped_data_ff_IMs = siteres_systematic_ff_IMs.groupby(bt)
grouped_data_nb_IMs = siteres_systematic_nb_IMs.groupby(bt)

desired_order = ['Type 1','Type 2','Type 3','Type 4']
# desired_order = ['Unmodeled']

x = np.arange(len(IM_cols))
x_values = ["PGA","PGV","CAV","AI","$D_{s575}$","$D_{s595}$"]

fig = plt.figure(figsize=(12,8), constrained_layout=True)

gs = fig.add_gridspec(
    nrows=2, ncols=4,
    width_ratios=[3,1,3,1],
    hspace=0.03,
    wspace=0.01
)

ax_sa_list = []
diff_Type_list = []
type_list = []
for i, cluster_id in enumerate(desired_order):

    row = i // 2
    col = (i % 2) * 2

    ax_sa = fig.add_subplot(gs[row, col])
    ax_im = fig.add_subplot(gs[row, col+1])

    ax_sa_list.append(ax_sa)

    ff_cluster = grouped_data_ff_pSA.get_group(cluster_id)
    nb_cluster = grouped_data_nb_pSA.get_group(cluster_id)

    ff_cluster_IM = grouped_data_ff_IMs.get_group(cluster_id)
    nb_cluster_IM = grouped_data_nb_IMs.get_group(cluster_id)

    ff_mean_IM = ff_cluster_IM.mean(axis=0)
    nb_mean_IM = nb_cluster_IM.mean(axis=0)

    # SA curves
    ax_sa.semilogx(T, ff_cluster.values.T, linewidth=1, color='gray', alpha=0.3)

    ff_mean = ff_cluster.values.T.mean(axis=1)
    ff_std  = ff_cluster.values.T.std(axis=1)

    ax_sa.semilogx(T, ff_mean, 'b', linewidth=3)

    ax_sa.fill_between(
        T, ff_mean-ff_std, ff_mean+ff_std,
        facecolor='gray', edgecolor='None',
        linewidth=1, alpha=0.3
    )

    nb_mean = nb_cluster.values.T.mean(axis=1)
    ax_sa.semilogx(T, nb_mean, 'r', linewidth=3)
    diff_Type = nb_mean - ff_mean
    diff_Type_list.append(diff_Type)
    type_list.append(cluster_id)
    ax_sa.set_xlim([0.01,10])
    ax_sa.set_ylim([-1.5,1.5])

    ax_sa.axhline(0,color='k',linestyle='--')
    ax_sa.axvline(1,color='k',linestyle='--')

    if i % 2 == 0:
        ax_sa.set_ylabel('${ a +  \\delta S2S_s}$', size=20)

    ax_sa.text(
        0.011, 1.2,
        f"{cluster_id} (N = {len(ff_cluster)})",
        size=17, fontweight='bold'
    )

    ax_sa.tick_params(labelsize=16,direction='in',axis='both',which='both')
    
    # IM scatter
    ax_im.set_xticks(x, x_values)

    ax_im.scatter(x_values, nb_mean_IM, s=20, c='r', marker='o')
    ax_im.scatter(x_values, ff_mean_IM, s=20, c='b', marker='o')

    ax_im.plot([-1,10],[0,0],color='k',linestyle='--')

    ax_im.set_xlim([-1,6])
    ax_im.set_ylim([-1.5,1.5])

    ax_im.set_xticklabels(x_values, rotation=90)

    ax_im.tick_params(labelsize=14,direction='in')
    ax_im.grid(color='k', linestyle=(0,(5,10)), linewidth=0.4)

    ax_im.yaxis.tick_right()
    
diff_array = np.array(diff_Type_list)   # shape: n_types x n_periods

max_idx = np.unravel_index(np.argmax(np.abs(diff_array)), diff_array.shape)

max_type = type_list[max_idx[0]]
max_period = T[max_idx[1]]
max_difference = diff_array[max_idx]

print(f"Maximum difference = {max_difference:.3f}")
print(f"Type = {max_type}")
print(f"Period = {max_period:.3f} s")

for type_name, diff in zip(type_list, diff_array):
    idx = np.argmax(diff)   # maximum positive difference

    print(
        f"{type_name}: max difference = {diff[idx]:.3f} "
        f"at T = {T[idx]:.3f} s"
    )

ax_sa_list[0].annotate("Category mean", xy=(1.8, 0.76), xytext=(0.4, 1.15), fontsize=18,arrowprops=dict(arrowstyle="->", lw=1, color="black"), ha="left", va="center")
# annotations and legend
ax_sa_list[0].text(0.011, -1.3, 'Overprediction', size=20, fontstyle='italic')
ax_sa_list[0].text(0.011,  0.8, 'Underprediction', size=20, fontstyle='italic')

ax_sa_list[0].semilogx([], [], 'b', linewidth=3, label='With Basins')
ax_sa_list[0].semilogx([], [], 'r', linewidth=3, label='Without Basins')

leg = ax_sa_list[0].legend(fontsize=14, loc='lower right')
leg.get_frame().set_edgecolor('k')

for ax in ax_sa_list[3:]:
    ax.set_xlabel('Vibration Period, T (s)', size=20)

fig.savefig(os.path.join(outDir,"All BasinType_Systematic.pdf"))
#%% Basin Type categories analysis - pSA + non-SA IMs - Only dS2S

pSA_cols = [c for c in siteres_ff.columns if c.startswith("pSA_")]
siteres_ff_pSA  = siteres_ff_sorted[pSA_cols]
siteres_nb_pSA  = siteres_nb_sorted[pSA_cols]

IM_cols = ["PGA","PGV","CAV","AI","Ds575","Ds595"]
siteres_ff_IMs = siteres_ff_sorted[IM_cols]
siteres_nb_IMs = siteres_nb_sorted[IM_cols]

bt = stations_sel_sorted['BasinType_2p09']

siteres_systematic_ff_pSA = siteres_ff_pSA 


siteres_systematic_ff_IMs = siteres_ff_IMs 


grouped_data_ff_pSA = siteres_systematic_ff_pSA.groupby(bt)
grouped_data_nb_pSA = siteres_systematic_nb_pSA.groupby(bt)

grouped_data_ff_IMs = siteres_systematic_ff_IMs.groupby(bt)
grouped_data_nb_IMs = siteres_systematic_nb_IMs.groupby(bt)

desired_order = ['Non-Basin','Unmodeled','Type 1','Type 2','Type 3','Type 4']

x = np.arange(len(IM_cols))
x_values = ["PGA","PGV","CAV","AI","$D_{s575}$","$D_{s595}$"]

fig = plt.figure(figsize=(16,7), constrained_layout=True)

gs = fig.add_gridspec(
    nrows=2, ncols=6,
    width_ratios=[3,1,3,1,3,1],
    hspace=0.03,
    wspace=0.01
)

ax_sa_list = []

for i, cluster_id in enumerate(desired_order):

    row = i // 3
    col = (i % 3) * 2

    ax_sa = fig.add_subplot(gs[row, col])
    ax_im = fig.add_subplot(gs[row, col+1])

    ax_sa_list.append(ax_sa)

    ff_cluster = grouped_data_ff_pSA.get_group(cluster_id)
    nb_cluster = grouped_data_nb_pSA.get_group(cluster_id)

    ff_cluster_IM = grouped_data_ff_IMs.get_group(cluster_id)
    nb_cluster_IM = grouped_data_nb_IMs.get_group(cluster_id)

    ff_mean_IM = ff_cluster_IM.mean(axis=0)
    nb_mean_IM = nb_cluster_IM.mean(axis=0)

    # SA curves
    ax_sa.semilogx(T, ff_cluster.values.T, linewidth=1, color='gray', alpha=0.3)

    ff_mean = ff_cluster.values.T.mean(axis=1)
    ff_std  = ff_cluster.values.T.std(axis=1)

    ax_sa.semilogx(T, ff_mean, 'b', linewidth=3)

    ax_sa.fill_between(
        T, ff_mean-ff_std, ff_mean+ff_std,
        facecolor='gray', edgecolor='None',
        linewidth=1, alpha=0.3
    )


    ax_sa.set_xlim([0.01,10])
    ax_sa.set_ylim([-1.5,1.5])

    ax_sa.axhline(0,color='k',linestyle='--')
    ax_sa.axvline(1,color='k',linestyle='--')

    if i % 3 == 0:
        ax_sa.set_ylabel('${\\delta S2S_s}$', size=20)

    ax_sa.text(
        0.011, 1.28,
        f"{cluster_id} (N = {len(ff_cluster)})",
        size=17, fontweight='bold'
    )

    ax_sa.tick_params(labelsize=16,direction='in',axis='both',which='both')

    # IM scatter
    ax_im.set_xticks(x, x_values)


    ax_im.scatter(x_values, ff_mean_IM, s=20, c='b', marker='o')

    ax_im.plot([-1,10],[0,0],color='k',linestyle='--')

    ax_im.set_xlim([-1,6])
    ax_im.set_ylim([-1.5,1.5])

    ax_im.set_xticklabels(x_values, rotation=90)

    ax_im.tick_params(labelsize=14,direction='in')
    ax_im.grid(color='k', linestyle=(0,(5,10)), linewidth=0.4)

    ax_im.yaxis.tick_right()

# annotations and legend
ax_sa_list[0].text(0.011, -1.3, 'Overprediction', size=20, fontstyle='italic')
ax_sa_list[0].text(0.011,  1.0, 'Underprediction', size=20, fontstyle='italic')

ax_sa_list[0].semilogx([], [], 'b', linewidth=3, label='With Basins')


leg = ax_sa_list[0].legend(fontsize=14, loc='lower right')
leg.get_frame().set_edgecolor('k')

for ax in ax_sa_list[3:]:
    ax.set_xlabel('Vibration Period, T (s)', size=20)

fig.savefig(os.path.join(outDir,"All BasinType_only_dS2S.pdf"))
#%% Basin Type categories analysis - EAS
EAS_cols = [c for c in siteres_ff.columns if c.startswith("EAS_")]
siteres_ff_EAS  = siteres_ff_sorted[EAS_cols]
siteres_nb_EAS  = siteres_nb_sorted[EAS_cols]
bt = stations_sel_sorted['BasinType_2p09']
siteres_systematic_ff_EAS = siteres_ff_EAS + np.array(bias_EAS_ff)
siteres_systematic_nb_EAS = siteres_nb_EAS + np.array(bias_EAS_nb)



grouped_data_ff_EAS = siteres_systematic_ff_EAS.groupby(bt)
grouped_data_nb_EAS = siteres_systematic_nb_EAS.groupby(bt)


desired_order = ['Type 1','Type 2','Type 3', 'Type 4']
# desired_order = ['Non-Basin','Unmodeled','Type 1','Type 3', 'Type 4']


fig, axes = plt.subplots(2, 2, figsize=(10.36,  7.03), constrained_layout=True)


axes[0][0].text(0.11, -1.3, 'Overprediction', size=20, fontstyle='italic')
axes[0][0].text(0.11,  1.0, 'Underprediction', size=20, fontstyle='italic')


for i, (cluster_id, ax) in enumerate(zip(desired_order, axes.flatten())):


    ff_cluster = grouped_data_ff_EAS.get_group(cluster_id)
    nb_cluster = grouped_data_nb_EAS.get_group(cluster_id)
    ax.semilogx(f, ff_cluster.values.T, linewidth=1, color='gray', alpha=0.3)
    ff_mean = ff_cluster.values.T.mean(axis=1)
    ff_std  = ff_cluster.values.T.std(axis=1)
    ax.semilogx(f, ff_mean, 'b', linewidth=3)
    ax.fill_between(f, ff_mean-ff_std, ff_mean+ff_std, facecolor='gray', edgecolor='None',linestyle='dashed', linewidth=1, alpha=0.3)
    nb_mean = nb_cluster.values.T.mean(axis=1)
    ax.semilogx(f, nb_mean, 'r', linewidth=3)
    ax.set_xlim([0.1, 20])
    ax.set_ylim([-1.5, 1.5])
    ax.axhline(0, color='k', linestyle='--')
    ax.axvline(1, color='k', linestyle='--')
    if i % 2 == 0:
        ax.set_ylabel('${ a +  \delta S2S_s}$', size=20)

    ax.text(0.11, 1.28,
            f"{cluster_id} (N = {len(ff_cluster)})",
            size=17, fontweight='bold')

    ax.tick_params(labelsize=16, direction='in', axis='both', which='both')
    # ax.grid(color='gray', linestyle='dashed', which='both', linewidth=0.4,alpha=0.5)


axes[0][0].semilogx([], [], 'b', linewidth=3, label='With Basins')
axes[0][0].semilogx([], [], 'r', linewidth=3, label='Without Basins')
leg = axes[0][0].legend(fontsize=14, loc='lower right')
leg.get_frame().set_edgecolor('k')


for ax in axes[1]:
    ax.set_xlabel('Frequency, f (Hz)', size=20)

fig.savefig(os.path.join(outDir,"All BasinType_Systematic_EAS.pdf"))
#%% Modeled Hill and Unmodeled Hill sites categories
stations_sel_sorted["Modeled_Hill"] = np.where(
    stations_sel_sorted["Geomorphology"] == "Hill",
    stations_sel_sorted["Basin"].notna(),
    np.nan
)

valid_mask = stations_sel_sorted["Modeled_Hill"].notna()

geomorph = stations_sel_sorted.loc[valid_mask, "Modeled_Hill"]

# Apply mask to residual datasets
siteres_systematic_ff_pSA = (siteres_ff_sorted[pSA_cols] + np.array(bias_SA_ff)).loc[valid_mask]
siteres_systematic_nb_pSA = (siteres_nb_sorted[pSA_cols] + np.array(bias_SA_nb)).loc[valid_mask]

siteres_systematic_ff_IMs = (siteres_ff_sorted[IM_cols] + np.array(bias_IMs_ff)).loc[valid_mask]
siteres_systematic_nb_IMs = (siteres_nb_sorted[IM_cols] + np.array(bias_IMs_nb)).loc[valid_mask]


grouped_data_ff_pSA = siteres_systematic_ff_pSA.groupby(geomorph)
grouped_data_nb_pSA = siteres_systematic_nb_pSA.groupby(geomorph)

grouped_data_ff_IMs = siteres_systematic_ff_IMs.groupby(geomorph)
grouped_data_nb_IMs = siteres_systematic_nb_IMs.groupby(geomorph)


desired_order = [True, False]

labels_map = {
    True: "Modeled Hill",
    False: "Unmodeled Hill"
}


x = np.arange(len(IM_cols))
x_values = ["PGA","PGV","CAV","AI","$D_{s575}$","$D_{s595}$","$D_{s595,LF}$"]

fig = plt.figure(figsize=(16.81,5.6), constrained_layout=True)

gs = fig.add_gridspec(
    nrows=1, ncols=4,
    width_ratios=[3,1,3,1],
    wspace=0.05
)

ax_sa_list = []


for i, cluster_id in enumerate(desired_order):

    # Skip safely if group missing
    if cluster_id not in grouped_data_ff_pSA.groups:
        print(f"Skipping {cluster_id} (no data)")
        continue

    col = i * 2

    ax_sa = fig.add_subplot(gs[0, col])
    ax_im = fig.add_subplot(gs[0, col+1])
    ax_sa_list.append(ax_sa)

    ff_cluster = grouped_data_ff_pSA.get_group(cluster_id)
    nb_cluster = grouped_data_nb_pSA.get_group(cluster_id)

    ff_cluster_IM = grouped_data_ff_IMs.get_group(cluster_id)
    nb_cluster_IM = grouped_data_nb_IMs.get_group(cluster_id)


    ax_sa.semilogx(T, ff_cluster.values.T, color='gray', alpha=0.3)

    ff_mean = ff_cluster.values.T.mean(axis=1)
    ff_std  = ff_cluster.values.T.std(axis=1)

    ax_sa.semilogx(T, ff_mean, 'b', linewidth=3)
    ax_sa.fill_between(T, ff_mean-ff_std, ff_mean+ff_std,
                       facecolor='gray', alpha=0.3)

    nb_mean = nb_cluster.values.T.mean(axis=1)
    ax_sa.semilogx(T, nb_mean, 'r', linewidth=3)

    ax_sa.axhline(0, color='k', linestyle='--')
    ax_sa.axvline(1, color='k', linestyle='--')

    ax_sa.set_xlim([0.01, 10])
    ax_sa.set_ylim([-1.5, 1.5])

    if i == 0:
        ax_sa.set_ylabel('${ a +  \delta S2S_s}$', size=16)

    ax_sa.text(
        0.011, 1.25,
        f"{labels_map[cluster_id]} (N = {len(ff_cluster)})",
        size=14, fontweight='bold'
    )

    ax_sa.tick_params(labelsize=12, direction='in')


    ff_mean_IM = ff_cluster_IM.mean(axis=0)
    nb_mean_IM = nb_cluster_IM.mean(axis=0)

    ax_im.scatter(x_values, nb_mean_IM, c='r', s=20)
    ax_im.scatter(x_values, ff_mean_IM, c='b', s=20)
    
    ax_im.grid(color='k', linestyle=(0,(5,10)), linewidth=0.4)
    ax_im.axhline(0, color='k', linestyle='--')

    ax_im.set_xticks(x)
    ax_im.set_xticklabels(x_values, rotation=90)

    ax_im.set_xlim([-1,7])
    ax_im.set_ylim([-1.5,1.5])

    ax_im.tick_params(labelsize=11, direction='in')
    ax_im.yaxis.tick_right()


if len(ax_sa_list) > 0:
    ax_sa_list[0].semilogx([], [], 'b', linewidth=3, label='With Basins')
    ax_sa_list[0].semilogx([], [], 'r', linewidth=3, label='Without Basins')

    leg = ax_sa_list[0].legend(loc='lower right', fontsize=12)
    leg.get_frame().set_edgecolor('k')


fig.savefig(os.path.join(outDir, "Modeled_Hill_Systematic.pdf"))
# plt.close()
#%% a+dS2S - pSA + non-SA IMs
def read_profile(file):
    df = pd.read_csv(
        file, delim_whitespace=True, skiprows=5,
        names=['Elevation (km)', 'Vp (km/s)', 'Vs (km/s)', 'Rho (g/cm^3)']
    )
    df = df.dropna(subset=['Vs (km/s)'])
    df['Elevation (m)'] = df['Elevation (km)'] * 1000
    return df


profile_dir_nb = r"C:\Users\ati47\OneDrive - University of Canterbury\Desktop\PhD\10. Research\Objective 2\4. Analysis\Final\All_NZ_SIMS_no_basin\Profiles"
profile_dir_ff = r"C:\Users\ati47\OneDrive - University of Canterbury\Desktop\PhD\10. Research\Objective 2\4. Analysis\Final\All_NZ_SIMS\Profiles"

pSA_cols = [c for c in siteres_ff.columns if c.startswith("pSA_")]
siteres_ff_pSA  = siteres_ff_sorted[pSA_cols]
siteres_nb_pSA  = siteres_nb_sorted[pSA_cols]

IM_cols = ["PGA","PGV","CAV","AI","Ds575","Ds595"]

siteres_ff_IMs = siteres_ff_sorted[IM_cols]
siteres_nb_IMs = siteres_nb_sorted[IM_cols]

siteres_systematic_ff_pSA = siteres_ff_pSA + np.array(bias_SA_ff)
siteres_systematic_nb_pSA = siteres_nb_pSA + np.array(bias_SA_nb)

siteres_systematic_ff_IMs = siteres_ff_IMs + np.array(bias_IMs_ff)
siteres_systematic_nb_IMs = siteres_nb_IMs + np.array(bias_IMs_nb)

out_dir = "stations_pSA_systematic"
os.makedirs(out_dir, exist_ok=True)

x = np.arange(len(IM_cols))
x_labels = ["PGA","PGV","CAV","AI",r"$D_{s575}$",r"$D_{s595}$"]

# for i in range(len(stations)):
i = 64
stat = stations.iloc[i]

name  = stat['stat_name']
geom  = stat['Geomorphology']
basin = stat['BasinType_2p09']
vs30  = stat['Vs30']
qvs30 = stat['Q_Vs30']
T0    = stat['T0']
qt0   = stat['Q_T0']
Z1    = stat['Z1_2p09']

ind1  = stat['stat_key']
ind2  = stations_sel_sorted[stations_sel_sorted['stat_key']==ind1].index

stat_id = i + 1
n_events = event_counts.get(stat_id, 0)

y_nb = siteres_systematic_nb_pSA.iloc[ind2,:].T.values
y_ff = siteres_systematic_ff_pSA.iloc[ind2,:].T.values

y_nb_IM = siteres_systematic_nb_IMs.iloc[ind2,:].values.flatten()
y_ff_IM = siteres_systematic_ff_IMs.iloc[ind2,:].values.flatten()

# figure with (a) residuals and (b) Vs profile
fig = plt.figure(figsize=(12, 6), constrained_layout=True)

outer_gs = fig.add_gridspec(
    nrows=1,
    ncols=2,
    width_ratios=[4, 1.4],
    wspace=0.08
)

# subplot (a): nested pSA + IM panels
gs_a = outer_gs[0, 0].subgridspec(
    nrows=1,
    ncols=2,
    width_ratios=[3, 1],
    wspace=0.02
)

ax_psa = fig.add_subplot(gs_a[0, 0])
ax_im  = fig.add_subplot(gs_a[0, 1])

# subplot (b): Vs profiles
ax_vs = fig.add_subplot(outer_gs[0, 1])

# annotations
ax_psa.text(0.011,-1.3,'Overprediction',size=16,fontstyle='italic')
ax_psa.text(0.011, 1.0,'Underprediction',size=16,fontstyle='italic')

# pSA curves
ax_psa.semilogx(T, y_nb,'r',linewidth=2,label='Without Basins')
ax_psa.semilogx(T, y_ff,'b',linewidth=2,label='With Basins')

ax_psa.axhline(0,color='k',linestyle='--')
ax_psa.axvline(1,color='k',linestyle='--')

ax_psa.set_xlim([0.01,10])
# ax_psa.set_ylim([-1.5,1.5])

ax_psa.set_xlabel('Vibration Period, T (s)',size=18)
ax_psa.set_ylabel(r'$a+\delta S2S_s$',size=18)

ax_psa.tick_params(labelsize=16,direction='in',axis='both',which='both')
ax_psa.grid(color='gray',linestyle='dashed',which='both',linewidth=0.4)

title = f"{name} | {n_events} | {Z1} m | {vs30} m/s ({qvs30}) | {T0} s ({qt0}) | {geom} | {basin}"
ax_psa.set_title(title,fontsize=14,fontweight='bold')

leg = ax_psa.legend(fontsize=14,loc='lower right')
leg.get_frame().set_edgecolor('k')

# non-SA IM scatter
ax_im.scatter(x,y_nb_IM,s=40,c='r',marker='o')
ax_im.scatter(x,y_ff_IM,s=40,c='b',marker='o')

ax_im.axhline(0,color='k',linestyle='--')

ax_im.set_xlim([-0.5,len(IM_cols)-0.5])
# ax_im.set_ylim([-1.5,1.5])

ax_im.set_xticks(x)
ax_im.set_xticklabels(x_labels,rotation=90)

ax_im.tick_params(labelsize=14,direction='in',axis='both',which='both')
ax_im.grid(color='gray',linestyle='dashed',which='both',linewidth=0.4,alpha=0.5)

ax_im.yaxis.set_label_position("right")
ax_im.yaxis.tick_right()

# -------------------------
# (b) Vs profile comparison
# -------------------------
profile_file_nb = os.path.join(profile_dir_nb, f"Profile_{name}.txt")
profile_file_ff = os.path.join(profile_dir_ff, f"Profile_{name}.txt")

if os.path.exists(profile_file_nb) and os.path.exists(profile_file_ff):

    profile_nb = read_profile(profile_file_nb)
    profile_ff = read_profile(profile_file_ff)

    ax_vs.step(
        profile_nb['Vs (km/s)'] * 1000,
        profile_nb['Elevation (m)'],
        where='post',
        color='r',
        linewidth=2,
        label='Without Basins'
    )

    ax_vs.step(
        profile_ff['Vs (km/s)'] * 1000,
        profile_ff['Elevation (m)'],
        where='post',
        color='b',
        linewidth=2,
        linestyle='--',
        label='With Basins'
    )

    ax_vs.set_xlabel(r'$V_s$ (m/s)', fontsize=16)
    ax_vs.set_ylabel('Elevation (m)', fontsize=16)

    ax_vs.tick_params(labelsize=14, direction='in', axis='both', which='both')
    ax_vs.grid(color='gray', linestyle='dashed', linewidth=0.4, alpha=0.5)
    ax_vs.legend(fontsize=12, loc='best')
    ax_vs.set_ylim([-2000, 200])

else:
    ax_vs.text(
        0.5, 0.5,
        'Profile file\nnot found',
        ha='center',
        va='center',
        transform=ax_vs.transAxes,
        fontsize=12
    )
    ax_vs.set_axis_off()
fname = f"{name.replace('/','_')}.png"

plt.savefig(os.path.join(out_dir,fname),dpi=600)
# plt.close(fig)

print("Done! All station PNGs saved.")
#%% Tail stations by dS2S at SA(2s): explicit vs implicit (basins only)
import os, shutil
import pandas as pd

period_col = "pSA_2.000000000000"     # confirm exact column name in siteres frames
id_col     = "Unnamed: 0"             # holds "stat_1", "stat_10", ...
N          = 10                       # per tail  -> 20 per representation

src_dir  = "stations_pSA_systematic"
dst_root = "stations_pSA_systematic_tails"

# --- basin-only stations, build stat_id -> stat_name map ---
excluded = {"Non-Basin", "Unmodeled"}
basins   = stations[~stations["BasinType_2p09"].isin(excluded)].copy()
id_to_name = dict(zip(basins["stat_id"], basins["stat_name"]))   # 175 entries

def siteres_to_name(s):
    # "stat_12" -> 12 -> stat_name ; None if not a basin station
    return id_to_name.get(int(str(s).split("_")[1]))

def tail_set(siteres, label):
    df = siteres[[id_col, period_col]].copy()
    df["stat_name"] = df[id_col].map(siteres_to_name)
    df = df.dropna(subset=["stat_name", period_col])   # drops non-basins automatically
    df = df.rename(columns={period_col: "dS2S_2s"})
    pos = df.nlargest(N, "dS2S_2s").assign(tail="pos")
    neg = df.nsmallest(N, "dS2S_2s").assign(tail="neg")
    out = pd.concat([pos, neg], ignore_index=True)
    out["selected_by"] = label
    return out

exp = tail_set(siteres_ff_sorted, "explicit")
imp = tail_set(siteres_nb_sorted, "implicit")
print(f"explicit: matched {len(exp)} / expected 20")
print(f"implicit: matched {len(imp)} / expected 20")

# --- copy the existing a+dS2S+profile PNGs into tail folders ---
def copy_tails(sel, subfolder):
    for _, r in sel.iterrows():
        dst_dir = os.path.join(dst_root, subfolder, r["tail"])
        os.makedirs(dst_dir, exist_ok=True)
        fname = f"{r['stat_name'].replace('/', '_')}.png"
        src = os.path.join(src_dir, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(dst_dir, fname))
        else:
            print("missing PNG:", fname)

copy_tails(exp, "explicit_dS2S_2s")
copy_tails(imp, "implicit_dS2S_2s")

# --- paired table: both representations side by side for every selected station ---
def name_value_map(siteres):
    df = siteres[[id_col, period_col]].copy()
    df["stat_name"] = df[id_col].map(siteres_to_name)
    return dict(zip(df.dropna(subset=["stat_name"])["stat_name"], df[period_col]))

exp_val, imp_val = name_value_map(siteres_ff_sorted), name_value_map(siteres_nb_sorted)
names = pd.Index(exp["stat_name"]).append(pd.Index(imp["stat_name"])).unique()
table = pd.DataFrame({
    "stat_name":        names,
    "dS2S_2s_explicit": [exp_val.get(n) for n in names],
    "dS2S_2s_implicit": [imp_val.get(n) for n in names],
})
os.makedirs(dst_root, exist_ok=True)
table.to_csv(os.path.join(dst_root, "tail_stations_dS2S_2s.csv"), index=False)
print("Done. Tail folders + paired table written.")
#%% Organize existing station PNGs by Z1_2p09

import os
import shutil
import pandas as pd

# Folder where your existing PNGs are saved
src_dir = "stations_pSA_systematic"

# New organized output folder
dst_root = "stations_pSA_systematic_by_Z1"
os.makedirs(dst_root, exist_ok=True)


def z1_category(z1):
    if pd.isna(z1):
        return "Z1_missing"
    elif z1 < 50:
        return "Z1_lt_50"
    elif z1 < 100:
        return "Z1_50_100"
    elif z1 < 150:
        return "Z1_100_150"
    elif z1 < 200:
        return "Z1_150_200"
    elif z1 <= 500:
        return "Z1_200_500"
    else:
        return "Z1_gt_500"


missing_images = []

for _, stat in stations.iterrows():
    name = stat["stat_name"]
    z1 = stat["Z1_2p09"]

    fname = f"{name.replace('/', '_')}.png"

    src_path = os.path.join(src_dir, fname)

    category = z1_category(z1)
    dst_dir = os.path.join(dst_root, category)
    os.makedirs(dst_dir, exist_ok=True)

    dst_path = os.path.join(dst_dir, fname)

    if os.path.exists(src_path):
        shutil.copy2(src_path, dst_path)
        # Use this instead if you want to MOVE rather than COPY:
        # shutil.move(src_path, dst_path)
    else:
        missing_images.append(fname)

print("Done! Images organized by Z1_2p09.")

if missing_images:
    print(f"{len(missing_images)} images were not found:")
    for f in missing_images:
        print(f)
#%% Organize existing station PNGs by Z1_2p09 (basins only)
import os
import shutil
import pandas as pd

# Folder where your existing PNGs are saved
src_dir = "stations_pSA_systematic"

# New organized output folder
dst_root = "stations_pSA_systematic_by_Z1_Robin"
os.makedirs(dst_root, exist_ok=True)

def z1_category(z1):
    if pd.isna(z1):
        return "Z1_missing"
    elif z1 < 100:
        return "shallow"
    elif z1 < 300:
        return "moderate"
    elif z1 < 500:
        return "deep"
    else:
        return "very_deep"

# Basin types to exclude
excluded_basin_types = {"Non-Basin", "Unmodeled"}

missing_images = []
skipped_non_basin = 0

for _, stat in stations.iterrows():
    basin_type = stat["BasinType_2p09"]

    # Skip stations that are Non-Basin or Unmodeled
    if basin_type in excluded_basin_types:
        skipped_non_basin += 1
        continue

    name = stat["stat_name"]
    z1 = stat["Z1_2p09"]
    fname = f"{name.replace('/', '_')}.png"
    src_path = os.path.join(src_dir, fname)

    category = z1_category(z1)
    dst_dir = os.path.join(dst_root, category)
    os.makedirs(dst_dir, exist_ok=True)
    dst_path = os.path.join(dst_dir, fname)

    if os.path.exists(src_path):
        shutil.copy2(src_path, dst_path)
        # Use this instead if you want to MOVE rather than COPY:
        # shutil.move(src_path, dst_path)
    else:
        missing_images.append(fname)

print("Done! Basin-only images organized by Z1_2p09.")
print(f"{skipped_non_basin} stations skipped (Non-Basin / Unmodeled).")
if missing_images:
    print(f"{len(missing_images)} images were not found:")
    for f in missing_images:
        print(f)
#%% Organize pSA systematic
png_folder = "stations_pSA_systematic"
geojson_folder = "../NZVM2p09"      # folder containing NZVM2p09 polygons
output_root = "sorted_stations_pSA_systematic"

os.makedirs(output_root, exist_ok=True)
geo_layers = {}   

for file in os.listdir(geojson_folder):
    if file.endswith(".geojson"):
        name = os.path.splitext(file)[0]
        gdf = gpd.read_file(os.path.join(geojson_folder, file))
        geo_layers[name] = gdf
        os.makedirs(os.path.join(output_root, name), exist_ok=True)


remaining_folder = os.path.join(output_root, "remaining")
os.makedirs(remaining_folder, exist_ok=True)
for i, row in stations.iterrows():

    stat_name = row["stat_name"]
    lat = row["Latitude"]
    lon = row["Longitude"]


    png_name = f"{stat_name.replace('/', '_')}.png"
    png_path = os.path.join(png_folder, png_name)

    if not os.path.exists(png_path):
        print(f"PNG missing for station: {stat_name}")
        continue

    # Create point
    pt = Point(lon, lat)

    placed = False


    for layer_name, layer_gdf in geo_layers.items():

        inside = layer_gdf.contains(pt).any()

        if inside:
            dest = os.path.join(output_root, layer_name, png_name)
            shutil.copyfile(png_path, dest)
            placed = True
            break


    if not placed:
        dest = os.path.join(remaining_folder, png_name)
        shutil.copyfile(png_path, dest)

print("Sorting complete!")
#%% a+dS2S - EAS
EAS_cols = [c for c in siteres_ff.columns if c.startswith("EAS_")]
siteres_ff_EAS  = siteres_ff_sorted[EAS_cols]
siteres_nb_EAS  = siteres_nb_sorted[EAS_cols]
siteres_systematic_ff_EAS = siteres_ff_EAS + np.array(bias_EAS_ff)
siteres_systematic_nb_EAS = siteres_nb_EAS + np.array(bias_EAS_nb)
out_dir = "stations_EAS_systematic"
os.makedirs(out_dir, exist_ok=True)
# for i in range(len(stations)):
i = 45
stat = stations.iloc[i]
name  = stat['stat_name']
geom  = stat['Geomorphology']
basin = stat['BasinType_2p09']
vs30  = stat['Vs30']
qvs30 = stat['Q_Vs30']
T0    = stat['T0']
qt0   = stat['Q_T0']
Z1    = stat['Z1_2p09']
stat_id = i + 1
ind1  = stat['stat_key']
ind2  = stations_sel_sorted[stations_sel_sorted['stat_key']==ind1].index
y_nb = siteres_systematic_nb_EAS.iloc[ind2, :].T.values   
y_ff = siteres_systematic_ff_EAS.iloc[ind2, :].T.values
n_events = event_counts.get(stat_id, 0)
fig = plt.figure(figsize=(12, 6), constrained_layout=True)

outer_gs = fig.add_gridspec(
    nrows=1,
    ncols=2,
    width_ratios=[4, 1.4],
    wspace=0.08
)

# subplot (a): EAS
ax = fig.add_subplot(outer_gs[0, 0])


# subplot (b): Vs profiles
ax_vs = fig.add_subplot(outer_gs[0, 1])

ax.semilogx(f, y_nb, 'r', linewidth=2, label='Without Basins')
ax.semilogx(f, y_ff, 'b', linewidth=2, label='With Basins')

ax.axhline(0, color='k', linestyle='--')
ax.set_xlim([0.1, 20])
# ax.set_ylim([-1.5, 1.5])
ax.tick_params(labelsize=20, direction='in', axis='both', which='both')
ax.grid(color='gray', linestyle='dashed', which='both', linewidth=0.4)
ax.set_xlabel('Frequency, f (Hz)', size=20)
ax.set_ylabel(r'$a + \delta S2S_s$', size=20)

title = f"{name} | {n_events} | {Z1} m | {vs30} m/s ({qvs30}) | {T0} s ({qt0}) | {geom} | {basin}"
ax.set_title(title, fontsize=16,fontweight='bold')
ax.legend(fontsize=20)

    
# (b) Vs profile comparison
# -------------------------
profile_file_nb = os.path.join(profile_dir_nb, f"Profile_{name}.txt")
profile_file_ff = os.path.join(profile_dir_ff, f"Profile_{name}.txt")

if os.path.exists(profile_file_nb) and os.path.exists(profile_file_ff):

    profile_nb = read_profile(profile_file_nb)
    profile_ff = read_profile(profile_file_ff)

    ax_vs.step(
        profile_nb['Vs (km/s)'] * 1000,
        profile_nb['Elevation (m)'],
        where='post',
        color='r',
        linewidth=2,
        label='Without Basins'
    )

    ax_vs.step(
        profile_ff['Vs (km/s)'] * 1000,
        profile_ff['Elevation (m)'],
        where='post',
        color='b',
        linewidth=2,
        linestyle='--',
        label='With Basins'
    )

    ax_vs.set_xlabel(r'$V_s$ (m/s)', fontsize=16)
    ax_vs.set_ylabel('Elevation (m)', fontsize=16)
    ax_vs.set_ylim([-1000, 0])
    ax_vs.tick_params(labelsize=14, direction='in', axis='both', which='both')
    ax_vs.grid(color='gray', linestyle='dashed', linewidth=0.4, alpha=0.5)
    ax_vs.legend(fontsize=12, loc='best')

else:
    ax_vs.text(
        0.5, 0.5,
        'Profile file\nnot found',
        ha='center',
        va='center',
        transform=ax_vs.transAxes,
        fontsize=12
    )
    ax_vs.set_axis_off()
    
fname = f"{name.replace('/', '_')}.png"   
plt.savefig(os.path.join(out_dir, fname), dpi=600)
# plt.close(fig)

print("Done! All station PNGs saved.")
#%% Organize existing station PNGs by Z1_2p09

import os
import shutil
import pandas as pd

# Folder where your existing PNGs are saved
src_dir = "stations_EAS_systematic"

# New organized output folder
dst_root = "stations_EAS_systematic_by_Z1"
os.makedirs(dst_root, exist_ok=True)


def z1_category(z1):
    if pd.isna(z1):
        return "Z1_missing"
    elif z1 < 50:
        return "Z1_lt_50"
    elif z1 < 100:
        return "Z1_50_100"
    elif z1 < 150:
        return "Z1_100_150"
    elif z1 < 200:
        return "Z1_150_200"
    elif z1 <= 500:
        return "Z1_200_500"
    else:
        return "Z1_gt_500"


missing_images = []

for _, stat in stations.iterrows():
    name = stat["stat_name"]
    z1 = stat["Z1_2p09"]

    fname = f"{name.replace('/', '_')}.png"

    src_path = os.path.join(src_dir, fname)

    category = z1_category(z1)
    dst_dir = os.path.join(dst_root, category)
    os.makedirs(dst_dir, exist_ok=True)

    dst_path = os.path.join(dst_dir, fname)

    if os.path.exists(src_path):
        shutil.copy2(src_path, dst_path)
        # Use this instead if you want to MOVE rather than COPY:
        # shutil.move(src_path, dst_path)
    else:
        missing_images.append(fname)

print("Done! Images organized by Z1_2p09.")

if missing_images:
    print(f"{len(missing_images)} images were not found:")
    for f in missing_images:
        print(f)
#%% Organize EAS systematic
png_folder = "stations_EAS_systematic"
geojson_folder = "../NZVM2p09"      # folder containing NZVM2p09 polygons
output_root = "sorted_stations_EAS_systematic"

os.makedirs(output_root, exist_ok=True)
geo_layers = {}   

for file in os.listdir(geojson_folder):
    if file.endswith(".geojson"):
        name = os.path.splitext(file)[0]
        gdf = gpd.read_file(os.path.join(geojson_folder, file))
        geo_layers[name] = gdf
        os.makedirs(os.path.join(output_root, name), exist_ok=True)


remaining_folder = os.path.join(output_root, "remaining")
os.makedirs(remaining_folder, exist_ok=True)
for i, row in stations.iterrows():

    stat_name = row["stat_name"]
    lat = row["Latitude"]
    lon = row["Longitude"]


    png_name = f"{stat_name.replace('/', '_')}.png"
    png_path = os.path.join(png_folder, png_name)

    if not os.path.exists(png_path):
        print(f"PNG missing for station: {stat_name}")
        continue

    # Create point
    pt = Point(lon, lat)

    placed = False


    for layer_name, layer_gdf in geo_layers.items():

        inside = layer_gdf.contains(pt).any()

        if inside:
            dest = os.path.join(output_root, layer_name, png_name)
            shutil.copyfile(png_path, dest)
            placed = True
            break


    if not placed:
        dest = os.path.join(remaining_folder, png_name)
        shutil.copyfile(png_path, dest)

print("Sorting complete!")
#%%
# #%% Old Vs New dS2S on the same figure
# for station_id in Simnames:
#     # station_id  = 'ADCS'
#     OldVs30 = OldStats[OldStats['stat_name']==station_id]['vs30'].values[0]
#     NewVs30 = NewStats[NewStats['stat_name']==station_id]['vs30'].values[0]
#     QVs30   = NewStats[NewStats['stat_name']==station_id]['QVs30'].values[0]
#     T       = OlddS2S.columns
#     oldind  = 'Station_' + str(OldStats.index[OldStats['stat_name']==station_id].values[0])
#     newind  = 'Station_' + str(NewStats.index[NewStats['stat_name']==station_id].values[0])
#     fig,ax  = plt.subplots(figsize=(8.91,6.99),constrained_layout=True)
#     plt.semilogx(T,OlddS2S.loc[oldind],'r',label='Lee et al. (2022)',linewidth=1.5)
#     plt.semilogx(T,NewdS2S.loc[newind],'b',label='This study',linewidth=1.5)
#     plt.fill_between(T, np.subtract(NewdS2S.loc[newind],(1.0 * NewS2Serr.loc[newind])), np.add(NewdS2S.loc[newind],(1.0 * NewS2Serr.loc[newind])),facecolor=[0.8, 0.8, 1.0], edgecolor=None, linestyle='dashed', linewidth=1.5, alpha=0.6)
#     plt.fill_between(T, np.subtract(OlddS2S.loc[oldind],(1.0 * OldS2Serr.loc[oldind])), np.add(OlddS2S.loc[oldind],(1.0 * OldS2Serr.loc[oldind])),facecolor=[1, 0.8, 0.8], edgecolor=None, linestyle='dashed', linewidth=1.5, alpha=0.6)
#     plt.title(f"{station_id}, OldVs30 = {OldVs30}, NewVs30 = {NewVs30} m/s ({QVs30})",fontsize=18,fontweight='bold')
#     plt.xlabel('Vibration Period, T (s)', size=18)
#     plt.ylabel('${\it \delta S2S_s}$', size=18)
#     plt.xlim([0.01,10])
#     plt.ylim([-1.5,1.5])
#     plt.axhline(0, color='gray', linewidth=1.5,linestyle='--')
#     plt.axvline(1,c='gray',linestyle='--')
#     plt.grid(color='gray', linestyle='dashed', which='both', linewidth=0.4)
#     plt.tick_params(labelsize=16,direction='in', axis='both', which='both')
#     leg = plt.legend(fontsize=18,loc='lower right')
#     leg.get_frame().set_edgecolor('k')
#     plt.text(min(plt.xlim())+0.005, max(plt.ylim())*0.90, 'Underprediction', size=16, fontweight='bold')
#     plt.text(min(plt.xlim())+0.005, min(plt.ylim())*0.90, 'Overprediction', size=16, fontweight='bold')
#     plt.savefig(os.path.join('GeomorphologyOldVsNew','%s.png' %station_id), dpi=300)
#     plt.close(fig)
# #%% Old Vs New a + dS2S on the same figure
# for station_id in Simnames:
#     # station_id  = 'CRLZ'
#     OldVs30 = OldStats[OldStats['stat_name']==station_id]['vs30'].values[0]
#     NewVs30 = NewStats[NewStats['stat_name']==station_id]['vs30'].values[0]
#     QVs30   = NewStats[NewStats['stat_name']==station_id]['QVs30'].values[0]
#     T       = aOlddS2S.columns
#     oldind  = 'Station_' + str(OldStats.index[OldStats['stat_name']==station_id].values[0])
#     newind  = 'Station_' + str(NewStats.index[NewStats['stat_name']==station_id].values[0])
#     fig,ax  = plt.subplots(figsize=(8.91,6.99),constrained_layout=True)
#     plt.semilogx(T,aOlddS2S.loc[oldind],'r',label='Lee et al. (2022)',linewidth=1.5)
#     plt.semilogx(T,aNewdS2S.loc[newind],'b',label='This study',linewidth=1.5)
#     plt.fill_between(T, np.subtract(aNewdS2S.loc[newind],(1.0 * NewS2Serr.loc[newind])), np.add(aNewdS2S.loc[newind],(1.0 * NewS2Serr.loc[newind])),facecolor=[0.8, 0.8, 1.0], edgecolor=None, linestyle='dashed', linewidth=1.5, alpha=0.6)
#     plt.fill_between(T, np.subtract(aOlddS2S.loc[oldind],(1.0 * OldS2Serr.loc[oldind])), np.add(aOlddS2S.loc[oldind],(1.0 * OldS2Serr.loc[oldind])),facecolor=[1, 0.8, 0.8], edgecolor=None, linestyle='dashed', linewidth=1.5, alpha=0.6)
#     plt.title(f"{station_id}, OldVs30 = {OldVs30}, NewVs30 = {NewVs30} m/s ({QVs30})",fontsize=18,fontweight='bold')
#     plt.xlabel('Vibration Period, T (s)', size=18)
#     plt.ylabel('${ a +  \delta S2S_s}$', size=18)
#     plt.xlim([0.01,10])
#     plt.ylim([-1.5,1.5])
#     plt.axhline(0, color='gray', linewidth=1.5,linestyle='--')
#     plt.axvline(1,c='gray',linestyle='--')
#     plt.grid(color='gray', linestyle='dashed', which='both', linewidth=0.4)
#     plt.tick_params(labelsize=16,direction='in', axis='both', which='both')
#     leg = plt.legend(fontsize=18,loc='lower right')
#     leg.get_frame().set_edgecolor('k')
#     plt.text(min(plt.xlim())+0.005, max(plt.ylim())*0.90, 'Underprediction', size=16, fontweight='bold')
#     plt.text(min(plt.xlim())+0.005, min(plt.ylim())*0.90, 'Overprediction', size=16, fontweight='bold')
#     plt.savefig(os.path.join('Bias added','%s.png' %station_id), dpi=300)
#     plt.close(fig)
# #%% Station subcategory analysis
# df_categories = pd.read_csv(Path(r"C:\Users\ati47\OneDrive - University of Canterbury\Desktop\PhD\10. Research\Objective 1\1. Residuals\3. Vs30 sensitivity\Station_subcategory.csv"))
# df_categories = df_categories.sort_values(by='stat_name',ascending=True)
# OldStats['Oldstat_id'] = ''
# NewStats['Newstat_id']=  ''
# for i in range(1,383):
#     OldStats['Oldstat_id'][i] = 'Station_'+ str(OldStats.index.values[i-1])
    
# for j in range(1,385):
#     NewStats['Newstat_id'][j] = 'Station_'+ str(NewStats.index.values[j-1])
    
# df_categories = pd.merge(df_categories,OldStats[['stat_name','Oldstat_id']],on='stat_name')
# df_categories = pd.merge(df_categories,NewStats[['stat_name','Newstat_id']],on='stat_name')
# grouped_data  = df_categories.groupby('Substation category')
# oldstatids    = df_categories['Oldstat_id'].unique()
# newstatids    = df_categories['Newstat_id'].unique()
# filtaOlddS2S  = aOlddS2S[aOlddS2S.index.isin(oldstatids)]
# filtaOlderrdS2S = OldS2Serr[OldS2Serr.index.isin(oldstatids)]
# filtaNewdS2S  = aNewdS2S[aNewdS2S.index.isin(newstatids)]
# filtaNewerrdS2S = NewS2Serr[NewS2Serr.index.isin(newstatids)]
# cluster_labels= np.array(df_categories['Substation category'])
# fig, axes = plt.subplots(2, 2, figsize=(9.89, 7.92),constrained_layout = True)
# grouped_data = filtaNewdS2S.groupby(cluster_labels)
# cluster_mean = filtaNewdS2S.groupby(cluster_labels).mean()
# cluster_std  = filtaNewdS2S.groupby(cluster_labels).std()
# oldclustermn = filtaOlddS2S.groupby(cluster_labels).mean()
# oldclustersd = filtaOlddS2S.groupby(cluster_labels).std()
# i =0 
# axes[0][0].text(0.45,-1.4,'Overprediction',size=20,fontstyle='italic')
# axes[0][0].text(0.42,0.8,'Underprediction',size=20,fontstyle='italic')

# for (cluster_id, cluster_data), ax in zip(grouped_data, axes.flatten()):
#     ax.semilogx(cluster_data.columns, cluster_data.values.T,linewidth=1, color='gray',alpha=0.4)
#     ax.semilogx(cluster_data.columns, cluster_data.values.T.mean(axis=1),'b',linewidth=3)
#     ax.fill_between(cluster_data.columns,np.subtract(cluster_data.values.T.mean(axis=1),cluster_data.values.T.std(axis=1)),np.add(cluster_data.values.T.mean(axis=1),cluster_data.values.T.std(axis=1)),facecolor='gray', edgecolor='None', alpha=0.4)
#     ax.semilogx(oldclustermn.loc[cluster_id].index, oldclustermn.loc[cluster_id].values,'r',linewidth=3)
#     # ax.fill_between(oldclustermn.loc[cluster_id].index,np.subtract(oldclustermn.loc[cluster_id].values,oldclustersd.loc[cluster_id].values),np.add(oldclustermn.loc[cluster_id].values,oldclustersd.loc[cluster_id].values),facecolor=[1,0.8,0.8], edgecolor='None', alpha=0.6)
#     ax.semilogx([],[],label='Lee et al. (2022)',color='r',linewidth=3)
#     ax.semilogx([],[],label='This study',color='b',linewidth=3)
#     ax.set_xlim([0.01, 10])
#     ax.set_ylim([-1.5, 1.5])
#     ax.axhline(0,color='k',linestyle='--')
#     if i % 2 == 0:
#         ax.set_ylabel('${ a +  \delta S2S_s}$', size=20)
#     ax.text(0.011, 1.28, f"{cluster_id} (N = {len(cluster_data)})", size=17, fontweight='bold')
#     ax.tick_params(labelsize=16,direction='in', axis='both', which='both')
#     ax.grid(color='gray', linestyle='dashed', which='both', linewidth=0.4)
#     i = i+1
# leg = axes[0][0].legend(fontsize = 14,loc='best')
# leg.get_frame().set_edgecolor('k')
# for ax in axes[1]:
#     ax.set_xlabel('Vibration Period, T (s)', size=20)
# plt.savefig('OldVsNew subcategories.pdf')
# # plt.savefig('OldVsNew subcategories.png',dpi=300)
# #%% Standard deviation of the subcategories comparison
# grouped_data_dS2S = filtaNewdS2S.groupby(cluster_labels)
# grouped_data_errdS2S = filtaNewerrdS2S.groupby(cluster_labels)
# cluster_std = {}
# for label, group_errdS2S in grouped_data_errdS2S:
#     group_dS2S = grouped_data_dS2S.get_group(label)
#     std_cluster = np.sqrt(np.mean(group_dS2S**2, axis=0) + np.mean(group_errdS2S**2, axis=0))
#     cluster_std[label]=std_cluster
# cluster_std = pd.DataFrame(cluster_std)
# cluster_std = cluster_std.transpose()
# grouped_data_olddS2S = filtaOlddS2S.groupby(cluster_labels)
# grouped_data_olderrdS2S = filtaOlderrdS2S.groupby(cluster_labels)
# cluster_oldstd = {}
# for label, group_errdS2S in grouped_data_olderrdS2S:
#     group_dS2S = grouped_data_olddS2S.get_group(label)
#     std_cluster = np.sqrt(np.mean(group_dS2S**2, axis=0) + np.mean(group_errdS2S**2, axis=0))
#     cluster_oldstd[label]=std_cluster
# cluster_oldstd = pd.DataFrame(cluster_oldstd)
# cluster_oldstd = cluster_oldstd.transpose()
# fig1,axes1 = plt.subplots(figsize=(11.76,8.24),constrained_layout=True)
# colors = ['r','b','g','purple']
# axes1.semilogx(T,OldphiS2S,'orange',linewidth=3,label='All sites - Lee et al. (2022)')
# axes1.semilogx(T,NewphiS2S,'k',linewidth=3,label='All sites - This study')
# for i in range(len(cluster_std)):
#     # axes1.semilogx(cluster_oldstd.columns,cluster_oldstd.iloc[i,:],color=colors[i],linestyle='--')
#     axes1.semilogx(cluster_std.columns,cluster_std.iloc[i,:],color=colors[i],linestyle='-',label=cluster_std.index[i])

# axes1.set_xlabel('Vibration Period, T (s)', size=20)
# axes1.set_ylabel('Standard deviation of the subcategories', size=20)
# axes1.grid(color='gray', linestyle='dashed', which='both', linewidth=0.4)
# axes1.tick_params(labelsize=16,direction='in', axis='both', which='both')
# axes1.set_xlim([0.01, 10])
# axes1.legend(fontsize=14)
# fig1.savefig('Standard deviation subcategories comparison.png',dpi=300)
# #%% Geomorphic subcategory analysis
# df_categories = pd.read_csv(Path(r"C:\Users\ati47\OneDrive - University of Canterbury\Desktop\PhD\10. Research\Objective 1\1. Residuals\3. Vs30 sensitivity\Station_subcategory.csv"))
# df_categories = df_categories.sort_values(by='stat_name',ascending=True)
# OldStats['Oldstat_id'] = ''
# NewStats['Newstat_id']=  ''
# for i in range(1,383):
#     OldStats['Oldstat_id'][i] = 'Station_'+ str(OldStats.index.values[i-1])
    
# for j in range(1,385):
#     NewStats['Newstat_id'][j] = 'Station_'+ str(NewStats.index.values[j-1])
    
# df_categories = pd.merge(df_categories,OldStats[['stat_name','Oldstat_id']],on='stat_name')
# df_categories = pd.merge(df_categories,NewStats[['stat_name','Newstat_id']],on='stat_name')
# grouped_data  = df_categories.groupby('Geomorphology')
# oldstatids    = df_categories['Oldstat_id'].unique()
# newstatids    = df_categories['Newstat_id'].unique()
# filtaOlddS2S  = aOlddS2S[aOlddS2S.index.isin(oldstatids)]
# filtaNewdS2S  = aNewdS2S[aNewdS2S.index.isin(newstatids)]
# cluster_labels= np.array(df_categories['Geomorphology'])
# fig, axes = plt.subplots(2, 2, figsize=(9.89, 7.92),constrained_layout = True)
# grouped_data = filtaNewdS2S.groupby(cluster_labels)
# cluster_mean = filtaNewdS2S.groupby(cluster_labels).mean()
# cluster_std  = filtaNewdS2S.groupby(cluster_labels).std()
# oldclustermn = filtaOlddS2S.groupby(cluster_labels).mean()
# oldclustersd = filtaOlddS2S.groupby(cluster_labels).std()
# i =0 
# for (cluster_id, cluster_data), ax in zip(grouped_data, axes.flatten()):
#     ax.semilogx(cluster_data.columns, cluster_data.values.T,linewidth=1, color='gray')
#     ax.semilogx(cluster_data.columns, cluster_data.values.T.mean(axis=1),'k',linewidth=3)
#     ax.fill_between(cluster_data.columns,np.subtract(cluster_data.values.T.mean(axis=1),cluster_data.values.T.std(axis=1)),np.add(cluster_data.values.T.mean(axis=1),cluster_data.values.T.std(axis=1)),facecolor='gray', edgecolor='None', linestyle='dashed', linewidth=1.5, alpha=0.8)
#     ax.semilogx(oldclustermn.loc[cluster_id].index, oldclustermn.loc[cluster_id].values,color = 'orange',linewidth=3)
#     # ax.fill_between(oldclustermn.loc[cluster_id].index,np.subtract(oldclustermn.loc[cluster_id].values,oldclustersd.loc[cluster_id].values),np.add(oldclustermn.loc[cluster_id].values,oldclustersd.loc[cluster_id].values),facecolor=[1,0.8,0.8], edgecolor=None, linestyle='dashed', linewidth=1.5, alpha=0.6)
#     ax.semilogx([],[],label='Lee et al. (2022)',color='orange',linewidth=3)
#     ax.semilogx([],[],label='This study',color='k',linewidth=3)
#     ax.set_xlim([0.01, 10])
#     ax.set_ylim([-1.5, 1.5])
#     ax.axhline(0,color='maroon',linestyle='--')
#     if i % 2 == 0:
#         ax.set_ylabel('${ a +  \delta S2S_s}$', size=20)
#     ax.text(0.011,1.28,f"{cluster_id}",size =20,fontweight='bold')
#     # ax.text(1,1.3,f"Count = {len(cluster_data)}")
#     ax.tick_params(labelsize=16,direction='in', axis='both', which='both')
#     ax.grid(color='gray', linestyle='dashed', which='both', linewidth=0.4)
#     i = i+1
# leg = axes[0][0].legend(fontsize = 14,loc='lower left')
# leg.get_frame().set_edgecolor('k')
# for ax in axes[1]:
#     ax.set_xlabel('Vibration Period, T (s)', size=20)
# plt.savefig('Geomorphic station subcategories.png',dpi=300)

# #%% Geomorphic categories analysis - Osaka
# df_categories = pd.read_csv(Path(r"C:\Users\ati47\OneDrive - University of Canterbury\Desktop\PhD\10. Research\Objective 1\1. Residuals\3. Vs30 sensitivity\Station_geom.csv"))
# df_categories = df_categories.sort_values(by='stat_name',ascending=True)
# OldStats['Oldstat_id'] = ''
# NewStats['Newstat_id']=  ''
# for i in range(1,383):
#     OldStats['Oldstat_id'][i] = 'Station_'+ str(OldStats.index.values[i-1])
    
# for j in range(1,385):
#     NewStats['Newstat_id'][j] = 'Station_'+ str(NewStats.index.values[j-1])
    
# df_categories = pd.merge(df_categories,OldStats[['stat_name','Oldstat_id']],on='stat_name')
# df_categories = pd.merge(df_categories,NewStats[['stat_name','Newstat_id']],on='stat_name')
# grouped_data  = df_categories.groupby('Geomorphology')
# oldstatids    = df_categories['Oldstat_id'].unique()
# newstatids    = df_categories['Newstat_id'].unique()
# filtaOlddS2S  = aOlddS2S[aOlddS2S.index.isin(oldstatids)]
# filtaNewdS2S  = aNewdS2S[aNewdS2S.index.isin(newstatids)]
# filtaOlderrdS2S = OldS2Serr[OldS2Serr.index.isin(oldstatids)]
# filtaNewerrdS2S = NewS2Serr[NewS2Serr.index.isin(newstatids)]
# cluster_labels= np.array(df_categories['Geomorphology'])
# fig, axes = plt.subplots(2, 2, figsize=(9.89, 7.92),constrained_layout = True)
# desired_order = ["Basin","Basin-edge","Valley","Hill"]
# grouped_data = NewdS2S.groupby(cluster_labels)
# sorted_grouped_data = [grouped_data.get_group(cluster_id) for cluster_id in desired_order]
# cluster_mean = filtaNewdS2S.groupby(cluster_labels).mean()
# cluster_std  = filtaNewdS2S.groupby(cluster_labels).std()
# oldclustermn = filtaOlddS2S.groupby(cluster_labels).mean()
# oldclustersd = filtaOlddS2S.groupby(cluster_labels).std()
# i =0 
# colors = ['red','orange','blue','green']
# for cluster_id, cluster_data, ax in zip(desired_order,sorted_grouped_data, axes.flatten()):
#     ax.semilogx(cluster_data.columns, cluster_data.values.T,linewidth=1, color=colors[i])
#     ax.semilogx(cluster_data.columns, cluster_data.values.T.mean(axis=1),'k',linewidth=3)
#     ax.semilogx(cluster_data.columns, cluster_data.values.T.mean(axis=1)-cluster_data.values.T.std(axis=1),'k--',linewidth=2)
#     ax.semilogx(cluster_data.columns, cluster_data.values.T.mean(axis=1)+cluster_data.values.T.std(axis=1),'k--',linewidth=2)
#     # ax.fill_between(cluster_data.columns,np.subtract(cluster_data.values.T.mean(axis=1),cluster_data.values.T.std(axis=1)),np.add(cluster_data.values.T.mean(axis=1),cluster_data.values.T.std(axis=1)),facecolor=colors[i], edgecolor='None', linestyle='dashed', linewidth=1.5, alpha=0.4)
#     # ax.semilogx(oldclustermn.loc[cluster_id].index, oldclustermn.loc[cluster_id].values,'orange',linewidth=3)
#     # ax.fill_between(oldclustermn.loc[cluster_id].index,np.subtract(oldclustermn.loc[cluster_id].values,oldclustersd.loc[cluster_id].values),np.add(oldclustermn.loc[cluster_id].values,oldclustersd.loc[cluster_id].values),facecolor=[1,0.8,0.8], edgecolor=None, linestyle='dashed', linewidth=1.5, alpha=0.6)
#     # ax.semilogx([],[],label='Lee et al. (2022)',color='orange',linewidth=3)
#     # ax.semilogx([],[],label='This study',color='k',linewidth=3)
#     ax.set_xlim([0.01, 10])
#     ax.set_ylim([-1.5, 1.5])
#     ax.axhline(0,color='maroon',linestyle='--')
#     if i % 2 == 0:
#         ax.set_ylabel('${\delta S2S_s}$', size=20)
#     ax.text(0.011,1.28,f"{cluster_id}",size =20,fontweight='bold')
#     # ax.text(1,1.3,f"Count = {len(cluster_data)}")
#     ax.tick_params(labelsize=16,direction='in', axis='both', which='both')
#     ax.grid(color='gray', linestyle='dashed', which='both', linewidth=0.4)
#     i = i+1
# # leg = axes[0][0].legend(fontsize = 14,loc='lower left')
# # leg.get_frame().set_edgecolor('k')
# for ax in axes[1]:
#     ax.set_xlabel('Vibration Period, T (s)', size=20)
# # plt.savefig('Geomorphic station categories.png',dpi=300)
# #%% Standard deviation of the geomorphic categories comparison
# grouped_data_dS2S = NewdS2S.groupby(cluster_labels)
# grouped_data_errdS2S = NewS2Serr.groupby(cluster_labels)
# cluster_std = {}
# for label, group_errdS2S in grouped_data_errdS2S: 
#     group_dS2S = grouped_data_dS2S.get_group(label)
#     group_errdS2Shill = group_errdS2S
#     std_cluster = np.sqrt(np.mean(group_dS2S**2, axis=0) + np.mean(group_errdS2S**2, axis=0))
#     cluster_std[label]=std_cluster
# cluster_std = pd.DataFrame(cluster_std)
# cluster_std = cluster_std.transpose()
# # grouped_data_olddS2S = filtaOlddS2S.groupby(cluster_labels)
# # grouped_data_olderrdS2S = filtaOlderrdS2S.groupby(cluster_labels)
# # cluster_oldstd = {}
# # for label, group_errdS2S in grouped_data_olderrdS2S:
# #     group_dS2S = grouped_data_olddS2S.get_group(label)
# #     std_cluster = np.sqrt(np.mean(group_dS2S**2, axis=0) + np.mean(group_errdS2S**2, axis=0))
# #     cluster_oldstd[label]=std_cluster
# # cluster_oldstd = pd.DataFrame(cluster_oldstd)
# # cluster_oldstd = cluster_oldstd.transpose()
# fig1,axes1 = plt.subplots(figsize=(11.76,8.24),constrained_layout=True)
# colors = ['red','lime','purple','cyan']
# axes1.semilogx(T,OldphiS2S,'orange',linewidth=3,label='All sites - Lee et al. (2022)')
# axes1.semilogx(T,NewphiS2S,'k',linewidth=3,label='All sites - This study')
# # desired_order = ['Basin','Basin-edge','Valley','Hill']
# # cluster_std   = cluster_std.reindex(desired_order)
# # for i in range(len(cluster_std)):
# #     # axes1.semilogx(oldclustersd.columns,oldclustersd.iloc[i,:],color=colors[i],linestyle='--')
# #     axes1.semilogx(cluster_std.columns,cluster_std.iloc[i,:],color=colors[i],linestyle='-',label=cluster_std.index[i],linewidth=1)

# axes1.set_xlabel('Vibration Period, T (s)', size=25)
# axes1.set_ylabel('Total standard deviation', size=25)
# axes1.grid(color='gray', linestyle='dashed', which='both', linewidth=0.4)
# axes1.tick_params(labelsize=20,direction='in', axis='both', which='both')
# axes1.set_xlim([0.01, 10])
# axes1.set_ylim([0.0, 0.65])
# leg = axes1.legend(fontsize=14)
# leg.get_frame().set_edgecolor('k')
# # fig1.savefig('Standard deviation geomorphic categories comparison.png',dpi=300)
# #%% Bias standard deviation comparison for Q1, Q2, Q3 Vs30 sites
# df_categories = pd.read_csv(Path(r"C:\Users\ati47\OneDrive - University of Canterbury\Desktop\PhD\10. Research\Objective 1\1. Residuals\3. Vs30 sensitivity\Station_geom.csv"))
# df_categories = df_categories.sort_values(by='stat_name',ascending=True)
# OldStats['Oldstat_id'] = ''
# NewStats['Newstat_id']=  ''
# for i in range(1,383):
#     OldStats['Oldstat_id'][i] = 'Station_'+ str(OldStats.index.values[i-1])
    
# for j in range(1,385):
#     NewStats['Newstat_id'][j] = 'Station_'+ str(NewStats.index.values[j-1])
    
# df_categories = df_categories.dropna()
# df_categories = pd.merge(df_categories,OldStats[['stat_name','Oldstat_id']],on='stat_name')
# df_categories = pd.merge(df_categories,NewStats[['stat_name','Newstat_id']],on='stat_name')
# grouped_data  = df_categories.groupby('QVs30')
# oldstatids    = df_categories['Oldstat_id'].unique()
# newstatids    = df_categories['Newstat_id'].unique()
# filtaOlddS2S  = aOlddS2S[aOlddS2S.index.isin(oldstatids)]
# filtaNewdS2S  = aNewdS2S[aNewdS2S.index.isin(newstatids)]
# filtaOlderrdS2S = OldS2Serr[OldS2Serr.index.isin(oldstatids)]
# filtaNewerrdS2S = NewS2Serr[NewS2Serr.index.isin(newstatids)]
# cluster_labels= np.array(df_categories['QVs30'])
# fig, axes = plt.subplots(2, 2, figsize=(9.89, 7.92),constrained_layout = True)
# grouped_data = filtaNewdS2S.groupby(cluster_labels)
# cluster_mean = filtaNewdS2S.groupby(cluster_labels).mean()
# cluster_std  = filtaNewdS2S.groupby(cluster_labels).std()
# oldclustermn = filtaOlddS2S.groupby(cluster_labels).mean()
# oldclustersd = filtaOlddS2S.groupby(cluster_labels).std()
# grouped_data_dS2S = filtaNewdS2S.groupby(cluster_labels)
# grouped_data_errdS2S = filtaNewerrdS2S.groupby(cluster_labels)
# cluster_std = {}
# for label, group_errdS2S in grouped_data_errdS2S:
#     group_dS2S = grouped_data_dS2S.get_group(label)
#     std_cluster = np.sqrt(np.mean(group_dS2S**2, axis=0) + np.mean(group_errdS2S**2, axis=0))
#     cluster_std[label]=std_cluster
# cluster_std = pd.DataFrame(cluster_std)
# cluster_std = cluster_std.transpose()
# grouped_data_olddS2S = filtaOlddS2S.groupby(cluster_labels)
# grouped_data_olderrdS2S = filtaOlderrdS2S.groupby(cluster_labels)
# cluster_oldstd = {}
# for label, group_errdS2S in grouped_data_olderrdS2S:
#     group_dS2S = grouped_data_olddS2S.get_group(label)
#     std_cluster = np.sqrt(np.mean(group_dS2S**2, axis=0) + np.mean(group_errdS2S**2, axis=0))
#     cluster_oldstd[label]=std_cluster
# cluster_oldstd = pd.DataFrame(cluster_oldstd)
# cluster_oldstd = cluster_oldstd.transpose()
# i =0 


# for (cluster_id, cluster_data), ax in zip(grouped_data, axes.flatten()):
#     l = []
#     for index in cluster_data.index:
#         stat_name = NewStats[NewStats['Newstat_id'] == index]['stat_name'].values[0]
#         if len(stat_name) > 0:
#             l.append(stat_name)
#         else:
#             l.append(None)
#     ax.semilogx(cluster_data.columns, cluster_data.values.T,linewidth=1, color='gray',label=l, picker=True,pickradius=5)
#     ax.semilogx(cluster_data.columns, cluster_data.values.T.mean(axis=1),'k',linewidth=3)
#     ax.fill_between(cluster_data.columns,np.subtract(cluster_data.values.T.mean(axis=1),cluster_data.values.T.std(axis=1)),np.add(cluster_data.values.T.mean(axis=1),cluster_data.values.T.std(axis=1)),facecolor='gray', edgecolor='None', linestyle='dashed', linewidth=1.5, alpha=0.8)
#     ax.semilogx(oldclustermn.loc[cluster_id].index, oldclustermn.loc[cluster_id].values,color = 'orange',linewidth=3)
#     # ax.fill_between(oldclustermn.loc[cluster_id].index,np.subtract(oldclustermn.loc[cluster_id].values,oldclustersd.loc[cluster_id].values),np.add(oldclustermn.loc[cluster_id].values,oldclustersd.loc[cluster_id].values),facecolor=[1,0.8,0.8], edgecolor=None, linestyle='dashed', linewidth=1.5, alpha=0.6)
#     ax.semilogx([],[],label='Lee et al. (2022)',color='orange',linewidth=3)
#     ax.semilogx([],[],label='This study',color='k',linewidth=3)
#     ax.set_xlim([0.01, 10])
#     ax.set_ylim([-1.5, 1.5])
#     ax.axhline(0,color='maroon',linestyle='--')
#     if i % 2 == 0:
#         ax.set_ylabel('${ a +  \delta S2S_s}$', size=20)
#     ax.text(0.011,1.28,f"{cluster_id}",size =20,fontweight='bold')
#     ax.text(1,1.3,f"Count = {len(cluster_data)}")
#     ax.tick_params(labelsize=16,direction='in', axis='both', which='both')
#     ax.grid(color='gray', linestyle='dashed', which='both', linewidth=0.4)
#     i = i+1
# # leg = axes[0][0].legend(fontsize = 14,loc='lower left')
# # leg.get_frame().set_edgecolor('k')


# for ax in axes[1]:
#     ax.set_xlabel('Vibration Period, T (s)', size=20)
    
# for i in range(len(grouped_data),2*2):
#     fig.delaxes(axes.flatten()[i]) 
    
# fig.canvas.mpl_connect("pick_event", onpick)
# cursor       = mpl.cursor(multiple=True)
# cursor.connect("add",on_add)

# fig.savefig('Vs30 Quality bias comparison.png',dpi=300)
# fig1,axes1 = plt.subplots(figsize=(11.76,8.24),constrained_layout=True)
# colors = ['g','b','r']
# axes1.semilogx(T,OldphiS2S,'orange',linewidth=3,label='All sites - Lee et al. (2022)')
# axes1.semilogx(T,NewphiS2S,'k',linewidth=3,label='All sites - This study')
# for i in range(len(cluster_std)):
#     # axes1.semilogx(oldclustersd.columns,oldclustersd.iloc[i,:],color=colors[i],linestyle='--')
#     axes1.semilogx(cluster_std.columns,cluster_std.iloc[i,:],color=colors[i],linestyle='-',label=cluster_std.index[i])

# axes1.set_xlabel('Vibration Period, T (s)', size=20)
# axes1.set_ylabel('Standard deviation of Vs30 categories', size=20)
# axes1.grid(color='gray', linestyle='dashed', which='both', linewidth=0.4)
# axes1.tick_params(labelsize=16,direction='in', axis='both', which='both')
# axes1.set_xlim([0.01, 10])
# leg = axes1.legend(fontsize=14)
# leg.get_frame().set_edgecolor('k')
# fig1.savefig('Standard deviation Vs30 categories comparison - This study.png',dpi=300)
# #%% Bias standard deviation comparison for Type 1, Type 2, Type 4 sites, Unmodelled, Non-Basin sites
# df_categories = pd.read_csv(Path(r"C:\Users\ati47\OneDrive - University of Canterbury\Desktop\PhD\10. Research\Objective 1\1. Residuals\3. Vs30 sensitivity\Station_geom.csv"))
# df_categories = df_categories.sort_values(by='stat_name',ascending=True)
# OldStats['Oldstat_id'] = ''
# NewStats['Newstat_id']=  ''
# for i in range(1,383):
#     OldStats['Oldstat_id'][i] = 'Station_'+ str(OldStats.index.values[i-1])
    
# for j in range(1,385):
#     NewStats['Newstat_id'][j] = 'Station_'+ str(NewStats.index.values[j-1])
    
# df_categories = pd.merge(df_categories,OldStats[['stat_name','Oldstat_id']],on='stat_name')
# df_categories = pd.merge(df_categories,NewStats[['stat_name','Newstat_id']],on='stat_name')
# grouped_data  = df_categories.groupby('Basin Type')
# oldstatids    = df_categories['Oldstat_id'].unique()
# newstatids    = df_categories['Newstat_id'].unique()
# filtaOlddS2S  = aOlddS2S[aOlddS2S.index.isin(oldstatids)]
# filtaNewdS2S  = aNewdS2S[aNewdS2S.index.isin(newstatids)]
# filtaOlderrdS2S = OldS2Serr[OldS2Serr.index.isin(oldstatids)]
# filtaNewerrdS2S = NewS2Serr[NewS2Serr.index.isin(newstatids)]
# cluster_labels= np.array(df_categories['Basin Type'])
# fig, axes = plt.subplots(2, 3, figsize=(17.875, 9.3),constrained_layout = True)
# grouped_data = filtaNewdS2S.groupby(cluster_labels)
# cluster_mean = filtaNewdS2S.groupby(cluster_labels).mean()
# cluster_std  = filtaNewdS2S.groupby(cluster_labels).std()
# oldclustermn = filtaOlddS2S.groupby(cluster_labels).mean()
# oldclustersd = filtaOlddS2S.groupby(cluster_labels).std()
# grouped_data_dS2S = filtaNewdS2S.groupby(cluster_labels)
# grouped_data_errdS2S = filtaNewerrdS2S.groupby(cluster_labels)
# cluster_std = {}
# for label, group_errdS2S in grouped_data_errdS2S:
#     group_dS2S = grouped_data_dS2S.get_group(label)
#     std_cluster = np.sqrt(np.mean(group_dS2S**2, axis=0) + np.mean(group_errdS2S**2, axis=0))
#     cluster_std[label]=std_cluster
# cluster_std = pd.DataFrame(cluster_std)
# cluster_std = cluster_std.transpose()
# grouped_data_olddS2S = filtaOlddS2S.groupby(cluster_labels)
# grouped_data_olderrdS2S = filtaOlderrdS2S.groupby(cluster_labels)
# cluster_oldstd = {}
# for label, group_errdS2S in grouped_data_olderrdS2S:
#     group_dS2S = grouped_data_olddS2S.get_group(label)
#     std_cluster = np.sqrt(np.mean(group_dS2S**2, axis=0) + np.mean(group_errdS2S**2, axis=0))
#     cluster_oldstd[label]=std_cluster
# cluster_oldstd = pd.DataFrame(cluster_oldstd)
# cluster_oldstd = cluster_oldstd.transpose()
# i =0 


# for (cluster_id, cluster_data), ax in zip(grouped_data, axes.flatten()):
#     l = []
#     for index in cluster_data.index:
#         stat_name = NewStats[NewStats['Newstat_id'] == index]['stat_name'].values[0]
#         if len(stat_name) > 0:
#             l.append(stat_name)
#         else:
#             l.append(None)
#     ax.semilogx(cluster_data.columns, cluster_data.values.T,linewidth=1, color='gray',label=l, picker=True,pickradius=5)
#     ax.semilogx(cluster_data.columns, cluster_data.values.T.mean(axis=1),'k',linewidth=3)
#     ax.fill_between(cluster_data.columns,np.subtract(cluster_data.values.T.mean(axis=1),cluster_data.values.T.std(axis=1)),np.add(cluster_data.values.T.mean(axis=1),cluster_data.values.T.std(axis=1)),facecolor='gray', edgecolor='None', linestyle='dashed', linewidth=1.5, alpha=0.8)
#     ax.semilogx(oldclustermn.loc[cluster_id].index, oldclustermn.loc[cluster_id].values,color = 'orange',linewidth=3)
#     # ax.fill_between(oldclustermn.loc[cluster_id].index,np.subtract(oldclustermn.loc[cluster_id].values,oldclustersd.loc[cluster_id].values),np.add(oldclustermn.loc[cluster_id].values,oldclustersd.loc[cluster_id].values),facecolor=[1,0.8,0.8], edgecolor=None, linestyle='dashed', linewidth=1.5, alpha=0.6)
#     ax.semilogx([],[],label='Lee et al. (2022)',color='orange',linewidth=3)
#     ax.semilogx([],[],label='This study',color='k',linewidth=3)
#     ax.set_xlim([0.01, 10])
#     ax.set_ylim([-1.5, 1.5])
#     ax.axhline(0,color='maroon',linestyle='--')
#     if i % 3 == 0:
#         ax.set_ylabel('${ a +  \delta S2S_s}$', size=20)
#     ax.text(0.011,1.28,f"{cluster_id}",size =20,fontweight='bold')
#     ax.text(1,1.3,f"Count = {len(cluster_data)}")
#     ax.tick_params(labelsize=16,direction='in', axis='both', which='both')
#     ax.grid(color='gray', linestyle='dashed', which='both', linewidth=0.4)
#     i = i+1
# # leg = axes[0][0].legend(fontsize = 14,loc='lower left')
# # leg.get_frame().set_edgecolor('k')


# for ax in axes[1]:
#     ax.set_xlabel('Vibration Period, T (s)', size=20)
    
# for i in range(len(grouped_data),2*3):
#     fig.delaxes(axes.flatten()[i]) 
    
# fig.canvas.mpl_connect("pick_event", onpick)
# cursor       = mpl.cursor(multiple=True)
# cursor.connect("add",on_add)

# fig.savefig('Basin Quality bias comparison.png',dpi=300)
# fig1,axes1 = plt.subplots(figsize=(11.76,8.24),constrained_layout=True)
# colors = ['cyan','b','g','purple','red']
# axes1.semilogx(T,OldphiS2S,'orange',linewidth=3,label='All sites - Lee et al. (2022)')
# axes1.semilogx(T,NewphiS2S,'k',linewidth=3,label='All sites - This study')
# for i in range(len(cluster_std)):
#     # axes1.semilogx(oldclustersd.columns,oldclustersd.iloc[i,:],color=colors[i],linestyle='--')
#     axes1.semilogx(cluster_std.columns,cluster_std.iloc[i,:],color=colors[i],linestyle='-',label=cluster_std.index[i])

# axes1.set_xlabel('Vibration Period, T (s)', size=20)
# axes1.set_ylabel('Standard deviation of basin quality categories', size=20)
# axes1.grid(color='gray', linestyle='dashed', which='both', linewidth=0.4)
# axes1.tick_params(labelsize=16,direction='in', axis='both', which='both')
# axes1.set_xlim([0.01, 10])
# leg = axes1.legend(fontsize=14)
# leg.get_frame().set_edgecolor('k')
# fig1.savefig('Standard deviation Basin categories comparison.png',dpi=300)
# #%% FS and RS on same graph
# FSOlddS2SPath = Path(r"C:\Users\ati47\OneDrive - University of Canterbury\Desktop\PhD\10. Research\Objective 1\1. Residuals\superceded\1. Initial Input\FAS Hanning\PJsreStationBiased_sim.csv")
# FSNewdS2SPath = Path(r"C:\Users\ati47\OneDrive - University of Canterbury\Desktop\PhD\10. Research\Objective 1\1. Residuals\1. Calculations\Simulation residuals\FAS Hanning\PJsreStationBiased_sim.csv")
# FSOldStats    = pd.read_csv(Path(r"C:\Users\ati47\OneDrive - University of Canterbury\Desktop\PhD\10. Research\Objective 1\1. Residuals\superceded\1. Initial Input\stations.csv"),index_col='stat_id')
# FSNewStats    = pd.read_csv(Path(r"C:\Users\ati47\OneDrive - University of Canterbury\Desktop\PhD\10. Research\Objective 1\1. Residuals\1. Calculations\Simulation residuals\stations.csv"),index_col='stat_id')
# FSOlddS2S     = load_res_pSA_from_csvFS(FSOlddS2SPath)
# FSNewdS2S     = load_res_pSA_from_csvFS(FSNewdS2SPath)
# FSOld = pd.read_csv(Path(r"C:\Users\ati47\OneDrive - University of Canterbury\Desktop\PhD\10. Research\Objective 1\1. Residuals\superceded\1. Initial Input\FAS\PJsvarCompsBiased_sim.csv"))
# FSNew = pd.read_csv(Path(r"C:\Users\ati47\OneDrive - University of Canterbury\Desktop\PhD\10. Research\Objective 1\1. Residuals\1. Calculations\Simulation residuals\FAS\PJsvarCompsBiased_sim.csv"))
# FST   = FSOld.iloc[0:,1]
# f     = 1/FST
# FSOldbias = FSOld.iloc[0:,2]
# FSNewbias = FSNew.iloc[0:,2]
# FSOldsigma = FSOld.iloc[0:,6]
# FSNewsigma = FSNew.iloc[0:,6]
# FSOldtau = FSOld.iloc[0:,3]
# FSNewtau = FSNew.iloc[0:,3]
# FSOldphiS2S = FSOld.iloc[0:,4]
# FSNewphiS2S = FSNew.iloc[0:,4]
# FSOldphiSS = FSOld.iloc[0:,5]
# FSNewphiSS = FSNew.iloc[0:,5]
# FSOldbias.index = FSOlddS2S.T.index
# FSNewbias.index = FSNewdS2S.T.index
# FSaOlddS2S = FSOlddS2S.T.add(Oldbias,axis=0)
# FSaOlddS2S = FSaOlddS2S.T
# FSaNewdS2S = FSNewdS2S.T.add(Newbias,axis=0)
# FSaNewdS2S = FSaNewdS2S.T
# fig1,ax1 = plt.subplots(2,3,figsize=(19.2,9.83),constrained_layout=True)
# ax1[0,0].semilogx(T,Oldbias,'r',label = 'Lee et al. (2022) RS')
# ax1[0,0].semilogx(T,Newbias,'b',label = 'This study RS')
# ax1[0,0].semilogx(f,FSOldbias,'r--',label = 'Lee et al. (2022) FS')
# ax1[0,0].semilogx(f,FSNewbias,'b--',label = 'This study FS')
# ax1[0,0].axhline(0,c='gray',linestyle='--')
# ax1[0,0].axvline(1,c='gray',linestyle='--')
# ax1[0,0].set_ylim([-1,2.5])
# ax1[0,0].set_xlim([0.01,10])
# ax1[0,0].set_ylabel('Model prediction bias, a',size=20)
# ax1[0,0].set_xlabel('Vibration period, T (s)',size=20)
# leg = ax1[0,0].legend(fontsize=18,loc='upper right')
# leg.get_frame().set_edgecolor('k')
# ax1[0,0].text(0.11,-0.8,'Overprediction',size=20)
# ax1[0,0].text(0.11,0.8,'Underprediction',size=20)
# ax1[0,0].tick_params(labelsize=16,direction='in', axis='both', which='both')
# ax1[0,0].grid(color='gray', linestyle='dashed', which='both', linewidth=0.4)

# ax1[0,1].semilogx(T,Oldsigma,'r',label = 'Lee et al. (2022) RS')
# ax1[0,1].semilogx(T,Newsigma,'b',label = 'This study RS')
# ax1[0,1].semilogx(f,FSOldsigma,'r--',label = 'Lee et al. (2022) FS')
# ax1[0,1].semilogx(f,FSNewsigma,'b--',label = 'This study FS')
# ax1[0,1].set_ylim([0,1.5])
# ax1[0,1].set_xlim([0.01,10])
# ax1[0,1].axvline(1,c='gray',linestyle='--')
# ax1[0,1].set_ylabel('Total standard deviation, $\sigma$',size=20)
# ax1[0,1].set_xlabel('Vibration period, T (s)',size=20)
# ax1[0,1].tick_params(labelsize=16,direction='in', axis='both', which='both')
# ax1[0,1].grid(color='gray', linestyle='dashed', which='both', linewidth=0.4)

# ax1[1,0].semilogx(T,Oldtau,'r',label = 'Lee et al. (2022) RS')
# ax1[1,0].semilogx(T,Newtau,'b',label = 'This study RS')
# ax1[1,0].semilogx(f,FSOldtau,'r--',label = 'Lee et al. (2022) FS')
# ax1[1,0].semilogx(f,FSNewtau,'b--',label = 'This study FS')
# ax1[1,0].set_ylim([0,0.5])
# ax1[1,0].set_xlim([0.01,10])
# ax1[1,0].axvline(1,c='gray',linestyle='--')
# ax1[1,0].set_ylabel(r'Between-event standard deviation, $\tau$',size=20)
# ax1[1,0].set_xlabel('Vibration period, T (s)',size=20)
# ax1[1,0].tick_params(labelsize=16,direction='in', axis='both', which='both')
# ax1[1,0].grid(color='gray', linestyle='dashed', which='both', linewidth=0.4)
# leg = ax1[1,0].legend(fontsize=18,loc='lower left')
# leg.get_frame().set_edgecolor('k')

# ax1[1,1].semilogx(T,OldphiS2S,'r',label = 'Lee et al. (2022) RS')
# ax1[1,1].semilogx(T,NewphiS2S,'b',label = 'This study RS')
# ax1[1,1].semilogx(f,FSOldphiS2S,'r--',label = 'Lee et al. (2022) FS')
# ax1[1,1].semilogx(f,FSNewphiS2S,'b--',label = 'This study FS')
# ax1[1,1].set_ylim([0,1.2])
# ax1[1,1].set_xlim([0.01,10])
# ax1[1,1].axvline(1,c='gray',linestyle='--')
# ax1[1,1].set_ylabel('Site-to-site standard deviation, $\phi_{S2S}$',size=20)
# ax1[1,1].set_xlabel('Vibration period, T (s)',size=20)
# ax1[1,1].tick_params(labelsize=16,direction='in', axis='both', which='both')
# ax1[1,1].grid(color='gray', linestyle='dashed', which='both', linewidth=0.4)

# ax1[1,2].semilogx(T,OldphiSS,'r',label = 'Lee et al. (2022) RS')
# ax1[1,2].semilogx(T,NewphiSS,'b',label = 'This study RS')
# ax1[1,2].semilogx(f,FSOldphiSS,'r--',label = 'Lee et al. (2022) FS')
# ax1[1,2].semilogx(f,FSNewphiSS,'b--',label = 'This study FS')
# ax1[1,2].set_ylim([0,1.2])
# ax1[1,2].set_xlim([0.01,10])
# ax1[1,2].axvline(1,c='gray',linestyle='--')
# ax1[1,2].set_ylabel('Single station standard deviation, $\phi_{SS}$',size=20)
# ax1[1,2].set_xlabel('Vibration period, T (s)',size=20)
# ax1[1,2].tick_params(labelsize=16,direction='in', axis='both', which='both')
# ax1[1,2].grid(color='gray', linestyle='dashed', which='both', linewidth=0.4)

# fig1.delaxes(ax1[0,2])
# fig1.savefig('Old Vs New bias & std RS & FS Hanning.png')
#%%