# NZ-Wide Sedimentary Basin Models: Development and Validation

This repository contains the intensity measures, residuals, scripts, and electronic supplements accompanying the paper:
**Development and validation of New Zealand-wide sedimentary basin models for broadband ground-motion simulation**  
Ayushi Tiwari, Brendon Bradley, Robin Lee, and Felipe Kuncar  
*Earthquake Spectra* - manuscript submitted for publication.

The NZVM version 2.09 is released with this article. The velocity model code and data are archived separately on Zenodo (code: https://doi.org/10.5281/zenodo.19228321; data: https://doi.org/10.5281/zenodo.19228330).

The repository is mainly intended as a companion archive for the paper. It brings together the files used to describe the basin models, compare simulations with and without basin structure, and reproduce the main figures.

# Repository Overview

The repository is organised by the workflow used in the paper, from the IM databases through residual calculation to the figures and supplements.

The repository includes:

- Electronic Supplements B and C in CSV format, along with the basin depth maps

- Observed intensity measures (IMs)

- Simulated IMs from NZVM 2.09, with and without explicit basin representation

- Residuals from Linear Mixed-Effects regression (MERA) for both basin cases

- Scripts for figure generation

## Repository structure

```text
NZWide_SedimentaryBasinModels_DevelopValidation/
|
|-- Electronic_Supplements/
|   |-- ElectronicSupplementB_TableB.1_BasinSurface_Information.csv
|   |-- ElectronicSupplementC_TableC.1_Event_Information.csv
|   |-- ElectronicSupplementC_TableC.2_Station_Information.csv
|   |-- FigureB.1_Basindepthplots/
|       |-- tier_500m/
|       |-- tier_1000m/
|       |-- tier_2000m/
|       |-- tier_3000m/
|       |-- tier_4000m/
|
|-- IMs/
|   |-- Events.csv
|   |-- Stations.csv
|   |-- Observed_IMs.csv
|   |-- Simulated_IMs_Nobasin.csv
|   |-- Simulated_IMs_Withbasin.csv
|
|-- Residuals/
|   |-- Nobasin/
|   |-- Withbasin/
|
|-- Scripts_for_figures/
|   |-- Figure_1.py
|   |-- Figure_2.py
|   |-- Figure_3.py
|   |-- Figure_6.py
|   |-- Figure_7.py
|   |-- Figure_8.py
|   |-- Figure_9.py
|   |-- Figure_10_12_13_14_15_16_17_18.py
|   |-- Figure_11_abde.py
|   |-- Figure_11_cf.py
|
|-- LICENSE
|-- README.md
```

# Folder descriptions

## 1. Electronic_Supplements/

Contains Electronic Supplements B and C of the paper in CSV format, plus the basin depth maps.

- `ElectronicSupplementB_TableB.1_BasinSurface_Information.csv`: the 45 sedimentary basins represented in NZVM 2.09, with basin type, the surfaces used, and modelling notes.
- `ElectronicSupplementC_TableC.1_Event_Information.csv`: the 73 events used in the analyses, with location, MW, depth, and focal mechanism (strike, dip, rake).
- `ElectronicSupplementC_TableC.2_Station_Information.csv`: the 300 stations used, with location, Vs30, basin and geomorphology classification, and Z1.0 depths from NZVM 2.09 and the GTL.
- `FigureB.1_Basindepthplots/`: basin depth maps, grouped into subfolders by depth tier (`tier_500m` through `tier_4000m`). Each tier folder contains one map per basin plus a shared colour bar for that tier.

## 2. IMs/

Contains the observed and simulated intensity measures (IMs) used in the analyses. Each IM file has one row per ground motion, keyed by `gm_id`, `event_id`, and `stat_id`, followed by the scalar IMs (PGA, PGV, CAV, AI, Ds575, Ds595) and then pSA and EAS across a range of periods and frequencies.

- `Observed_IMs.csv`: IMs for the 3743 observed ground motions.
- `Simulated_IMs_Nobasin.csv`: simulated IMs with implicit basin representation.
- `Simulated_IMs_Withbasin.csv`: simulated IMs with explicit basin representation.
- `Events.csv` and `Stations.csv`: metadata mapping `event_id` to the event name and `stat_id` to the station code.


## 3. Residuals/

Residuals from the Linear Mixed-Effects regression (MERA), computed separately for each basin case. The two subfolders, `Withbasin/` and `Nobasin/`, contain the same set of files:

- `bias_std_df.csv`: bias and standard-deviation components for all 210 IMs. The columns are the bias term (`bias`) and its standard error (`bias_std_err`), the between-event standard deviation (`tau`), the between-site (site-to-site) standard deviation (`phi_S2S`), the remaining within-event residual standard deviation (`phi_w`), and the total standard deviation (`sigma`, the root-sum-square of `tau`, `phi_S2S`, and `phi_w`).
- `event_res_df.csv`: between-event residuals for the 73 events.
- `event_cond_std_df.csv`: uncertainty in the point estimate of the between-event residuals (conditional standard deviation).
- `site_res_df.csv`: site-to-site residuals for the 300 stations.
- `site_cond_std_df.csv`: uncertainty in the point estimate of the site-to-site residuals (conditional standard deviation).
- `rem_res_df.csv`: remaining within-event, within-site residuals for each ground motion.
- `fit_df.csv`: fitted values from the regression for each ground motion.

## 4. Scripts_for_figures/

Contains the scripts used to generate the figures in the paper. Filenames correspond to figure numbers (e.g. `Figure_2.py`; `Figure_10_12_13_14_15_16_17_18.py` produces several related figures). The scripts use Python with `pandas` and `numpy`, `pygmt` and `geopandas` for maps, and `matplotlib`.

Note: These scripts are provided for reference and may require adjustments:

- Filenames of input data may differ from those in this repository.

- Some scripts may be incomplete or contain extra code unrelated to figure generation.

- Ensure the required data files are present before running the scripts.

## Citation

If you use this repository, please cite the accompanying paper:

```bibtex
@article{tiwari2026nzwidebasins,
  title = {Development and validation of New Zealand-wide sedimentary basin models for broadband ground-motion simulation},
  author = {Tiwari, Ayushi and Bradley, Brendon and Lee, Robin and Kuncar, Felipe},
  journal = {Earthquake Spectra},
  year = {2026},
  note = {Manuscript submitted for publication}
}
```

The citation can be updated once the final publication details and DOI are available.

## License

This repository is released under the MIT License. See `LICENSE` for details.