# OnSSET climate extension

An extension of the Open Source Spatial Electrification Tool (OnSSET) that adds
climate-risk-informed prioritization to settlement electrification timing.
The current version covers two hazards: heatwaves and droughts.

## What this is

OnSSET is a bottom-up, GIS-based, least-cost electrification model. Given a
country's population settlements and geospatial inputs, it identifies the
least-cost mix of grid extension, mini-grids, and stand-alone systems needed to
reach an electrification target by a future year.

This repository is a **fork of OnSSET** that introduces an optional climate
pre-processing stage and one new prioritization mode. Under that mode, when
OnSSET decides which unserved settlements to electrify in each time-step, it
orders candidates by a combined hazard-and-vulnerability score so that more
climate-exposed and more vulnerable settlements are electrified earlier in the
sequence.

Upstream OnSSET: <https://github.com/OnSSET/onsset>.

## What the climate extension adds

- A climate pre-processing pipeline that loads heatwave and drought data,
  aggregates it to regions (e.g. admin-2 or admin-3), and attaches per-settlement
  risk columns to the OnSSET DataFrame.
  Source: [`onsset/climate_algorithm.py`](onsset/climate_algorithm.py) and
  [`onsset/climate_calculations/`](onsset/climate_calculations/)
  (`heatwave_calculation.py`, `drought_calculation.py`).
- A new prioritization choice (`prio_choice = 6`) inside
  `SettlementProcessor.pre_selection`
  ([`onsset/onsset.py`](onsset/onsset.py)) that sorts unserved settlements by
  `ClimatePriority = ClimateHazard × ClimateVulnerability` (descending) before
  the rollout target is applied.
- A third menu option (3 = run scenario(s) with climate) in
  [`onsset/gui_runner.py`](onsset/gui_runner.py) that prompts for a climate
  data folder and an admin-3 shapefile.
- Optional `ClimateData` sheet in the specs Excel file for overriding
  thresholds, weights, the SPI baseline window, and input column names.

## What it does not change

The extension affects **timing and sequencing only**. It does not change:

- LCOE calculations for grid, mini-grid, or stand-alone systems.
- Technology choice (the least-cost technology is still selected per
  settlement based on LCOE, though the timing might affect e.g. 
  whether grid extension is possible).
- Grid-extension network logic.
- Population/demand projection.
- Investment, capacity, or emissions accounting.

When the climate inputs are not supplied, `runner.scenario` runs upstream
OnSSET unchanged. If `prio_choice = 6` is selected but the `ClimatePriority`
column is absent, the code logs a warning and falls back to
population-based ordering.

## Installation

Installation follows the upstream OnSSET workflow. A conda environment is
recommended because the climate pipeline depends on geospatial libraries
(`geopandas`, `shapely`, `fiona`, `pyogrio`, `gdal`, `scipy`).

```
conda env create -n OnSSET -f onsset_env.yml
conda activate OnSSET
```

The conda environment is defined in [`onsset_env.yml`](onsset_env.yml)
(Python 3.12). To launch the notebooks:

```
jupyter notebook
```

## Quick start: a climate-informed scenario

The repository ships three notebooks at the root:

1. [`1. OnSSET_Calibration.ipynb`](1.%20OnSSET_Calibration.ipynb) — calibrates
   the base-year settlement file. This step is identical to upstream OnSSET.
2. [`2. OnSSET_Scenarios.ipynb`](2.%20OnSSET_Scenarios.ipynb) — runs a
   single-period scenario.
3. [`2. OnSSET_Scenarios_MultipleTimeSteps.ipynb`](2.%20OnSSET_Scenarios_MultipleTimeSteps.ipynb)
   — runs a multi-period scenario.

To run a climate-informed scenario:

1. Calibrate as usual.
2. In the specs Excel file, set `PrioritizationAlgorithm = 6` in the
   `ScenarioParameters` sheet for the scenario row.
3. Either run the climate-aware GUI:

   ```
   python -m onsset.gui_runner
   ```

   and choose option `3` when prompted, or call
   `onsset.runner.scenario(...)` directly with the two new keyword arguments:

   ```python
   from onsset.runner import scenario
   scenario(
       specs_path,
       calibrated_csv_path,
       results_folder,
       summary_folder,
       pv_path,
       wind_path,
       mv_path,
       climate_folder="path/to/climate_csvs",
       admin3_shapefile="path/to/gadm41_XXX_3.shp",
   )
   ```

When both `climate_folder` and `admin3_shapefile` are supplied,
[`runner.scenario`](onsset/runner.py) calls `process_climate_data` once,
before the per-year loop, and the per-year loop then uses `prio_choice = 6` if
that scenario row is configured for it.

## Required climate inputs

In addition to the standard OnSSET inputs (calibrated settlements CSV, specs
Excel, hourly PV and wind capacity-factor CSVs, MV-line shapefile), climate
mode needs:

**A folder of climate CSV or Excel files.** The loader classifies files by
filename. Patterns recognised today:

| Hazard   | Temporal | Filename hints                                           | Required column |
|----------|----------|----------------------------------------------------------|-----------------|
| Heatwave | daily    | `t2m`, `temp`, `tmax`, `daily`, or `t2m*max*`            | `t2m_max_C`     |
| Drought  | monthly  | `tp_`, `precip`, `precipitation`, `rainfall`, `monthly`  | `tp_mm_month`   |

All files must also contain `latitude`, `longitude`, and `date` columns
(fallback names `lat`/`lon`/`x`/`y`/`x_deg`/`y_deg`/`timestamp`/`time`/`datetime`
are also recognised). Column names can be overridden in the `ClimateData`
sheet of the specs Excel file.

**An admin-3 (municipality) shapefile.** Default attribute columns: `GID_3`
(region ID) and `NAME_3` (region name). These names are configurable through
`Admin3IDColumn` and `Admin3NameColumn` in the `ClimateData` sheet.

**Optional: a `ClimateData` sheet in the specs Excel file.** Recognised
parameters and their defaults:

- Heatwave: `HeatwaveThresholdC` (32.0), `HeatwaveRiskWeight` (0.5),
  `TemperatureColumnName` (`t2m_max_C`).
- Drought: `SPIScale` (3), `SPIBaselineStartYear` (1950),
  `SPIBaselineEndYear` (2000), `SPIDroughtThreshold` (−1.0),
  `DroughtRiskWeight` (0.5), `PrecipitationColumnName` (`tp_mm_month`).
- Common: `LatitudeColumnName`, `LongitudeColumnName`, `DateColumnName`,
  `Admin3IDColumn`, `Admin3NameColumn`.

Vulnerability is computed from columns already present in the calibrated
settlements CSV: `NormalizedRelativeWealth` and `NormalizedTravelHours`
(min-max-normalised wealth index and travel-time-to-city, both in `[0, 1]`).
If either is missing, vulnerability falls back to a neutral value of 0.5.

## Outputs added by the extension

When climate mode runs, the per-scenario settlements CSV gains the following
columns (in addition to all standard OnSSET output columns):

| Column                     | Description                                                                 |
|----------------------------|-----------------------------------------------------------------------------|
| `Admin3ID`                 | Admin-3 region the settlement falls within.                                 |
| `ClimateRiskHeatwave`      | Per-region heatwave hazard score in `[0, 1]`.                               |
| `ClimateRiskDrought`       | Per-region drought hazard score in `[0, 1]`.                                |
| `ClimateHazard`            | Compound hazard (weighted average of heatwave and drought), in `[0, 1]`.    |
| `NormalizedClimateHazard`  | Same as `ClimateHazard`, kept for backward compatibility.                   |
| `ClimateVulnerability`     | `((1 − NormalizedRelativeWealth) + NormalizedTravelHours) / 2`, in `[0, 1]`. |
| `ClimatePriority`          | `ClimateHazard × ClimateVulnerability` — the score used for sequencing.     |

`pre_selection` additionally writes `NormalizedVulnerabilityScore`, which
holds the same values as `ClimatePriority` for that time-step.

Settlements outside the admin-3 shapefile or outside the climate grid receive
`0` for all climate columns.

Standard OnSSET outputs (per-time-step `FinalElecCode{year}`,
`MinimumOverallLCOE{year}`, `NewConnections{year}`, `NewCapacity{year}`,
`InvestmentCost{year}`, the summary CSV, and the per-year MV-line GeoJSON)
are unchanged in format and are written by [`onsset/runner.py`](onsset/runner.py).

## Minimal example workflow

```python
from onsset.runner import calibration, scenario

# 1. Calibrate (unchanged from upstream OnSSET)
calibration(
    specs_path="OnSSET v.2.0 -- non-GIS modelling parameters.xlsx",
    csv_path="path/to/settlements_gis.csv",
    specs_path_calib="specs_calib.xlsx",
    calibrated_csv_path="settlements_calibrated.csv",
)

# 2. Run a climate-informed scenario (set PrioritizationAlgorithm = 6
#    in the ScenarioParameters sheet of specs_calib.xlsx)
scenario(
    specs_path="specs_calib.xlsx",
    calibrated_csv_path="settlements_calibrated.csv",
    results_folder="results",
    summary_folder="summaries",
    pv_path="pv_hourly.csv",
    wind_path="wind_hourly.csv",
    mv_path="mv_lines.shp",
    climate_folder="climate_csvs",
    admin3_shapefile="gadm41_XXX_3.shp",
)
```

To compare against a baseline (non-climate) ordering, re-run step 2 with the
same inputs but `PrioritizationAlgorithm` set to one of `1`–`5`. The two
output files will differ in their filename suffix (`prio_index` is encoded
in the filename) and can be compared directly.

## Testing

The `test/` directory contains the upstream OnSSET regression suite (diesel
cost, elevation, grid penalty, land cover, road distance, slope, substation
distance, wind capacity factors, and an end-to-end run). To execute it:

```
pytest test
```

The end-to-end regression test in [`test/test_runner.py`](test/test_runner.py)
runs against the Djibouti fixture in `test/test_data/`. **The current test
suite does not cover the climate code paths**; users running the climate
pipeline should validate against their own datasets.

## Citation

If you use this software in academic work, please cite both the upstream
OnSSET project and this extension. Upstream OnSSET references are listed in
[`docs/publications.rst`](docs/publications.rst).

A `CITATION.cff` for this fork will be added with the JOSS submission.
Until then, please cite the repository URL and the commit hash.

## Acknowledgements

This repository extends OnSSET, developed at KTH Royal Institute of
Technology (dESA / Division of Energy Systems) and the broader OnSSET
community. The MIT [`LICENSE`](LICENSE) preserves the original copyright of
the OnSSET authors. We thank the OnSSET maintainers and contributors for
making the original tool open-source.

## Support and issues

Please file bug reports and feature requests for the **climate extension**
on this fork's GitHub Issues page. Questions or bugs that concern upstream
OnSSET (LCOE engines, calibration, grid extension, demand projection) are
better directed to the upstream repository at
<https://github.com/OnSSET/onsset>.

## License

MIT — see [`LICENSE`](LICENSE).
