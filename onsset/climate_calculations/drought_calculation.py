"""SPI-based drought risk calculation.

Computes per-admin-3-region drought risk from monthly precipitation data using
the Standardized Precipitation Index (SPI-k). Public entry point:
    - calculate_spi_drought_risk

Private helpers (_compute_spi_for_cell, _fit_countrywide_spi_baseline) implement
the per-cell gamma-fit baseline and SPI transform; they are not used outside
this module. The compound risk combination happens in climate_algorithm.py.
"""

import logging
from typing import Dict, Optional, Tuple

import pandas as pd
import numpy as np
import geopandas as gpd
from scipy.stats import gamma, norm

logger = logging.getLogger(__name__)


# Per-hazard configuration schema. Each entry maps a config dict key to a
# (excel_column_name, default_value) tuple. climate_algorithm.load_climate_config
# iterates all registered hazard schemas to build the merged config dict and to
# read overrides from the specs file. Adding/removing a parameter for this
# hazard only requires editing this dict.
CONFIG_SCHEMA = {
    'fields': {
        'spi_scale':              ('SPIScale',                  3),
        'spi_baseline_start':     ('SPIBaselineStartYear',      1950),
        'spi_baseline_end':       ('SPIBaselineEndYear',        2000),
        'precip_column':          ('PrecipitationColumnName',   'tp_mm_month'),
        'spi_drought_threshold':  ('SPIDroughtThreshold',      -1.0),
        'spi_mild_threshold':     ('SPIMildThreshold',         -1.5),
        'spi_moderate_threshold': ('SPIModerateThreshold',     -2.0),
        'spi_severe_threshold':   ('SPISevereThreshold',       -2.5),
        'drought_weight':         ('DroughtRiskWeight',         0.5),
    },
}


def _compute_spi_for_cell(
    df_cell: pd.DataFrame,
    date_col: str,
    precip_col: str,
    scale: int,
    baseline_start: int,
    baseline_end: int,
    baseline_params: Optional[Dict[int, Tuple[float, float, float]]] = None,
) -> pd.DataFrame:
    """Compute SPI-k for one grid cell.

    Args:
        df_cell: DataFrame for one cell with date and precipitation columns.
        date_col: Name of date column.
        precip_col: Name of precipitation column.
        scale: SPI accumulation period in months.
        baseline_start: Start year for gamma distribution fitting.
        baseline_end: End year for gamma distribution fitting.

    Returns:
        DataFrame with: date, year, month, P_k, spi
    """
    df = df_cell.sort_values(date_col).copy()

    # Drop duplicate dates (keep first occurrence), then set index
    df = df.drop_duplicates(subset=[date_col], keep='first')
    df = df.set_index(date_col)

    # Full continuous monthly index
    full_index = pd.date_range(df.index.min(), df.index.max(), freq='MS')
    df = df.reindex(full_index)

    # Fill missing precip with 0 mm
    df[precip_col] = df[precip_col].fillna(0.0)
    df['year'] = df.index.year
    df['month'] = df.index.month

    # k-month rolling accumulation
    df['P_k'] = df[precip_col].rolling(window=scale, min_periods=scale).sum()
    df['spi'] = np.nan

    # Compute SPI separately for each calendar month
    for m in range(1, 13):
        mask_month = df['month'] == m
        if not mask_month.any():
            continue

        series = df.loc[mask_month, 'P_k']

        if baseline_params is not None and m in baseline_params:
            shape, scale_param, q = baseline_params[m]
        else:
            # Fallback to cell-specific baseline subset for fitting
            baseline_mask = (
                mask_month &
                (df['year'] >= baseline_start) &
                (df['year'] <= baseline_end)
            )
            baseline_values = df.loc[baseline_mask, 'P_k'].dropna()

            if len(baseline_values) < 10:
                continue

            positive = baseline_values[baseline_values > 0]
            if len(positive) < 2:
                continue

            # Gamma fit: shape, loc=0, scale
            try:
                shape, loc, scale_param = gamma.fit(positive, floc=0)
            except Exception:
                continue

            q = len(positive) / len(baseline_values)  # non-zero probability

        x = series.values
        x_clipped = np.maximum(x, 0.0001)

        G = gamma.cdf(x_clipped, shape, loc=0, scale=scale_param)

        # Mixed distribution: mass at zero
        H = (1.0 - q) + q * G
        H[x <= 0] = (1.0 - q)

        H = np.clip(H, 1e-6, 1 - 1e-6)
        spi_vals = norm.ppf(H)

        df.loc[series.index, 'spi'] = spi_vals

    df = df.dropna(subset=['P_k', 'spi']).reset_index().rename(columns={'index': 'date'})
    return df[['date', 'year', 'month', 'P_k', 'spi']]


def _fit_countrywide_spi_baseline(
    climate_df: pd.DataFrame,
    lat_col: str,
    lon_col: str,
    date_col: str,
    precip_col: str,
    scale: int,
    baseline_start: int,
    baseline_end: int,
) -> Dict[int, Tuple[float, float, float]]:
    """Fit month-wise SPI baseline parameters using all cells countrywide.

    Returns a dict mapping month -> (shape, scale_param, q_nonzero).
    """
    df = climate_df[[lat_col, lon_col, date_col, precip_col]].copy()
    df = df.sort_values([lat_col, lon_col, date_col])
    df['year'] = df[date_col].dt.year
    df['month'] = df[date_col].dt.month

    # Build SPI-k precipitation sums for each cell using available monthly sequence.
    df['P_k'] = (
        df.groupby([lat_col, lon_col], sort=False)[precip_col]
        .transform(lambda s: s.rolling(window=scale, min_periods=scale).sum())
    )

    baseline_mask = (df['year'] >= baseline_start) & (df['year'] <= baseline_end)
    baseline_df = df.loc[baseline_mask, ['month', 'P_k']].dropna()

    baseline_params: Dict[int, Tuple[float, float, float]] = {}
    for m in range(1, 13):
        vals = baseline_df.loc[baseline_df['month'] == m, 'P_k']
        if len(vals) < 10:
            continue

        positive = vals[vals > 0]
        if len(positive) < 2:
            continue

        try:
            shape, loc, scale_param = gamma.fit(positive, floc=0)
        except Exception:
            continue

        q = len(positive) / len(vals)
        baseline_params[m] = (shape, scale_param, q)

    return baseline_params


def calculate_spi_drought_risk(
    climate_df: pd.DataFrame,
    admin3_gdf: gpd.GeoDataFrame,
    config: Dict,
    detected_columns: Dict[str, str]
) -> pd.DataFrame:
    """Calculate SPI-based drought risk scores per admin-3 region.

    This function:
    1. Computes SPI-k (Standardized Precipitation Index) per grid cell
    2. Aggregates to admin-3 regions
    3. Counts drought years by intensity category
    4. Produces normalized risk score (0-1)

    Args:
        climate_df: DataFrame with monthly precipitation data.
        admin3_gdf: GeoDataFrame with admin-3 boundaries.
        config: Configuration dictionary.
        detected_columns: Dict mapping standard names to actual column names.

    Returns:
        DataFrame with columns: admin3_id, admin3_name, drought_risk (0-1)
    """
    logger.info("Calculating SPI drought risk...")

    lat_col = detected_columns.get('latitude')
    lon_col = detected_columns.get('longitude')
    date_col = detected_columns.get('date')
    precip_col = detected_columns.get('precipitation')
    admin3_id_col = config['admin3_id_column']
    admin3_name_col = config['admin3_name_column']

    if precip_col is None or precip_col not in climate_df.columns:
        logger.warning("No precipitation column found, skipping drought calculation")
        return pd.DataFrame(columns=[admin3_id_col, admin3_name_col, 'drought_risk'])

    # Parse dates
    df = climate_df.copy()

    # Check if date column exists; if not, try to construct from year/month columns
    if date_col is not None and date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col])
        if 'year' not in df.columns:
            df['year'] = df[date_col].dt.year
        if 'month' not in df.columns:
            df['month'] = df[date_col].dt.month
    elif 'year' in df.columns and 'month' in df.columns:
        # Construct date from year and month columns
        logger.info("No date column found, constructing from year/month columns")
        df['date'] = pd.to_datetime(df[['year', 'month']].assign(day=1))
        date_col = 'date'
        detected_columns['date'] = 'date'
    else:
        logger.warning("No date column and no year/month columns found, skipping drought calculation")
        return pd.DataFrame(columns=[admin3_id_col, admin3_name_col, 'drought_risk'])

    # Compute SPI per grid cell
    spi_scale = int(config['spi_scale'])
    baseline_start = int(config['spi_baseline_start'])
    baseline_end = int(config['spi_baseline_end'])

    # Fit a countrywide monthly SPI baseline (shared across all cells).
    baseline_params = _fit_countrywide_spi_baseline(
        climate_df=df,
        lat_col=lat_col,
        lon_col=lon_col,
        date_col=date_col,
        precip_col=precip_col,
        scale=spi_scale,
        baseline_start=baseline_start,
        baseline_end=baseline_end,
    )
    if baseline_params:
        logger.info(
            f"Using countrywide SPI baseline from {baseline_start} to {baseline_end} "
            f"for months: {sorted(baseline_params.keys())}"
        )
    else:
        logger.warning(
            "Could not fit countrywide SPI baseline; falling back to cell-specific baseline fitting"
        )

    grouped_cells = df.groupby([lat_col, lon_col], sort=False)
    logger.info(f"Computing SPI-{spi_scale} for {len(grouped_cells)} grid cells...")

    cell_rows = []
    for (lat, lon), df_cell in grouped_cells:
        spi_df = _compute_spi_for_cell(
            df_cell, date_col, precip_col,
            spi_scale, baseline_start, baseline_end,
            baseline_params=baseline_params if baseline_params else None,
        )
        if spi_df.empty:
            continue
        spi_df[lat_col] = lat
        spi_df[lon_col] = lon
        cell_rows.append(spi_df)

    if not cell_rows:
        logger.warning("No SPI computed for any cell")
        return pd.DataFrame(columns=[admin3_id_col, admin3_name_col, 'drought_risk'])

    cell_spi = pd.concat(cell_rows, ignore_index=True)
    logger.info(f"Total SPI records (cell-level): {len(cell_spi):,}")

    # Spatial join to admin-3
    gdf_cells = gpd.GeoDataFrame(
        cell_spi,
        geometry=gpd.points_from_xy(cell_spi[lon_col], cell_spi[lat_col]),
        crs="EPSG:4326"
    )

    if admin3_gdf.crs is None or admin3_gdf.crs.to_epsg() != 4326:
        admin3_gdf = admin3_gdf.to_crs(epsg=4326)

    gdf_join = gpd.sjoin(
        gdf_cells,
        admin3_gdf[[admin3_id_col, admin3_name_col, 'geometry']],
        how="inner",
        predicate="within"
    )

    # Aggregate to region-month (mean over cells)
    df_reg_month = (
        gdf_join
        .groupby([admin3_id_col, admin3_name_col, 'date'], as_index=False)['spi']
        .mean()
        .rename(columns={'spi': 'spi_region_mean'})
    )
    df_reg_month['year'] = df_reg_month['date'].dt.year

    # Get thresholds
    drought_threshold = config['spi_drought_threshold']
    mild_threshold = config['spi_mild_threshold']
    moderate_threshold = config['spi_moderate_threshold']
    severe_threshold = config['spi_severe_threshold']

    # Calculate drought stats per region
    regions = admin3_gdf[[admin3_id_col, admin3_name_col]].drop_duplicates().reset_index(drop=True)

    stats_rows = []
    for _, reg in regions.iterrows():
        gid = reg[admin3_id_col]
        name = reg[admin3_name_col]

        df_r = df_reg_month[df_reg_month[admin3_id_col] == gid].copy()
        if df_r.empty:
            stats_rows.append({
                admin3_id_col: gid,
                admin3_name_col: name,
                'years_with_data': 0,
                'drought_years_total': 0,
                'frac_drought_years': 0,
            })
            continue

        years_present = sorted(df_r['year'].unique())
        drought_years = 0

        for year in years_present:
            df_y = df_r[df_r['year'] == year]
            if df_y.empty:
                continue

            min_spi = df_y['spi_region_mean'].min()
            if np.isnan(min_spi):
                continue

            if min_spi <= drought_threshold:
                drought_years += 1

        years_with_data = len(years_present)
        frac_drought = drought_years / years_with_data if years_with_data > 0 else 0

        stats_rows.append({
            admin3_id_col: gid,
            admin3_name_col: name,
            'years_with_data': years_with_data,
            'drought_years_total': drought_years,
            'frac_drought_years': frac_drought,
        })

    df_stats = pd.DataFrame(stats_rows)

    if df_stats.empty:
        logger.warning("No drought stats computed")
        return pd.DataFrame(columns=[admin3_id_col, admin3_name_col, 'drought_risk'])

    # Normalize to 0-1 risk score using min-max normalization
    # NOTE: Using min-max normalization. Consider revisiting for percentile-based approach.
    min_val = df_stats['frac_drought_years'].min()
    max_val = df_stats['frac_drought_years'].max()

    if max_val > min_val:
        df_stats['drought_risk'] = (df_stats['frac_drought_years'] - min_val) / (max_val - min_val)
    else:
        df_stats['drought_risk'] = 0.0

    logger.info(f"Computed drought risk for {len(df_stats)} regions")

    return df_stats[[admin3_id_col, admin3_name_col, 'drought_risk',
                     'drought_years_total', 'frac_drought_years']]
