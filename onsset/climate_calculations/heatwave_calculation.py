"""Heatwave risk calculation.

Computes per-admin-3-region heatwave risk from daily maximum temperature data.
Two entry points:
    - calculate_heatwave_risk: legacy in-memory path for small datasets
    - calculate_heatwave_risk_incremental: memory-efficient per-file streaming path

Both return a DataFrame indexed by admin-3 with a `heatwave_risk` column
normalized to 0-1. The compound risk combination happens in climate_algorithm.py.
"""

import logging
from collections import defaultdict
from typing import Dict

import pandas as pd
import numpy as np
import geopandas as gpd

logger = logging.getLogger(__name__)


# Per-hazard configuration schema. Each entry maps a config dict key to a
# (excel_column_name, default_value) tuple. climate_algorithm.load_climate_config
# iterates all registered hazard schemas to build the merged config dict and to
# read overrides from the specs file. Adding/removing a parameter for this
# hazard only requires editing this dict.
CONFIG_SCHEMA = {
    'fields': {
        'heatwave_threshold_c':   ('HeatwaveThresholdC',     32.0),
        'heatwave_duration_days': ('HeatwaveDurationDays',   3),
        'temp_column':            ('TemperatureColumnName',  't2m_max_C'),
        'hw_cat1_low':            ('HeatwaveCat1Low',        32.0),
        'hw_cat1_high':           ('HeatwaveCat1High',       35.0),
        'hw_cat2_low':            ('HeatwaveCat2Low',        35.0),
        'hw_cat2_high':           ('HeatwaveCat2High',       38.0),
        'hw_cat3_low':            ('HeatwaveCat3Low',        38.0),
        'hw_cat3_high':           ('HeatwaveCat3High',       42.0),
        'hw_cat4_low':            ('HeatwaveCat4Low',        42.0),
        'heatwave_weight':        ('HeatwaveRiskWeight',     0.5),
    },
}


def calculate_heatwave_risk(
    climate_df: pd.DataFrame,
    admin3_gdf: gpd.GeoDataFrame,
    config: Dict,
    detected_columns: Dict[str, str]
) -> pd.DataFrame:
    """Calculate heatwave risk scores per admin-3 region.

    This function:
    1. Assigns climate data points to admin-3 regions via spatial join
    2. Calculates region-mean daily max temperatures
    3. Counts heatwave days (temp > threshold)
    4. Calculates 3-day rolling mean temperatures and categorizes
    5. Produces normalized risk score (0-1)

    Args:
        climate_df: DataFrame with daily temperature data.
        admin3_gdf: GeoDataFrame with admin-3 boundaries.
        config: Configuration dictionary.
        detected_columns: Dict mapping standard names to actual column names.

    Returns:
        DataFrame with columns: admin3_id, admin3_name, heatwave_risk (0-1)
    """
    logger.info("Calculating heatwave risk...")

    lat_col = detected_columns.get('latitude')
    lon_col = detected_columns.get('longitude')
    date_col = detected_columns.get('date')
    temp_col = detected_columns.get('temperature')
    admin3_id_col = config['admin3_id_column']
    admin3_name_col = config['admin3_name_column']

    if temp_col is None or temp_col not in climate_df.columns:
        logger.warning("No temperature column found, skipping heatwave calculation")
        return pd.DataFrame(columns=[admin3_id_col, admin3_name_col, 'heatwave_risk'])

    # Build cell-to-region lookup
    cells = climate_df[[lat_col, lon_col]].drop_duplicates().reset_index(drop=True)
    gdf_cells = gpd.GeoDataFrame(
        cells,
        geometry=gpd.points_from_xy(cells[lon_col], cells[lat_col]),
        crs="EPSG:4326"
    )

    # Ensure admin3 is in correct CRS
    if admin3_gdf.crs is None or admin3_gdf.crs.to_epsg() != 4326:
        admin3_gdf = admin3_gdf.to_crs(epsg=4326)

    gdf_join = gpd.sjoin(
        gdf_cells,
        admin3_gdf[[admin3_id_col, admin3_name_col, 'geometry']],
        how="inner",
        predicate="within"
    )

    cell_region_lookup = gdf_join[[lat_col, lon_col, admin3_id_col, admin3_name_col]].copy()
    logger.info(f"Cell-region lookup rows: {len(cell_region_lookup)}")

    # Parse dates and merge with region lookup
    df = climate_df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.merge(cell_region_lookup, on=[lat_col, lon_col], how='left')
    df = df.dropna(subset=[admin3_id_col])

    if df.empty:
        logger.warning("No data after spatial join, returning empty heatwave risk")
        return pd.DataFrame(columns=[admin3_id_col, admin3_name_col, 'heatwave_risk'])

    df['year'] = df[date_col].dt.year
    years = sorted(df['year'].unique())

    # Region accumulators
    region_stats = defaultdict(lambda: {
        'heatwave_days_total': 0,
        'years_with_data': 0,
        'max3day_list': [],
    })

    hw_threshold = config['heatwave_threshold_c']

    for year in years:
        df_year = df[df['year'] == year].copy()
        if df_year.empty:
            continue

        # Region-mean daily max temperature
        reg_day = (
            df_year.groupby([admin3_id_col, admin3_name_col, date_col], as_index=False)[temp_col]
            .mean()
            .rename(columns={temp_col: 'tmax_reg_C'})
        )

        # Count heatwave days (temp > threshold)
        hw_mask = reg_day['tmax_reg_C'] > hw_threshold
        hw_days = (
            reg_day[hw_mask]
            .groupby([admin3_id_col, admin3_name_col], as_index=False)[date_col]
            .nunique()
            .rename(columns={date_col: 'heatwave_days_year'})
        )
        hw_days_dict = {
            (row[admin3_id_col], row[admin3_name_col]): int(row['heatwave_days_year'])
            for _, row in hw_days.iterrows()
        }

        # Max 3-day rolling mean per region
        max3day_this_year = {}
        for (gid, name), df_reg in reg_day.groupby([admin3_id_col, admin3_name_col], sort=False):
            s = df_reg.sort_values(date_col).set_index(date_col)['tmax_reg_C']
            if len(s) < 3:
                continue
            roll3 = s.rolling(window=3, min_periods=3).mean()
            max_val = float(roll3.max())
            if np.isfinite(max_val):
                max3day_this_year[(gid, name)] = max_val

        # Update accumulators
        regions_in_year = set(
            tuple(x) for x in reg_day[[admin3_id_col, admin3_name_col]].drop_duplicates().values
        )

        for key in regions_in_year:
            gid, name = key
            stats = region_stats[key]
            stats['heatwave_days_total'] += hw_days_dict.get(key, 0)
            stats['years_with_data'] += 1
            if key in max3day_this_year:
                stats['max3day_list'].append(max3day_this_year[key])

    # Build stats DataFrame
    rows = []
    for (gid, name), stats in region_stats.items():
        years_count = stats['years_with_data']
        if years_count == 0:
            mean_hw_days = 0
            mean_max3day = 0
        else:
            mean_hw_days = stats['heatwave_days_total'] / years_count
            mean_max3day = (
                float(np.mean(stats['max3day_list']))
                if stats['max3day_list'] else 0
            )

        rows.append({
            admin3_id_col: gid,
            admin3_name_col: name,
            'mean_heatwave_days_per_year': mean_hw_days,
            'mean_max_3day_T_C': mean_max3day,
            'years_with_data': years_count,
        })

    df_heat = pd.DataFrame(rows)

    if df_heat.empty:
        logger.warning("No heatwave stats computed")
        return pd.DataFrame(columns=[admin3_id_col, admin3_name_col, 'heatwave_risk'])

    # Normalize to 0-1 risk score using percentile-based ranking
    hw_days_col = 'mean_heatwave_days_per_year'
    df_heat['heatwave_risk'] = df_heat[hw_days_col].rank(method='average', pct=True)

    logger.info(f"Computed heatwave risk for {len(df_heat)} regions")

    return df_heat[[admin3_id_col, admin3_name_col, 'heatwave_risk',
                    'mean_heatwave_days_per_year', 'mean_max_3day_T_C']]


def calculate_heatwave_risk_incremental(
    loader: 'ClimateDataLoader',
    admin3_gdf: gpd.GeoDataFrame,
    config: Dict,
    detected_columns: Dict[str, str]
) -> pd.DataFrame:
    """Memory-efficient heatwave risk calculation processing files one at a time.

    This function processes daily temperature files incrementally to avoid
    loading all data into memory at once.

    Args:
        loader: ClimateDataLoader instance with classified files.
        admin3_gdf: GeoDataFrame with admin-3 boundaries.
        config: Configuration dictionary.
        detected_columns: Dict mapping standard names to actual column names.

    Returns:
        DataFrame with columns: admin3_id, admin3_name, heatwave_risk (0-1)
    """
    logger.info("Calculating heatwave risk (incremental mode)...")

    lat_col = detected_columns.get('latitude')
    lon_col = detected_columns.get('longitude')
    date_col = detected_columns.get('date')
    temp_col = detected_columns.get('temperature')
    admin3_id_col = config['admin3_id_column']
    admin3_name_col = config['admin3_name_column']
    hw_threshold = config['heatwave_threshold_c']

    # Ensure admin3 is in correct CRS
    if admin3_gdf.crs is None or admin3_gdf.crs.to_epsg() != 4326:
        admin3_gdf = admin3_gdf.to_crs(epsg=4326)

    # Build cell-to-region lookup from first file
    cell_region_lookup = None

    # Region accumulators (persist across all files)
    region_stats = defaultdict(lambda: {
        'heatwave_days_total': 0,
        'years_with_data': set(),  # Use set to track unique years
        'max3day_list': [],
    })

    files_processed = 0

    for filename, df in loader.iter_daily_temp_files():
        if temp_col is None:
            # Auto-detect from first file
            for col in df.columns:
                if 't2m' in col.lower() or 'temp' in col.lower():
                    temp_col = col
                    detected_columns['temperature'] = temp_col
                    break

        if temp_col is None or temp_col not in df.columns:
            logger.warning(f"No temperature column found in {filename}, skipping")
            continue

        # Build cell-region lookup once from first file's coordinates
        if cell_region_lookup is None:
            cells = df[[lat_col, lon_col]].drop_duplicates().reset_index(drop=True)
            gdf_cells = gpd.GeoDataFrame(
                cells,
                geometry=gpd.points_from_xy(cells[lon_col], cells[lat_col]),
                crs="EPSG:4326"
            )

            gdf_join = gpd.sjoin(
                gdf_cells,
                admin3_gdf[[admin3_id_col, admin3_name_col, 'geometry']],
                how="inner",
                predicate="within"
            )

            cell_region_lookup = gdf_join[[lat_col, lon_col, admin3_id_col, admin3_name_col]].copy()
            logger.info(f"Built cell-region lookup: {len(cell_region_lookup)} cells mapped to regions")

        # Parse dates and merge with region lookup
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.merge(cell_region_lookup, on=[lat_col, lon_col], how='inner')

        if df.empty:
            continue

        df['year'] = df[date_col].dt.year
        years = sorted(df['year'].unique())

        # Process each year in this file
        for year in years:
            df_year = df[df['year'] == year].copy()
            if df_year.empty:
                continue

            # Region-mean daily max temperature
            reg_day = (
                df_year.groupby([admin3_id_col, admin3_name_col, date_col], as_index=False)[temp_col]
                .mean()
                .rename(columns={temp_col: 'tmax_reg_C'})
            )

            # Count heatwave days (temp > threshold)
            hw_mask = reg_day['tmax_reg_C'] > hw_threshold
            hw_days = (
                reg_day[hw_mask]
                .groupby([admin3_id_col, admin3_name_col], as_index=False)[date_col]
                .nunique()
                .rename(columns={date_col: 'heatwave_days_year'})
            )
            hw_days_dict = {
                (row[admin3_id_col], row[admin3_name_col]): int(row['heatwave_days_year'])
                for _, row in hw_days.iterrows()
            }

            # Max 3-day rolling mean per region
            max3day_this_year = {}
            for (gid, name), df_reg in reg_day.groupby([admin3_id_col, admin3_name_col], sort=False):
                s = df_reg.sort_values(date_col).set_index(date_col)['tmax_reg_C']
                if len(s) < 3:
                    continue
                roll3 = s.rolling(window=3, min_periods=3).mean()
                max_val = float(roll3.max())
                if np.isfinite(max_val):
                    max3day_this_year[(gid, name)] = max_val

            # Update accumulators
            regions_in_year = set(
                tuple(x) for x in reg_day[[admin3_id_col, admin3_name_col]].drop_duplicates().values
            )

            for key in regions_in_year:
                gid, name = key
                stats = region_stats[key]
                stats['heatwave_days_total'] += hw_days_dict.get(key, 0)
                stats['years_with_data'].add(year)
                if key in max3day_this_year:
                    stats['max3day_list'].append(max3day_this_year[key])

        files_processed += 1
        # Free memory after processing each file
        del df

    logger.info(f"Processed {files_processed} daily temperature files")

    if not region_stats:
        logger.warning("No heatwave stats computed")
        return pd.DataFrame(columns=[admin3_id_col, admin3_name_col, 'heatwave_risk'])

    # Build stats DataFrame
    rows = []
    for (gid, name), stats in region_stats.items():
        years_count = len(stats['years_with_data'])
        if years_count == 0:
            mean_hw_days = 0
            mean_max3day = 0
        else:
            mean_hw_days = stats['heatwave_days_total'] / years_count
            mean_max3day = (
                float(np.mean(stats['max3day_list']))
                if stats['max3day_list'] else 0
            )

        rows.append({
            admin3_id_col: gid,
            admin3_name_col: name,
            'mean_heatwave_days_per_year': mean_hw_days,
            'mean_max_3day_T_C': mean_max3day,
            'years_with_data': years_count,
        })

    df_heat = pd.DataFrame(rows)

    if df_heat.empty:
        logger.warning("No heatwave stats computed")
        return pd.DataFrame(columns=[admin3_id_col, admin3_name_col, 'heatwave_risk'])

    # Normalize to 0-1 risk score using percentile-based ranking
    hw_days_col = 'mean_heatwave_days_per_year'
    df_heat['heatwave_risk'] = df_heat[hw_days_col].rank(method='average', pct=True)

    logger.info(f"Computed heatwave risk for {len(df_heat)} regions")

    return df_heat[[admin3_id_col, admin3_name_col, 'heatwave_risk',
                    'mean_heatwave_days_per_year', 'mean_max_3day_T_C']]
