"""Climate risk algorithm for OnSSET prioritization.

This module processes climate data (temperature, drought, etc.) from CSV files,
aggregates it to admin-3 level (municipality), calculates risk scores, and
provides prioritization data for electrification planning.

Data flow:
    1. GUI runner prompts for climate data folder + admin-3 shapefile
    2. ClimateDataLoader reads all CSVs, detects temporal resolution
    3. ClimateAggregator averages data per admin-3 area
    4. ClimateRiskCalculator computes individual and compound risk scores
    5. Results are mapped to population clusters for use in onsset.py

Configuration:
    All thresholds and parameters are read from the 'ClimateData' sheet in the
    specs Excel file. See specs.py for column name constants.
"""

import os
import logging
from enum import Enum
from typing import List, Dict, Optional, Tuple, Generator
from collections import defaultdict

import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from scipy.stats import gamma, norm

# Per-hazard risk calculators (moved to climate_calculations/ for modularity).
# Each module also exposes a CONFIG_SCHEMA so its parameters travel with the
# calculator — see the HAZARD_MODULES registration below.
try:
    from onsset.climate_calculations import heatwave_calculation, drought_calculation
except ImportError:
    from climate_calculations import heatwave_calculation, drought_calculation

calculate_heatwave_risk = heatwave_calculation.calculate_heatwave_risk
calculate_heatwave_risk_incremental = heatwave_calculation.calculate_heatwave_risk_incremental
calculate_spi_drought_risk = drought_calculation.calculate_spi_drought_risk

logging.basicConfig(format='%(asctime)s\t\t%(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS & CONFIGURATION
# =============================================================================

class TemporalResolution(Enum):
    """Detected temporal resolution of climate data."""
    HOURLY = 'hourly'
    DAILY = 'daily'
    MONTHLY = 'monthly'
    YEARLY = 'yearly'
    UNKNOWN = 'unknown'


class ClimateDataType(Enum):
    """Type of climate variable measured."""
    TEMPERATURE = 'temperature'
    PRECIPITATION = 'precipitation'
    PEV = 'pev'  # Potential Evapotranspiration
    UNKNOWN = 'unknown'


def detect_temporal_from_filename(filename: str) -> TemporalResolution:
    """Detect temporal resolution from filename patterns.

    Args:
        filename: Name of the climate data file.

    Returns:
        Detected TemporalResolution enum value.
    """
    fname_lower = filename.lower()
    if 'hourly' in fname_lower:
        return TemporalResolution.HOURLY
    elif 'daily' in fname_lower or 'dailymax' in fname_lower:
        return TemporalResolution.DAILY
    elif 'monthly' in fname_lower or 'monthlytotal' in fname_lower:
        return TemporalResolution.MONTHLY
    elif 'yearly' in fname_lower or 'annual' in fname_lower:
        return TemporalResolution.YEARLY
    return TemporalResolution.UNKNOWN


def detect_datatype_from_filename(filename: str) -> ClimateDataType:
    """Detect climate data type from filename patterns.

    Args:
        filename: Name of the climate data file.

    Returns:
        Detected ClimateDataType enum value.
    """
    fname_lower = filename.lower()
    # Temperature indicators
    if any(pat in fname_lower for pat in ['t2m', 'temp', 'temperature', 'tmax', 'tmin']):
        return ClimateDataType.TEMPERATURE
    # Precipitation indicators
    if any(pat in fname_lower for pat in ['tp_', 'tp-', 'precip', 'precipitation', 'rainfall', 'rain']):
        return ClimateDataType.PRECIPITATION
    # PEV indicators
    if any(pat in fname_lower for pat in ['pev', 'evapotranspiration', 'evap', 'pet']):
        return ClimateDataType.PEV
    return ClimateDataType.UNKNOWN


# Output column names (for integration with onsset.py)
SET_NORMALIZED_CLIMATE_HAZARD = 'NormalizedClimateHazard'  # Compound normalized hazard score
SET_CLIMATE_HAZARD = 'ClimateHazard'  # Explicit hazard component (heatwave + drought)
SET_CLIMATE_VULNERABILITY = 'ClimateVulnerability'  # Susceptibility component (wealth + remoteness)
SET_CLIMATE_PRIORITY = 'ClimatePriority'  # Prioritization score: Hazard × Vulnerability
SET_CLIMATE_RISK_HEATWAVE = 'ClimateRiskHeatwave'  # Individual heatwave hazard
SET_CLIMATE_RISK_DROUGHT = 'ClimateRiskDrought'  # Individual drought hazard
SET_ADMIN3_ID = 'Admin3ID'  # Municipality identifier

# Deprecated: SET_CLIMATE_EXPOSURE no longer used (population accounts for scheduling, not priority)
# Deprecated: SET_CLIMATE_RISK previously stored population-weighted risk; now use ClimatePriority

# Global variable for detected temporal resolution
_temporal_resolution: TemporalResolution = TemporalResolution.UNKNOWN


# =============================================================================
# CONFIGURATION SCHEMA
# =============================================================================
# Per-hazard parameters are owned by each climate_calculations/<hazard>.py
# module via its CONFIG_SCHEMA attribute. To add a new hazard (e.g. flood):
#   1. Add climate_calculations/flood_calculation.py with a CONFIG_SCHEMA dict
#   2. Append the module to HAZARD_MODULES below
# Nothing in specs.py needs to change.

# Name of the Excel sheet that holds climate config overrides. A hazard module
# may override this per-schema via the 'spec_sheet' key.
CLIMATE_CONFIG_SHEET = 'ClimateData'

# Row index (0-based) within the sheet to read. A hazard module may override
# via the 'spec_row' key on its schema.
CLIMATE_CONFIG_ROW = 0

# Shared parameters used by the loader, spatial join, and any hazard module.
COMMON_CONFIG_SCHEMA = {
    'fields': {
        'lat_column':         ('LatitudeColumnName',  'latitude'),
        'lon_column':         ('LongitudeColumnName', 'longitude'),
        'date_column':        ('DateColumnName',      'date'),
        'admin3_id_column':   ('Admin3IDColumn',      'GID_3'),
        'admin3_name_column': ('Admin3NameColumn',    'NAME_3'),
    },
}

# Registered hazard modules. Each module must expose a CONFIG_SCHEMA dict.
HAZARD_MODULES = [heatwave_calculation, drought_calculation]


def _iter_schemas():
    """Yield (schema_dict, sheet_name, row_idx) for common + every registered hazard."""
    yield COMMON_CONFIG_SCHEMA, CLIMATE_CONFIG_SHEET, CLIMATE_CONFIG_ROW
    for module in HAZARD_MODULES:
        schema = module.CONFIG_SCHEMA
        yield (
            schema,
            schema.get('spec_sheet', CLIMATE_CONFIG_SHEET),
            schema.get('spec_row', CLIMATE_CONFIG_ROW),
        )


def load_climate_config(specs_path: Optional[str] = None) -> Dict:
    """Load climate configuration from specs file or use defaults.

    Iterates the common schema and every registered hazard schema to build the
    merged config dict, then overrides values from the specs Excel file when
    present.

    Args:
        specs_path: Path to specs Excel file. If None, use defaults.

    Returns:
        Dictionary with configuration values.
    """
    # Start from defaults declared in each schema.
    config: Dict = {}
    for schema, _, _ in _iter_schemas():
        for key, (_, default) in schema['fields'].items():
            config[key] = default

    if specs_path is None or not os.path.exists(specs_path):
        logger.info("Using default climate configuration values")
        return config

    # Group schemas by (sheet, row) so each sheet is read at most once.
    by_sheet: Dict[Tuple[str, int], List[Tuple[str, str]]] = defaultdict(list)
    for schema, sheet, row_idx in _iter_schemas():
        for key, (spec_col, _) in schema['fields'].items():
            by_sheet[(sheet, row_idx)].append((spec_col, key))

    for (sheet, row_idx), entries in by_sheet.items():
        try:
            sheet_df = pd.read_excel(specs_path, sheet_name=sheet)
            if sheet_df.empty or row_idx >= len(sheet_df):
                logger.warning(f"Sheet '{sheet}' has no row {row_idx}, using defaults for its fields")
                continue
            row = sheet_df.iloc[row_idx]
            for spec_col, config_key in entries:
                if spec_col in row.index and pd.notna(row[spec_col]):
                    config[config_key] = row[spec_col]
        except Exception as e:
            logger.warning(f"Failed to load sheet '{sheet}': {e}. Using defaults for its fields.")

    logger.info("Loaded climate configuration from specs file")
    return config


# =============================================================================
# DATA LOADING
# =============================================================================

class ClimateDataLoader:
    """Loads and validates climate data from a folder of CSV files.

    Memory-efficient: classifies files by temporal resolution and data type,
    then loads them separately to avoid memory issues with large datasets.
    """

    def __init__(self, folder_path: str, config: Dict):
        """
        Args:
            folder_path: Path to folder containing climate CSV files.
            config: Configuration dictionary from load_climate_config().
        """
        self.folder_path = folder_path
        self.config = config
        self.dataframes: List[pd.DataFrame] = []
        self.temporal_resolution: TemporalResolution = TemporalResolution.UNKNOWN

        # New nested classification structure: classified_files[temporal][datatype] = [filenames]
        self.classified_files: Dict[TemporalResolution, Dict[ClimateDataType, List[str]]] = {
            res: {dtype: [] for dtype in ClimateDataType}
            for res in TemporalResolution
        }

        # Common columns (lat, lon, date) - shared across data types
        self.common_columns: Dict[str, str] = {}

        # Per-datatype column detection
        self._detected_columns_by_type: Dict[ClimateDataType, Dict[str, str]] = {
            dtype: {} for dtype in ClimateDataType
        }

        # Legacy detected_columns dict (for backward compatibility)
        self.detected_columns: Dict[str, str] = {}

        # Flag to prevent re-classification
        self._files_classified: bool = False

    # -------------------------------------------------------------------------
    # Backward Compatibility Properties
    # -------------------------------------------------------------------------

    @property
    def daily_temp_files(self) -> List[str]:
        """Backward compatibility: list of daily temperature files."""
        return self.classified_files[TemporalResolution.DAILY][ClimateDataType.TEMPERATURE]

    @property
    def monthly_precip_files(self) -> List[str]:
        """Backward compatibility: list of monthly precipitation files."""
        return self.classified_files[TemporalResolution.MONTHLY][ClimateDataType.PRECIPITATION]

    @property
    def monthly_pev_files(self) -> List[str]:
        """Backward compatibility: list of monthly PEV files."""
        return self.classified_files[TemporalResolution.MONTHLY][ClimateDataType.PEV]

    @property
    def other_files(self) -> List[str]:
        """Backward compatibility: files that couldn't be classified."""
        result = []
        for datatype in ClimateDataType:
            result.extend(self.classified_files[TemporalResolution.UNKNOWN][datatype])
        return result

    # -------------------------------------------------------------------------
    # File Classification
    # -------------------------------------------------------------------------

    def _classify_files(self) -> None:
        """Classify files by temporal resolution AND data type independently."""
        # Check if already classified to prevent duplicate entries
        if self._files_classified:
            return

        if not os.path.isdir(self.folder_path):
            raise ValueError(f"Climate data folder not found: {self.folder_path}")

        csv_files = [f for f in os.listdir(self.folder_path) if f.endswith('.csv')]
        excel_files = [f for f in os.listdir(self.folder_path)
                       if f.endswith(('.xlsx', '.xls'))]
        all_files = csv_files + excel_files

        if not all_files:
            raise ValueError(f"No CSV or Excel files found in {self.folder_path}")

        for filename in all_files:
            # Independent detection
            temporal = detect_temporal_from_filename(filename)
            datatype = detect_datatype_from_filename(filename)

            # Apply legacy rules for backward compatibility
            temporal, datatype = self._apply_legacy_rules(filename, temporal, datatype)

            # Store in nested structure
            self.classified_files[temporal][datatype].append(filename)

        # Sort all lists for consistent processing order
        for temporal in self.classified_files:
            for datatype in self.classified_files[temporal]:
                self.classified_files[temporal][datatype].sort()

        self._files_classified = True
        self._log_classification_summary()

    def _apply_legacy_rules(
        self,
        filename: str,
        temporal: TemporalResolution,
        datatype: ClimateDataType
    ) -> Tuple[TemporalResolution, ClimateDataType]:
        """Apply backward-compatible rules from original hard-coded patterns.

        Original patterns:
        - t2m + (daily|max) -> daily + temperature
        - tp_ + monthly -> monthly + precipitation
        - pev + monthly -> monthly + pev
        """
        fname_lower = filename.lower()

        # Rule 1: t2m with 'max' implies daily temperature even without 'daily' keyword
        if 't2m' in fname_lower and 'max' in fname_lower:
            if temporal == TemporalResolution.UNKNOWN:
                temporal = TemporalResolution.DAILY
            if datatype == ClimateDataType.UNKNOWN:
                datatype = ClimateDataType.TEMPERATURE

        return temporal, datatype

    def _log_classification_summary(self) -> None:
        """Log summary of file classification."""
        summary_parts = []
        for temporal in TemporalResolution:
            if temporal == TemporalResolution.UNKNOWN:
                continue
            for datatype in ClimateDataType:
                if datatype == ClimateDataType.UNKNOWN:
                    continue
                count = len(self.classified_files[temporal][datatype])
                if count > 0:
                    summary_parts.append(f"{count} {temporal.value} {datatype.value}")

        unknown_count = sum(
            len(self.classified_files[TemporalResolution.UNKNOWN][dt])
            for dt in ClimateDataType
        )
        if unknown_count:
            summary_parts.append(f"{unknown_count} unclassified")

        logger.info(f"Classified files: {', '.join(summary_parts)}")

    # -------------------------------------------------------------------------
    # Data Availability Checks
    # -------------------------------------------------------------------------

    def has_data(self, temporal: TemporalResolution, datatype: ClimateDataType) -> bool:
        """Check if data is available for a specific temporal/datatype combination.

        Args:
            temporal: Temporal resolution to check.
            datatype: Data type to check.

        Returns:
            True if files exist for this combination.
        """
        if not any(
            self.classified_files[t][d]
            for t in TemporalResolution
            for d in ClimateDataType
        ):
            self._classify_files()
        return len(self.classified_files[temporal][datatype]) > 0

    def get_available_combinations(self) -> List[Tuple[TemporalResolution, ClimateDataType]]:
        """Get list of all (temporal, datatype) combinations with available data.

        Returns:
            List of (TemporalResolution, ClimateDataType) tuples with files.
        """
        if not any(
            self.classified_files[t][d]
            for t in TemporalResolution
            for d in ClimateDataType
        ):
            self._classify_files()

        result = []
        for temporal in TemporalResolution:
            for datatype in ClimateDataType:
                if self.classified_files[temporal][datatype]:
                    result.append((temporal, datatype))
        return result

    def has_daily_temp_data(self) -> bool:
        """Backward compatibility check for daily temperature data."""
        return self.has_data(TemporalResolution.DAILY, ClimateDataType.TEMPERATURE)

    def has_monthly_precip_data(self) -> bool:
        """Backward compatibility check for monthly precipitation data."""
        return self.has_data(TemporalResolution.MONTHLY, ClimateDataType.PRECIPITATION)

    def load_all_files(self) -> pd.DataFrame:
        """Load all CSV/Excel files from the folder and combine into single DataFrame.

        WARNING: This method loads ALL files at once. For large datasets with
        mixed temporal resolutions, use load_daily_temp_files() and
        load_monthly_precip_files() separately instead.

        Returns:
            Combined DataFrame with all climate data.
        """
        if not os.path.isdir(self.folder_path):
            raise ValueError(f"Climate data folder not found: {self.folder_path}")

        csv_files = [f for f in os.listdir(self.folder_path) if f.endswith('.csv')]
        excel_files = [f for f in os.listdir(self.folder_path)
                       if f.endswith(('.xlsx', '.xls'))]

        if not csv_files and not excel_files:
            raise ValueError(f"No CSV or Excel files found in {self.folder_path}")

        logger.info(f"Found {len(csv_files)} CSV files and {len(excel_files)} Excel files")

        for filename in csv_files:
            filepath = os.path.join(self.folder_path, filename)
            try:
                df = pd.read_csv(filepath)
                self.dataframes.append(df)
                logger.info(f"Loaded {filename}: {len(df)} rows")
            except Exception as e:
                logger.warning(f"Failed to load {filename}: {e}")

        for filename in excel_files:
            filepath = os.path.join(self.folder_path, filename)
            try:
                df = pd.read_excel(filepath)
                self.dataframes.append(df)
                logger.info(f"Loaded {filename}: {len(df)} rows")
            except Exception as e:
                logger.warning(f"Failed to load {filename}: {e}")

        if not self.dataframes:
            raise ValueError("No climate data files could be loaded")

        combined_df = pd.concat(self.dataframes, ignore_index=True)
        logger.info(f"Combined climate data: {len(combined_df)} total rows")

        self._detect_columns(combined_df)
        self.temporal_resolution = self.detect_temporal_resolution(combined_df)

        return combined_df

    def _load_file(self, filename: str) -> Optional[pd.DataFrame]:
        """Load a single file (CSV or Excel)."""
        filepath = os.path.join(self.folder_path, filename)
        try:
            if filename.endswith('.csv'):
                return pd.read_csv(filepath)
            else:
                return pd.read_excel(filepath)
        except Exception as e:
            logger.warning(f"Failed to load {filename}: {e}")
            return None

    # -------------------------------------------------------------------------
    # Generic Loader Methods
    # -------------------------------------------------------------------------

    def iter_files(
        self,
        temporal: Optional[TemporalResolution] = None,
        datatype: Optional[ClimateDataType] = None
    ) -> Generator[Tuple[str, pd.DataFrame], None, None]:
        """Iterate over classified files, optionally filtered by temporal/datatype.

        Args:
            temporal: Filter by temporal resolution (None = all)
            datatype: Filter by data type (None = all)

        Yields:
            Tuple of (filename, DataFrame) for each matching file.
        """
        if not any(
            self.classified_files[t][d]
            for t in TemporalResolution
            for d in ClimateDataType
        ):
            self._classify_files()

        temporals = [temporal] if temporal else list(TemporalResolution)
        datatypes = [datatype] if datatype else list(ClimateDataType)

        for t in temporals:
            for d in datatypes:
                for filename in self.classified_files[t][d]:
                    df = self._load_file(filename)
                    if df is not None:
                        logger.info(f"Loaded {filename}: {len(df)} rows")
                        yield filename, df

    def load_files(
        self,
        temporal: Optional[TemporalResolution] = None,
        datatype: Optional[ClimateDataType] = None
    ) -> pd.DataFrame:
        """Load and combine files matching temporal/datatype criteria.

        Args:
            temporal: Filter by temporal resolution (None = all)
            datatype: Filter by data type (None = all)

        Returns:
            Combined DataFrame with all matching data.
        """
        dfs = []
        for filename, df in self.iter_files(temporal, datatype):
            dfs.append(df)

        if not dfs:
            return pd.DataFrame()

        combined = pd.concat(dfs, ignore_index=True)

        # Detect columns for the appropriate datatype
        if datatype and datatype != ClimateDataType.UNKNOWN:
            self._detect_columns_for_datatype(combined, datatype)
        else:
            self._detect_columns(combined)

        logger.info(f"Combined data: {len(combined)} total rows")
        return combined

    # -------------------------------------------------------------------------
    # Backward Compatible Loader Methods
    # -------------------------------------------------------------------------

    def load_monthly_precip_files(self) -> pd.DataFrame:
        """Load only monthly precipitation files (memory-efficient for drought analysis).

        Returns:
            Combined DataFrame with monthly precipitation data.
        """
        df = self.load_files(
            temporal=TemporalResolution.MONTHLY,
            datatype=ClimateDataType.PRECIPITATION
        )
        if not df.empty:
            logger.info(f"Combined monthly precip data: {len(df)} total rows")
        else:
            logger.warning("No monthly precipitation files found")
        return df

    def iter_daily_temp_files(self):
        """Iterate over daily temperature files one at a time (memory-efficient).

        Yields:
            Tuple of (filename, DataFrame) for each daily temperature file.
        """
        yield from self.iter_files(
            temporal=TemporalResolution.DAILY,
            datatype=ClimateDataType.TEMPERATURE
        )

    def get_sample_for_column_detection(self) -> pd.DataFrame:
        """Load a small sample to detect column names without loading all data.

        Returns:
            Sample DataFrame for column detection.
        """
        self._classify_files()

        # Try to get a sample from each type
        sample_files = []
        if self.daily_temp_files:
            sample_files.append(self.daily_temp_files[0])
        if self.monthly_precip_files:
            sample_files.append(self.monthly_precip_files[0])
        if not sample_files and self.other_files:
            sample_files.append(self.other_files[0])

        dfs = []
        for filename in sample_files:
            df = self._load_file(filename)
            if df is not None:
                dfs.append(df.head(1000))  # Only take first 1000 rows

        if not dfs:
            raise ValueError("Could not load any sample files for column detection")

        sample_df = pd.concat(dfs, ignore_index=True)
        self._detect_columns(sample_df)
        return sample_df

    def has_daily_temp_data(self) -> bool:
        """Check if daily temperature data is available."""
        if not self.daily_temp_files:
            self._classify_files()
        return len(self.daily_temp_files) > 0

    def has_monthly_precip_data(self) -> bool:
        """Check if monthly precipitation data is available."""
        if not self.monthly_precip_files:
            self._classify_files()
        return len(self.monthly_precip_files) > 0

    # -------------------------------------------------------------------------
    # Column Detection (Modular)
    # -------------------------------------------------------------------------

    def _detect_columns(self, df: pd.DataFrame):
        """Auto-detect column names for lat, lon, date, temp, precip.

        This is the legacy method that populates self.detected_columns.
        """
        columns = df.columns.tolist()
        columns_lower = [c.lower() for c in columns]

        # Detect common columns
        self._detect_common_columns(columns, columns_lower)

        # Detect all data type columns
        self._detect_temperature_columns(columns, columns_lower)
        self._detect_precipitation_columns(columns, columns_lower)
        self._detect_pev_columns(columns, columns_lower)

        # Populate legacy detected_columns dict from common_columns and type-specific
        self.detected_columns.update(self.common_columns)
        if 'value' in self._detected_columns_by_type[ClimateDataType.TEMPERATURE]:
            self.detected_columns['temperature'] = self._detected_columns_by_type[ClimateDataType.TEMPERATURE]['value']
        if 'value' in self._detected_columns_by_type[ClimateDataType.PRECIPITATION]:
            self.detected_columns['precipitation'] = self._detected_columns_by_type[ClimateDataType.PRECIPITATION]['value']

        logger.info(f"Detected columns: {self.detected_columns}")

    def _detect_columns_for_datatype(self, df: pd.DataFrame, datatype: ClimateDataType):
        """Detect columns specific to a data type.

        Args:
            df: DataFrame to analyze.
            datatype: The type of climate data to detect columns for.
        """
        columns = df.columns.tolist()
        columns_lower = [c.lower() for c in columns]

        # Always detect common columns
        self._detect_common_columns(columns, columns_lower)

        # Type-specific detection
        if datatype == ClimateDataType.TEMPERATURE:
            self._detect_temperature_columns(columns, columns_lower)
        elif datatype == ClimateDataType.PRECIPITATION:
            self._detect_precipitation_columns(columns, columns_lower)
        elif datatype == ClimateDataType.PEV:
            self._detect_pev_columns(columns, columns_lower)

        # Update legacy detected_columns for backward compatibility
        self.detected_columns.update(self.common_columns)

    def _detect_common_columns(self, columns: List[str], columns_lower: List[str]) -> None:
        """Detect latitude, longitude, and date columns."""
        # Latitude
        lat_options = [self.config['lat_column'], 'latitude', 'lat', 'y', 'y_deg']
        for opt in lat_options:
            if opt.lower() in columns_lower:
                idx = columns_lower.index(opt.lower())
                self.common_columns['latitude'] = columns[idx]
                break

        # Longitude
        lon_options = [self.config['lon_column'], 'longitude', 'lon', 'x', 'x_deg']
        for opt in lon_options:
            if opt.lower() in columns_lower:
                idx = columns_lower.index(opt.lower())
                self.common_columns['longitude'] = columns[idx]
                break

        # Date
        date_options = [self.config['date_column'], 'date', 'timestamp', 'time', 'datetime']
        for opt in date_options:
            if opt.lower() in columns_lower:
                idx = columns_lower.index(opt.lower())
                self.common_columns['date'] = columns[idx]
                break

    def _detect_temperature_columns(self, columns: List[str], columns_lower: List[str]) -> None:
        """Detect temperature-specific columns."""
        temp_options = [
            self.config['temp_column'],
            't2m_max_c', 't2m_max', 'temperature', 'temp', 'tmax', 'tmin', 't2m'
        ]
        for opt in temp_options:
            if opt.lower() in columns_lower:
                idx = columns_lower.index(opt.lower())
                self._detected_columns_by_type[ClimateDataType.TEMPERATURE]['value'] = columns[idx]
                break

    def _detect_precipitation_columns(self, columns: List[str], columns_lower: List[str]) -> None:
        """Detect precipitation-specific columns."""
        precip_options = [
            self.config['precip_column'],
            'tp_mm_month', 'tp_mm', 'precipitation', 'precip', 'rainfall', 'rain', 'tp'
        ]
        for opt in precip_options:
            if opt.lower() in columns_lower:
                idx = columns_lower.index(opt.lower())
                self._detected_columns_by_type[ClimateDataType.PRECIPITATION]['value'] = columns[idx]
                break

    def _detect_pev_columns(self, columns: List[str], columns_lower: List[str]) -> None:
        """Detect PEV-specific columns."""
        pev_options = ['pev', 'evapotranspiration', 'evap', 'et', 'pet']
        for opt in pev_options:
            if opt.lower() in columns_lower:
                idx = columns_lower.index(opt.lower())
                self._detected_columns_by_type[ClimateDataType.PEV]['value'] = columns[idx]
                break

    def get_columns(self, datatype: Optional[ClimateDataType] = None) -> Dict[str, str]:
        """Get detected column names, merging common and type-specific columns.

        Args:
            datatype: Optional data type to get specific columns for.

        Returns:
            Dictionary with column mappings.
        """
        result = self.common_columns.copy()

        if datatype and datatype in self._detected_columns_by_type:
            type_cols = self._detected_columns_by_type[datatype]
            if 'value' in type_cols:
                if datatype == ClimateDataType.TEMPERATURE:
                    result['temperature'] = type_cols['value']
                elif datatype == ClimateDataType.PRECIPITATION:
                    result['precipitation'] = type_cols['value']
                elif datatype == ClimateDataType.PEV:
                    result['pev'] = type_cols['value']

        return result

    def detect_temporal_resolution(self, df: pd.DataFrame) -> TemporalResolution:
        """Analyze date column to determine data resolution.

        Args:
            df: DataFrame with date column.

        Returns:
            Detected TemporalResolution enum value.
        """
        global _temporal_resolution

        date_col = self.detected_columns.get('date')
        if date_col is None or date_col not in df.columns:
            logger.warning("No date column found, defaulting to UNKNOWN resolution")
            _temporal_resolution = TemporalResolution.UNKNOWN
            return _temporal_resolution

        try:
            timestamps = pd.to_datetime(df[date_col])
        except Exception as e:
            logger.warning(f"Failed to parse timestamps: {e}")
            _temporal_resolution = TemporalResolution.UNKNOWN
            return _temporal_resolution

        timestamps_sorted = timestamps.sort_values().reset_index(drop=True)
        if len(timestamps_sorted) < 2:
            logger.warning("Not enough timestamps to detect resolution")
            _temporal_resolution = TemporalResolution.UNKNOWN
            return _temporal_resolution

        time_diffs = timestamps_sorted.diff().dropna()
        median_diff_hours = time_diffs.median().total_seconds() / 3600

        if median_diff_hours < 2:
            _temporal_resolution = TemporalResolution.HOURLY
        elif median_diff_hours < 48:
            _temporal_resolution = TemporalResolution.DAILY
        elif median_diff_hours < 45 * 24:
            _temporal_resolution = TemporalResolution.MONTHLY
        else:
            _temporal_resolution = TemporalResolution.YEARLY

        logger.info(f"Detected temporal resolution: {_temporal_resolution.value} "
                   f"(median diff: {median_diff_hours:.1f} hours)")
        return _temporal_resolution


# =============================================================================
# COMPOUND RISK & SETTLEMENT MAPPING
# =============================================================================

def calculate_compound_risk(
    heatwave_risk_df: pd.DataFrame,
    drought_risk_df: pd.DataFrame,
    config: Dict
) -> pd.DataFrame:
    """Combine heatwave and drought risks into compound risk score.

    Args:
        heatwave_risk_df: DataFrame with heatwave_risk column.
        drought_risk_df: DataFrame with drought_risk column.
        config: Configuration dictionary with risk weights.

    Returns:
        DataFrame with compound_risk column.
    """
    admin3_id_col = config['admin3_id_column']
    admin3_name_col = config['admin3_name_column']
    hw_weight = config['heatwave_weight']
    drought_weight = config['drought_weight']

    # Handle empty DataFrames
    if heatwave_risk_df.empty and drought_risk_df.empty:
        return pd.DataFrame(columns=[admin3_id_col, admin3_name_col, 'compound_risk'])

    if heatwave_risk_df.empty:
        df = drought_risk_df.copy()
        df['compound_risk'] = df['drought_risk']
        return df[[admin3_id_col, admin3_name_col, 'compound_risk', 'drought_risk']]

    if drought_risk_df.empty:
        df = heatwave_risk_df.copy()
        df['compound_risk'] = df['heatwave_risk']
        return df[[admin3_id_col, admin3_name_col, 'compound_risk', 'heatwave_risk']]

    # Merge on admin3 ID
    df = heatwave_risk_df[[admin3_id_col, admin3_name_col, 'heatwave_risk']].merge(
        drought_risk_df[[admin3_id_col, 'drought_risk']],
        on=admin3_id_col,
        how='outer'
    )

    # Fill missing values with 0
    df['heatwave_risk'] = df['heatwave_risk'].fillna(0)
    df['drought_risk'] = df['drought_risk'].fillna(0)

    # Weighted combination
    total_weight = hw_weight + drought_weight
    df['compound_risk'] = (
        (df['heatwave_risk'] * hw_weight + df['drought_risk'] * drought_weight)
        / total_weight
    )

    return df


def map_risk_to_settlements(
    settlements_df: pd.DataFrame,
    risk_df: pd.DataFrame,
    admin3_gdf: gpd.GeoDataFrame,
    config: Dict,
    lat_col: str = 'Y_deg',
    lon_col: str = 'X_deg'
) -> pd.DataFrame:
    """Map climate risk from admin-3 regions to settlements.

    Computes climate prioritization score: Priority = Hazard × Vulnerability
    Population is NOT included in the priority score because it is already accounted
    for by the electrification rollout rule (fixed % targets per timestep).

    Args:
        settlements_df: OnSSET settlements DataFrame.
        risk_df: DataFrame with risk scores per admin3.
        admin3_gdf: GeoDataFrame with admin-3 boundaries.
        config: Configuration dictionary.
        lat_col: Name of latitude column in settlements.
        lon_col: Name of longitude column in settlements.

    Returns:
        Settlements DataFrame with climate columns added:
        - ClimateHazard: Compound hazard (heatwave + drought)
        - ClimateVulnerability: Susceptibility (wealth + remoteness)
        - ClimatePriority: Final prioritization score (Hazard × Vulnerability)
        - ClimateRiskHeatwave, ClimateRiskDrought: Individual hazards
    """
    admin3_id_col = config['admin3_id_column']

    # Convert settlements to GeoDataFrame
    gdf_settlements = gpd.GeoDataFrame(
        settlements_df,
        geometry=gpd.points_from_xy(settlements_df[lon_col], settlements_df[lat_col]),
        crs="EPSG:4326"
    )

    # Ensure admin3 is in correct CRS
    if admin3_gdf.crs is None or admin3_gdf.crs.to_epsg() != 4326:
        admin3_gdf = admin3_gdf.to_crs(epsg=4326)

    # Spatial join
    gdf_join = gpd.sjoin(
        gdf_settlements,
        admin3_gdf[[admin3_id_col, 'geometry']],
        how='left',
        predicate='within'
    )

    # Add admin3 ID to settlements
    settlements_df[SET_ADMIN3_ID] = gdf_join[admin3_id_col].values

    # Merge risk scores
    risk_cols = [c for c in risk_df.columns if 'risk' in c.lower()]
    merge_cols = [admin3_id_col] + risk_cols

    settlements_df = settlements_df.merge(
        risk_df[merge_cols],
        left_on=SET_ADMIN3_ID,
        right_on=admin3_id_col,
        how='left'
    )

    # Rename columns for OnSSET integration
    if 'compound_risk' in settlements_df.columns:
        hazard_values = pd.to_numeric(settlements_df['compound_risk'], errors='coerce')
        hazard_min = hazard_values.min()
        hazard_max = hazard_values.max()
        if pd.notna(hazard_min) and pd.notna(hazard_max) and hazard_max > hazard_min:
            hazard_values = (hazard_values - hazard_min) / (hazard_max - hazard_min)
        else:
            hazard_values = hazard_values.fillna(0)

        # Store normalized hazard component (keep legacy name for backward compatibility)
        settlements_df[SET_NORMALIZED_CLIMATE_HAZARD] = hazard_values
        settlements_df[SET_CLIMATE_HAZARD] = hazard_values

        # Compute Vulnerability component: ((1 - normalized_wealth) + normalized_travel) / 2
        # Try multiple column name variants for robustness
        wealth_col = next((col for col in [
            'NormalizedRelativeWealth',
            'normalized_wealth_index',
            'NormalizedWealth',
            'normalized_relative_wealth',
        ] if col in settlements_df.columns), None)

        travel_col = next((col for col in [
            'NormalizedTravelHours',
            'normalized_travel_hours',
            'NormalizedTravel',
        ] if col in settlements_df.columns), None)

        if wealth_col and travel_col:
            wealth_values = pd.to_numeric(settlements_df[wealth_col], errors='coerce').fillna(0)
            travel_values = pd.to_numeric(settlements_df[travel_col], errors='coerce').fillna(0)
            # Vulnerability: invert wealth (high wealth = low vulnerability), average with travel
            vulnerability_values = ((1 - wealth_values) + travel_values) / 2
            settlements_df[SET_CLIMATE_VULNERABILITY] = vulnerability_values
        else:
            # Fallback: if wealth/travel missing, set vulnerability to neutral (0.5)
            vulnerability_values = pd.Series(0.5, index=settlements_df.index)
            logger.warning("Wealth or travel columns not found; using neutral vulnerability value 0.5")
            settlements_df[SET_CLIMATE_VULNERABILITY] = vulnerability_values

        # Compute Climate Priority: Hazard × Vulnerability
        # NOTE: Population is NOT included here. Population affects WHEN targets are reached
        # (cumulative % targets per timestep), not WHO should be prioritized first.
        priority_values = hazard_values * vulnerability_values
        settlements_df[SET_CLIMATE_PRIORITY] = priority_values

    if 'heatwave_risk' in settlements_df.columns:
        settlements_df[SET_CLIMATE_RISK_HEATWAVE] = settlements_df['heatwave_risk']
    if 'drought_risk' in settlements_df.columns:
        settlements_df[SET_CLIMATE_RISK_DROUGHT] = settlements_df['drought_risk']

    # Fill NaN with 0 (settlements outside coverage)
    for col in [SET_NORMALIZED_CLIMATE_HAZARD, SET_CLIMATE_HAZARD,
                SET_CLIMATE_VULNERABILITY, SET_CLIMATE_PRIORITY, SET_CLIMATE_RISK_HEATWAVE, SET_CLIMATE_RISK_DROUGHT]:
        if col in settlements_df.columns:
            settlements_df[col] = settlements_df[col].fillna(0)

    logger.info(f"Mapped climate risk to {len(settlements_df)} settlements")

    return settlements_df


# =============================================================================
# MAIN PROCESSING FUNCTION
# =============================================================================

def process_climate_data(
    climate_folder: str,
    admin3_shapefile: str,
    settlements_df: pd.DataFrame,
    specs_path: Optional[str] = None
) -> pd.DataFrame:
    """Main entry point: process climate data and add risk to settlements.

    This function orchestrates the full pipeline:
    1. Load climate configuration from specs file
    2. Classify and load climate data files by temporal resolution and data type
    3. Run appropriate analysis for each data type
    4. Calculate risk scores
    5. Map to settlements

    Args:
        climate_folder: Path to folder with climate CSV files.
        admin3_shapefile: Path to admin-3 level shapefile.
        settlements_df: OnSSET settlements DataFrame.
        specs_path: Optional path to specs Excel file for configuration.

    Returns:
        Settlements DataFrame with climate risk columns added:
        - ClimateRisk (compound)
        - ClimateRiskHeatwave (individual)
        - ClimateRiskDrought (individual)
        - Admin3ID
    """
    logger.info("=" * 60)
    logger.info("Starting climate data processing...")
    logger.info("=" * 60)

    # Step 1: Load configuration
    config = load_climate_config(specs_path)

    # Step 2: Load admin-3 boundaries
    logger.info(f"Loading admin-3 shapefile: {admin3_shapefile}")
    admin3_gdf = gpd.read_file(admin3_shapefile)
    if admin3_gdf.crs is None or admin3_gdf.crs.to_epsg() != 4326:
        admin3_gdf = admin3_gdf.to_crs(epsg=4326)
    logger.info(f"Loaded {len(admin3_gdf)} admin-3 regions")

    # Step 3: Initialize loader and classify files (modular detection)
    loader = ClimateDataLoader(climate_folder, config)

    # Get available data combinations (temporal resolution x data type)
    available = loader.get_available_combinations()
    logger.info(f"Available data combinations: {[(t.value, d.value) for t, d in available]}")

    # Get initial columns from sample
    sample_df = loader.get_sample_for_column_detection()
    detected_columns = loader.detected_columns

    heatwave_risk_df = pd.DataFrame()
    drought_risk_df = pd.DataFrame()

    # Step 4: Process temperature data for heatwave analysis
    temp_combinations = [(t, d) for t, d in available if d == ClimateDataType.TEMPERATURE]
    if temp_combinations:
        # Prefer daily for heatwave analysis
        temp_combinations.sort(key=lambda x: 0 if x[0] == TemporalResolution.DAILY else 1)
        best_temporal, _ = temp_combinations[0]

        if best_temporal == TemporalResolution.DAILY:
            logger.info("Daily temperature data detected - running incremental heatwave analysis")
            # Get temperature-specific columns
            temp_columns = loader.get_columns(ClimateDataType.TEMPERATURE)
            detected_columns.update(temp_columns)
            heatwave_risk_df = calculate_heatwave_risk_incremental(
                loader, admin3_gdf, config, detected_columns
            )
        else:
            logger.warning(f"Only {best_temporal.value} temperature data available. "
                          f"Heatwave analysis works best with daily data.")

    # Step 5: Process precipitation data for drought analysis
    precip_combinations = [(t, d) for t, d in available if d == ClimateDataType.PRECIPITATION]
    if precip_combinations:
        # Prefer monthly for SPI drought analysis
        precip_combinations.sort(key=lambda x: 0 if x[0] == TemporalResolution.MONTHLY else 1)
        best_temporal, _ = precip_combinations[0]

        if best_temporal == TemporalResolution.MONTHLY:
            logger.info("Monthly precipitation data detected - running SPI drought analysis")
            monthly_precip_df = loader.load_files(
                temporal=TemporalResolution.MONTHLY,
                datatype=ClimateDataType.PRECIPITATION
            )
            if not monthly_precip_df.empty:
                # Get precipitation-specific columns
                precip_columns = loader.get_columns(ClimateDataType.PRECIPITATION)
                detected_columns.update(precip_columns)
                drought_risk_df = calculate_spi_drought_risk(
                    monthly_precip_df, admin3_gdf, config, detected_columns
                )
                # Free memory
                del monthly_precip_df
        else:
            logger.warning(f"Only {best_temporal.value} precipitation data available. "
                          f"SPI analysis works best with monthly data.")

    # Step 6: If no classified files, fall back to legacy loading (for small datasets)
    if heatwave_risk_df.empty and drought_risk_df.empty:
        if not available or all(t == TemporalResolution.UNKNOWN for t, d in available):
            logger.warning("No classified climate files found. Attempting legacy load...")
            # Only for small datasets - this may fail for large datasets
            if len(loader.other_files) < 20:  # Safety threshold
                try:
                    climate_df = loader.load_all_files()
                    temporal_res = loader.temporal_resolution
                    detected_columns = loader.detected_columns

                    if temporal_res == TemporalResolution.DAILY:
                        heatwave_risk_df = calculate_heatwave_risk(
                            climate_df, admin3_gdf, config, detected_columns
                        )
                    elif temporal_res == TemporalResolution.MONTHLY:
                        drought_risk_df = calculate_spi_drought_risk(
                            climate_df, admin3_gdf, config, detected_columns
                        )
                except MemoryError:
                    logger.error("Memory error during legacy loading. Dataset too large.")
                    logger.error("Please ensure files follow naming conventions: "
                               "*t2m*daily*.csv for temperature, *tp*monthly*.csv for precipitation")

    # Step 7: Calculate compound risk
    compound_risk_df = calculate_compound_risk(
        heatwave_risk_df, drought_risk_df, config
    )

    # Step 8: Map to settlements
    settlements_df = map_risk_to_settlements(
        settlements_df, compound_risk_df, admin3_gdf, config
    )

    logger.info("=" * 60)
    logger.info("Climate data processing complete.")
    logger.info("=" * 60)

    return settlements_df


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_temporal_resolution() -> TemporalResolution:
    """Get the detected temporal resolution (global state)."""
    return _temporal_resolution


def get_risk_column_names() -> Dict[str, str]:
    """Get dictionary of risk column names for use in onsset.py."""
    return {
        'normalized_hazard': SET_NORMALIZED_CLIMATE_HAZARD,
        'compound': SET_CLIMATE_RISK,
        'heatwave': SET_CLIMATE_RISK_HEATWAVE,
        'drought': SET_CLIMATE_RISK_DROUGHT,
        'admin3_id': SET_ADMIN3_ID,
    }

