import pandas as pd
import numpy as np
import streamlit as st
from typing import Dict, Any, List, Tuple
from config import DATA_VALIDATION, ERROR_MESSAGES
try:
    from fuzzywuzzy import fuzz
except ImportError:
    # Fallback if fuzzywuzzy is not available
    def fuzz_ratio(a, b):
        return 100 if a.lower() == b.lower() else 0
    
    class fuzz:
        @staticmethod
        def ratio(a, b):
            return fuzz_ratio(a, b)

def find_best_column_match(df: pd.DataFrame, target_names: List[str]) -> str:
    """
    Find the best matching column name from a list of possibilities
    
    Args:
        df: DataFrame to search in
        target_names: List of possible column names to match
        
    Returns:
        Best matching column name or None if no match found
    """
    df_columns_lower = [col.lower().strip() for col in df.columns]
    
    for target in target_names:
        target_lower = target.lower().strip()
        # Exact match
        if target_lower in df_columns_lower:
            return df.columns[df_columns_lower.index(target_lower)]
        
        # Partial match
        for i, col in enumerate(df_columns_lower):
            if target_lower in col or col in target_lower:
                return df.columns[i]
    
    return None

def auto_detect_and_standardize_columns(df: pd.DataFrame, warnings: List[str]) -> pd.DataFrame:
    """
    Auto-detect clinical trial columns and standardize names
    
    Args:
        df: Input dataframe
        warnings: List to append warnings to
        
    Returns:
        DataFrame with standardized column names
    """
    cleaned_df = df.copy()
    
    # Column mapping patterns
    column_mappings = {
        'participant_id': ['id', 'participant_id', 'patient_id', 'subject_id', 'participant', 'patient', 'subject'],
        'age': ['age', 'patient_age', 'participant_age', 'years', 'age_years'],
        'gender': ['gender', 'sex', 'male_female', 'gender_mf'],
        'location': ['location', 'site', 'center', 'facility', 'clinic', 'hospital', 'site_name'],
        'meets_criteria': ['eligible', 'eligibility', 'meets_criteria', 'qualified', 'included', 'enrollment_status'],
        'dropout_risk': ['risk', 'dropout_risk', 'risk_level', 'dropout', 'retention_risk'],
        'enrollment_date': ['date', 'enrollment_date', 'enroll_date', 'start_date', 'entry_date'],
        'bmi': ['bmi', 'body_mass_index', 'weight_height_ratio'],
        'ethnicity': ['ethnicity', 'race', 'ethnic_group'],
        'protocol_deviation': ['deviation', 'protocol_deviation', 'protocol_violation', 'non_compliance'],
        'followup_status': ['status', 'followup_status', 'follow_up', 'current_status', 'participant_status']
    }
    
    # Try to map columns automatically
    detected_mappings = {}
    for standard_name, possible_names in column_mappings.items():
        matched_col = find_best_column_match(cleaned_df, possible_names)
        if matched_col:
            detected_mappings[matched_col] = standard_name
    
    # Rename detected columns
    if detected_mappings:
        cleaned_df = cleaned_df.rename(columns=detected_mappings)
        warnings.append(f"Auto-detected columns: {', '.join([f'{k}→{v}' for k, v in detected_mappings.items()])}")
    
    # If no participant ID found, create one
    if 'participant_id' not in cleaned_df.columns:
        cleaned_df['participant_id'] = [f'P{i+1:04d}' for i in range(len(cleaned_df))]
        warnings.append("Created participant IDs automatically")
    
    # Basic data cleaning for any format
    for col in cleaned_df.columns:
        if cleaned_df[col].dtype == 'object':
            # Clean string columns
            cleaned_df[col] = cleaned_df[col].astype(str).str.strip()
            cleaned_df[col] = cleaned_df[col].replace(['nan', 'NaN', 'null', 'NULL', ''], np.nan)
    
    return cleaned_df

def clean_and_validate_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Flexibly clean and process any clinical trial CSV data
    
    Args:
        df: Raw dataframe from uploaded CSV
        
    Returns:
        Tuple of (cleaned_df, list_of_warnings)
    """
    warnings = []
    cleaned_df = df.copy()
    
    # Basic data cleaning - handle any CSV format
    if len(cleaned_df) == 0:
        raise ValueError("The uploaded file is empty or has no valid data rows.")
    
    # Auto-detect and standardize column names
    cleaned_df = auto_detect_and_standardize_columns(cleaned_df, warnings)
    
    # Clean participant_id
    if 'participant_id' in cleaned_df.columns:
        cleaned_df['participant_id'] = cleaned_df['participant_id'].astype(str)
        if cleaned_df['participant_id'].duplicated().any():
            warnings.append("Duplicate participant IDs found and removed")
            cleaned_df = cleaned_df.drop_duplicates(subset=['participant_id'])
    
    # Apply intelligent categorical conversion
    cleaned_df, conversion_results, original_column_metadata, categorical_warnings = intelligent_categorical_converter(cleaned_df)
    warnings.extend(categorical_warnings)
    
    # Store metadata in session state for frontend use
    if hasattr(st, 'session_state'):
        st.session_state.original_column_metadata = original_column_metadata
        st.session_state.conversion_results = conversion_results
    
    # Clean meets_criteria column (flexible)
    if 'meets_criteria' in cleaned_df.columns:
        cleaned_df['meets_criteria'] = standardize_boolean_column(cleaned_df['meets_criteria'])
        
    # Clean dropout_risk column
    if 'dropout_risk' in cleaned_df.columns:
        cleaned_df['dropout_risk'] = standardize_risk_column(cleaned_df['dropout_risk'])
        invalid_risk = ~cleaned_df['dropout_risk'].isin(['Low', 'Medium', 'High'])
        
        if invalid_risk.sum() > 0:
            warnings.append(f"Found {invalid_risk.sum()} invalid dropout risk values, setting to 'Medium'")
            cleaned_df.loc[invalid_risk, 'dropout_risk'] = 'Medium'
    
    # Clean location column
    if 'location' in cleaned_df.columns:
        cleaned_df['location'] = cleaned_df['location'].fillna('Unknown Site')
        cleaned_df['location'] = cleaned_df['location'].astype(str).str.strip()
    
    # Clean optional columns if present
    if 'gender' in cleaned_df.columns:
        cleaned_df['gender'] = cleaned_df['gender'].fillna('Not Specified')
        
    if 'enrollment_date' in cleaned_df.columns:
        cleaned_df['enrollment_date'] = pd.to_datetime(cleaned_df['enrollment_date'], errors='coerce')
        
    if 'bmi' in cleaned_df.columns:
        cleaned_df['bmi'] = pd.to_numeric(cleaned_df['bmi'], errors='coerce')
        # Set reasonable BMI bounds
        cleaned_df.loc[(cleaned_df['bmi'] < 10) | (cleaned_df['bmi'] > 50), 'bmi'] = np.nan
        
    if 'protocol_deviation' in cleaned_df.columns:
        cleaned_df['protocol_deviation'] = standardize_boolean_column(cleaned_df['protocol_deviation'])
    
    # Flexible validation: Only require some kind of identifier
    if 'participant_id' not in cleaned_df.columns:
        cleaned_df['participant_id'] = [f'P{i:04d}' for i in range(len(cleaned_df))]
        warnings.append("Added participant IDs as data didn't contain identifiers")
    
    # Ensure we have some data
    if len(cleaned_df) == 0:
        warnings.append("No valid data rows found after cleaning")
        # Create a minimal placeholder dataset
        cleaned_df = pd.DataFrame({
            'participant_id': ['PLACEHOLDER_001'],
            'age': [30],
            'meets_criteria': [True]
        })
        warnings.append("Created placeholder data for demonstration")
    
    # Reset index
    cleaned_df = cleaned_df.reset_index(drop=True)
    
    return cleaned_df, warnings

def intelligent_categorical_converter(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict, Dict, List[str]]:
    """
    Apply intelligent categorical conversion to ALL columns while preserving original metadata
    
    Args:
        df: Input DataFrame with categorical columns
        
    Returns:
        Tuple of (processed_df, conversion_results, original_metadata, warnings)
    """
    warnings = []
    processed_df = df.copy()
    conversion_results = {}
    original_column_metadata = {}
    
    for column in processed_df.columns:
        if processed_df[column].dtype == 'object':  # Categorical column
            # Store original data for frontend reference
            original_column_metadata[column] = {
                'original_values': processed_df[column].unique().tolist(),
                'value_counts': processed_df[column].value_counts().to_dict(),
                'column_type': 'categorical'
            }
            
            conversion_result = convert_single_categorical_column(processed_df[column], column)
            
            if conversion_result['success']:
                conversion_results[column] = conversion_result
                
                if conversion_result['conversion_method'] == 'one_hot_encoding':
                    # For one-hot encoding, keep original column AND add encoded columns
                    original_column_metadata[column]['one_hot_columns'] = list(conversion_result['one_hot_encoded'].columns)
                    original_column_metadata[column]['encoding_method'] = 'one_hot_encoding'
                    
                    # Add one-hot columns with special prefix to identify them as encoded (optimized)
                    one_hot_renamed = conversion_result['one_hot_encoded'].add_prefix('_encoded_')
                    processed_df = pd.concat([processed_df, one_hot_renamed], axis=1)
                    
                    warnings.append(f"Created one-hot encoding for '{column}' (original column preserved)")
                    
                elif conversion_result['converted_series'] is not None:
                    # Store original-to-encoded mapping
                    if conversion_result['encoding_map']:
                        original_column_metadata[column]['encoding_map'] = conversion_result['encoding_map']
                        original_column_metadata[column]['reverse_map'] = {v: k for k, v in conversion_result['encoding_map'].items()}
                    
                    original_column_metadata[column]['encoding_method'] = conversion_result['conversion_method']
                    
                    # Create a parallel encoded column instead of replacing
                    processed_df[f'_encoded_{column}'] = conversion_result['converted_series']
                    
                    # Add informative warning
                    method = conversion_result['conversion_method']
                    if method == 'age_mapping':
                        warnings.append(f"Converted age categories in '{column}' to numeric (original column preserved)")
                    elif method == 'ordinal_encoding':
                        warnings.append(f"Converted ordinal categories in '{column}' (original column preserved)")
                    elif method == 'binary_encoding':
                        warnings.append(f"Converted binary categories in '{column}' (original column preserved)")
                    elif method == 'label_encoding':
                        warnings.append(f"Applied label encoding to '{column}' (original column preserved)")
                    elif method == 'frequency_encoding':
                        warnings.append(f"Applied frequency encoding to '{column}' (original column preserved)")
    
    return processed_df, conversion_results, original_column_metadata, warnings

def standardize_boolean_column(series: pd.Series) -> pd.Series:
    """
    Standardize boolean-like values to True/False
    
    Args:
        series: Pandas series with boolean-like values
        
    Returns:
        Series with standardized boolean values
    """
    # Convert to string first for consistent processing
    str_series = series.astype(str).str.lower().str.strip()
    
    # Map various representations to boolean
    true_values = ['true', 'yes', '1', '1.0', 'y']
    false_values = ['false', 'no', '0', '0.0', 'n']
    
    result = series.copy()
    result.loc[str_series.isin(true_values)] = True
    result.loc[str_series.isin(false_values)] = False
    
    # Handle NaN values
    result = result.fillna(False)
    
    return result.astype(bool)

def standardize_risk_column(series: pd.Series) -> pd.Series:
    """
    Standardize risk level values to Low/Medium/High
    
    Args:
        series: Pandas series with risk level values
        
    Returns:
        Series with standardized risk levels
    """
    # Convert to string for consistent processing
    str_series = series.astype(str).str.lower().str.strip()
    
    # Map various representations
    risk_mapping = {
        'low': 'Low', '1': 'Low', 'l': 'Low',
        'medium': 'Medium', 'med': 'Medium', '2': 'Medium', 'm': 'Medium',
        'high': 'High', '3': 'High', 'h': 'High'
    }
    
    result = series.copy()
    for key, value in risk_mapping.items():
        result.loc[str_series == key] = value
    
    return result

def convert_single_categorical_column(series: pd.Series, column_name: str = None) -> Dict[str, Any]:
    """
    Intelligently convert categorical data to numeric using multiple techniques
    
    Args:
        series: Series containing categorical data
        column_name: Name of the column for context-aware processing
        
    Returns:
        Dictionary with conversion results and metadata
    """
    column_name = column_name or 'unknown'
    result = {
        'original_series': series,
        'converted_series': None,
        'conversion_method': None,
        'encoding_map': None,
        'one_hot_encoded': None,
        'success': False
    }
    
    # If already numeric, return as-is
    if series.dtype in ['int64', 'float64']:
        result['converted_series'] = series
        result['conversion_method'] = 'already_numeric'
        result['success'] = True
        return result
    
    # Determine the best conversion strategy based on column name and data
    column_lower = column_name.lower()
    unique_values = series.unique()
    unique_count = len(unique_values)
    
    # Strategy 1: Age-specific conversion (integrated into ordinal conversion)
    if any(keyword in column_lower for keyword in ['age', 'years', 'birth']):
        # Use ordinal conversion with age-specific mappings
        converted_series, encoding_map = convert_ordinal_categories(series, include_age_mappings=True)
        if converted_series is not None:
            result['converted_series'] = converted_series
            result['conversion_method'] = 'age_mapping'
            result['encoding_map'] = encoding_map
            result['success'] = True
            return result
    
    # Strategy 2: Ordinal/Level conversion (High/Medium/Low, etc.)
    if any(keyword in column_lower for keyword in ['risk', 'level', 'grade', 'severity', 'priority']):
        converted_series, encoding_map = convert_ordinal_categories(series)
        if converted_series is not None:
            result['converted_series'] = converted_series
            result['conversion_method'] = 'ordinal_encoding'
            result['encoding_map'] = encoding_map
            result['success'] = True
            return result
    
    # Strategy 3: Boolean conversion (Yes/No, True/False, etc.)
    if unique_count <= 2 or any(keyword in column_lower for keyword in ['eligible', 'qualified', 'pass', 'fail', 'active']):
        converted_series, encoding_map = convert_binary_categories(series)
        if converted_series is not None:
            result['converted_series'] = converted_series
            result['conversion_method'] = 'binary_encoding'
            result['encoding_map'] = encoding_map
            result['success'] = True
            return result
    
    # Strategy 4: One-Hot Encoding for nominal categories (when unique count is reasonable)
    if 2 < unique_count <= 10:
        one_hot_df = create_one_hot_encoding(series, column_name)
        if one_hot_df is not None:
            result['one_hot_encoded'] = one_hot_df
            result['conversion_method'] = 'one_hot_encoding'
            result['success'] = True
            return result
    
    # Strategy 5: Label Encoding for high-cardinality categories
    if unique_count > 10:
        converted_series, encoding_map = create_label_encoding(series)
        result['converted_series'] = converted_series
        result['conversion_method'] = 'label_encoding'
        result['encoding_map'] = encoding_map
        result['success'] = True
        return result
    
    # Strategy 6: Frequency Encoding as fallback
    converted_series = create_frequency_encoding(series)
    result['converted_series'] = converted_series
    result['conversion_method'] = 'frequency_encoding'
    result['success'] = True
    return result

# Old age function removed - now handled by comprehensive categorical converter

def convert_ordinal_categories(series: pd.Series, include_age_mappings: bool = False) -> Tuple[pd.Series, Dict]:
    """Convert ordinal categories like High/Medium/Low to numeric"""
    ordinal_mappings = {
        # Risk levels
        'low': 1, 'minimal': 1, 'minor': 1,
        'medium': 2, 'moderate': 2, 'average': 2, 'normal': 2,
        'high': 3, 'severe': 3, 'major': 3, 'critical': 4, 'extreme': 5,
        
        # Performance levels
        'poor': 1, 'bad': 1, 'weak': 1,
        'fair': 2, 'adequate': 2, 'satisfactory': 2,
        'good': 3, 'strong': 3, 'excellent': 4, 'outstanding': 5,
        
        # Priority levels
        'lowest': 1, 'lower': 2, 'normal': 3, 'higher': 4, 'highest': 5,
        
        # Numeric-like ordinals
        'first': 1, 'second': 2, 'third': 3, 'fourth': 4, 'fifth': 5,
        '1st': 1, '2nd': 2, '3rd': 3, '4th': 4, '5th': 5
    }
    
    # Add age-specific mappings if requested
    if include_age_mappings:
        age_mappings = {
            'infant': 1, 'baby': 1, 'newborn': 0.5,
            'toddler': 2, 'preschool': 4, 'child': 8, 'children': 8, 'pediatric': 8,
            'school_age': 10, 'adolescent': 15, 'teen': 16, 'teenager': 16, 'youth': 18,
            'young_adult': 25, 'adult': 35, 'middle_age': 45, 'middle_aged': 45,
            'older_adult': 65, 'elderly': 72, 'senior': 70, 'geriatric': 78,
            '0-18': 9, '18-30': 24, '30-50': 40, '50-70': 60, '70+': 75, '65+': 72,
            'under_18': 12, 'over_65': 72, '18-65': 35
        }
        ordinal_mappings.update(age_mappings)
    
    converted = series.copy()
    encoding_map = {}
    changes_made = False
    
    for idx, value in series.items():
        if pd.isna(value):
            continue
        str_val = str(value).lower().strip()
        
        if str_val in ordinal_mappings:
            numeric_val = ordinal_mappings[str_val]
            converted.iloc[idx] = numeric_val
            encoding_map[value] = numeric_val
            changes_made = True
    
    if changes_made:
        return pd.to_numeric(converted, errors='coerce'), encoding_map
    return None, None

def convert_binary_categories(series: pd.Series) -> Tuple[pd.Series, Dict]:
    """Convert binary categories to 0/1"""
    true_values = ['yes', 'true', '1', 'pass', 'qualified', 'eligible', 'active', 'positive', 'success']
    false_values = ['no', 'false', '0', 'fail', 'unqualified', 'ineligible', 'inactive', 'negative', 'failure']
    
    converted = series.copy()
    encoding_map = {}
    
    for idx, value in series.items():
        if pd.isna(value):
            continue
        str_val = str(value).lower().strip()
        
        if str_val in true_values:
            converted.iloc[idx] = 1
            encoding_map[value] = 1
        elif str_val in false_values:
            converted.iloc[idx] = 0
            encoding_map[value] = 0
    
    if len(encoding_map) > 0:
        return pd.to_numeric(converted, errors='coerce'), encoding_map
    return None, None

def create_one_hot_encoding(series: pd.Series, column_name: str) -> pd.DataFrame:
    """Create one-hot encoded columns for nominal categories"""
    try:
        # Clean the series
        clean_series = series.fillna('Missing').astype(str)
        
        # Create one-hot encoding
        one_hot = pd.get_dummies(clean_series, prefix=column_name)
        
        # Convert to integer type (0/1)
        one_hot = one_hot.astype(int)
        
        return one_hot
    except Exception:
        return None

def create_label_encoding(series: pd.Series) -> Tuple[pd.Series, Dict]:
    """Create label encoding for high-cardinality categories"""
    unique_values = series.dropna().unique()
    encoding_map = {val: idx for idx, val in enumerate(sorted(unique_values))}
    
    converted = series.map(encoding_map)
    return converted, encoding_map

def create_frequency_encoding(series: pd.Series) -> pd.Series:
    """Create frequency-based encoding"""
    frequency_map = series.value_counts().to_dict()
    return series.map(frequency_map)

def compute_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute key metrics from any clinical trial data - completely flexible
    
    Args:
        df: Input dataframe (any structure)
        
    Returns:
        Dictionary of computed metrics
    """
    metrics = {}
    
    # Basic counts - always available
    metrics['total_participants'] = len(df)
    metrics['total_columns'] = len(df.columns)
    
    # Auto-detect key columns dynamically
    location_cols = [col for col in df.columns if any(keyword in col.lower() 
                    for keyword in ['location', 'site', 'center', 'facility', 'clinic', 'hospital'])]
    if location_cols:
        location_col = location_cols[0]
        metrics['total_sites'] = df[location_col].nunique()
        metrics['location_column'] = location_col
    else:
        metrics['total_sites'] = 1
    
    # Eligibility metrics - flexible detection
    eligibility_cols = [col for col in df.columns if any(keyword in col.lower() 
                       for keyword in ['eligible', 'criteria', 'qualified', 'pass', 'meets'])]
    if eligibility_cols:
        eligibility_col = eligibility_cols[0]
        if df[eligibility_col].dtype == 'bool':
            eligible_count = df[eligibility_col].sum()
        else:
            # Handle various eligibility formats
            eligible_count = df[eligibility_col].astype(str).str.lower().isin(['yes', 'true', '1', 'pass', 'qualified']).sum()
        
        metrics['eligible_participants'] = eligible_count
        metrics['eligibility_rate'] = (eligible_count / len(df)) * 100 if len(df) > 0 else 0
        metrics['ineligible_participants'] = len(df) - eligible_count
        metrics['eligibility_column'] = eligibility_col
    
    # Risk distribution - flexible detection
    risk_cols = [col for col in df.columns if any(keyword in col.lower() 
                for keyword in ['risk', 'dropout', 'attrition', 'level'])]
    if risk_cols:
        risk_col = risk_cols[0]
        risk_counts = df[risk_col].value_counts()
        metrics['risk_distribution'] = risk_counts.to_dict()
        metrics['risk_column'] = risk_col
        
        # Try to identify high/medium/low risk categories
        high_risk_keywords = ['high', 'severe', 'critical', '3']
        medium_risk_keywords = ['medium', 'moderate', '2']
        low_risk_keywords = ['low', 'mild', '1']
        
        high_risk_count = 0
        medium_risk_count = 0
        low_risk_count = 0
        
        for category, count in risk_counts.items():
            category_str = str(category).lower()
            if any(keyword in category_str for keyword in high_risk_keywords):
                high_risk_count += count
            elif any(keyword in category_str for keyword in medium_risk_keywords):
                medium_risk_count += count
            elif any(keyword in category_str for keyword in low_risk_keywords):
                low_risk_count += count
        
        metrics['high_risk_count'] = high_risk_count
        metrics['medium_risk_count'] = medium_risk_count
        metrics['low_risk_count'] = low_risk_count
        
        total_risk = len(df[df[risk_col].notna()])
        if total_risk > 0:
            metrics['high_risk_percent'] = (high_risk_count / total_risk) * 100
            metrics['medium_risk_percent'] = (medium_risk_count / total_risk) * 100
            metrics['low_risk_percent'] = (low_risk_count / total_risk) * 100
    
    # Age/Numeric statistics - flexible detection with categorical age support
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    age_cols_numeric = [col for col in numeric_cols if 'age' in col.lower()]
    age_cols_all = [col for col in df.columns if 'age' in col.lower()]
    
    # Handle numeric age columns
    if age_cols_numeric:
        age_col = age_cols_numeric[0]
        age_data = df[age_col].dropna()
        if len(age_data) > 0:
            metrics['avg_age'] = age_data.mean()
            metrics['median_age'] = age_data.median()
            metrics['min_age'] = age_data.min()
            metrics['max_age'] = age_data.max()
            metrics['age_std'] = age_data.std()
            metrics['age_column'] = age_col
    # Handle converted categorical age columns or one-hot encoded age columns
    elif age_cols_all:
        age_col = age_cols_all[0]
        try:
            # Check if it's now numeric after conversion
            if pd.api.types.is_numeric_dtype(df[age_col]):
                age_data = df[age_col].dropna()
                if len(age_data) > 0:
                    metrics['avg_age'] = age_data.mean()
                    metrics['median_age'] = age_data.median()
                    metrics['min_age'] = age_data.min()
                    metrics['max_age'] = age_data.max()
                    metrics['age_std'] = age_data.std()
                    metrics['age_column'] = age_col
                    metrics['age_was_converted'] = True
            else:
                # Still categorical, record distribution
                age_counts = df[age_col].value_counts()
                metrics['age_categories'] = age_counts.to_dict()
                metrics['age_column'] = age_col
        except Exception:
            # Fallback for any issues
            pass
    
    # General numeric statistics for any numeric columns
    metrics['numeric_columns'] = len(numeric_cols)
    if len(numeric_cols) > 0:
        metrics['numeric_summary'] = {}
        for col in numeric_cols[:3]:  # Limit to first 3 numeric columns to avoid overwhelming
            col_data = df[col].dropna()
            if len(col_data) > 0:
                metrics['numeric_summary'][col] = {
                    'mean': col_data.mean(),
                    'median': col_data.median(),
                    'std': col_data.std()
                }
    
    # Location-based performance - flexible detection
    if location_cols and eligibility_cols:
        location_col = location_cols[0]
        eligibility_col = eligibility_cols[0]
        
        # Get participant ID column
        id_cols = [col for col in df.columns if any(keyword in col.lower() 
                  for keyword in ['id', 'participant', 'subject', 'patient'])]
        id_col = id_cols[0] if id_cols else df.columns[0]
        
        try:
            site_stats = df.groupby(location_col).agg({
                id_col: 'count'
            }).rename(columns={id_col: 'total_participants'})
            
            # Add eligibility stats if available
            if df[eligibility_col].dtype == 'bool':
                site_stats['eligible_participants'] = df.groupby(location_col)[eligibility_col].sum()
            else:
                eligible_mask = df[eligibility_col].astype(str).str.lower().isin(['yes', 'true', '1', 'pass', 'qualified'])
                site_stats['eligible_participants'] = df[eligible_mask].groupby(location_col).size().reindex(site_stats.index, fill_value=0)
            
            site_stats['eligibility_rate'] = (site_stats['eligible_participants'] / site_stats['total_participants']) * 100
            site_stats = site_stats.round(3)
            
            metrics['site_performance'] = site_stats.to_dict('index')
            
            # Best performing site
            if len(site_stats) > 0:
                best_site = site_stats['eligibility_rate'].idxmax()
                metrics['best_performing_site'] = best_site
                metrics['best_site_rate'] = site_stats.loc[best_site, 'eligibility_rate']
        except:
            pass  # Skip if grouping fails
    
    # Categorical data analysis - flexible for any categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    metrics['categorical_columns'] = len(categorical_cols)
    
    # Analyze top categorical distributions
    metrics['categorical_distributions'] = {}
    for col in categorical_cols[:5]:  # Limit to first 5 categorical columns
        if df[col].nunique() <= 20:  # Only analyze columns with reasonable number of categories
            col_counts = df[col].value_counts()
            metrics['categorical_distributions'][col] = col_counts.head(5).to_dict()
    
    # Data quality metrics
    metrics['missing_data_percent'] = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
    metrics['duplicate_rows'] = df.duplicated().sum()
    
    # Date columns analysis
    date_cols = df.select_dtypes(include=['datetime64']).columns
    metrics['date_columns'] = len(date_cols)
    
    if len(date_cols) > 0:
        date_col = date_cols[0]
        date_data = df[date_col].dropna()
        if len(date_data) > 0:
            metrics['date_range'] = {
                'earliest': date_data.min(),
                'latest': date_data.max(),
                'span_days': (date_data.max() - date_data.min()).days
            }
    
    return metrics

def get_data_summary(df: pd.DataFrame) -> str:
    """
    Generate a text summary of the data for AI processing
    
    Args:
        df: Cleaned dataframe
        
    Returns:
        String summary of the data
    """
    summary_parts = []
    
    # Basic info
    summary_parts.append(f"Dataset contains {len(df)} participants across {df['location'].nunique()} sites.")
    
    # Demographics
    if 'age' in df.columns:
        summary_parts.append(f"Age range: {df['age'].min():.0f}-{df['age'].max():.0f} years (mean: {df['age'].mean():.1f}).")
    
    if 'gender' in df.columns:
        gender_dist = df['gender'].value_counts()
        summary_parts.append(f"Gender distribution: {gender_dist.to_dict()}")
    
    # Key metrics
    if 'meets_criteria' in df.columns:
        eligibility_rate = (df['meets_criteria'].sum() / len(df)) * 100
        summary_parts.append(f"Eligibility rate: {eligibility_rate:.1f}% ({df['meets_criteria'].sum()} eligible)")
    
    if 'dropout_risk' in df.columns:
        risk_dist = df['dropout_risk'].value_counts()
        summary_parts.append(f"Risk distribution: {risk_dist.to_dict()}")
    
    # Site information
    if 'location' in df.columns:
        top_sites = df['location'].value_counts().head(3)
        summary_parts.append(f"Top sites: {', '.join([f'{site} ({count})' for site, count in top_sites.items()])}")
    
    return " ".join(summary_parts)

def filter_data(df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
    """
    Apply filters to the dataframe - handles both numeric and categorical data
    
    Args:
        df: Dataframe to filter
        filters: Dictionary of filter criteria
        
    Returns:
        Filtered dataframe
    """
    filtered_df = df.copy()
    
    for column, value in filters.items():
        # Handle special age categories filter
        if column == 'age_categories' and 'age' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['age'].isin(value)]
        elif column in filtered_df.columns and value is not None:
            if isinstance(value, list):
                filtered_df = filtered_df[filtered_df[column].isin(value)]
            elif isinstance(value, tuple) and len(value) == 2:
                # Range filter - handle both numeric and string ranges
                try:
                    # For numeric ranges
                    filtered_df = filtered_df[
                        (pd.to_numeric(filtered_df[column], errors='coerce') >= value[0]) & 
                        (pd.to_numeric(filtered_df[column], errors='coerce') <= value[1])
                    ]
                except:
                    # If numeric conversion fails, treat as categorical
                    filtered_df = filtered_df[filtered_df[column].isin(value)]
            else:
                filtered_df = filtered_df[filtered_df[column] == value]
    
    return filtered_df

def validate_file_upload(uploaded_file) -> Tuple[bool, str]:
    """
    Validate uploaded file before processing
    
    Args:
        uploaded_file: Streamlit uploaded file object
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if uploaded_file is None:
        return False, "No file uploaded"
    
    # Check file size
    max_size = 200 * 1024 * 1024  # 200 MB
    if uploaded_file.size > max_size:
        return False, ERROR_MESSAGES['file_too_large'].format(max_size=200)
    
    # Check file extension
    if not uploaded_file.name.lower().endswith('.csv'):
        return False, ERROR_MESSAGES['invalid_format']
    
    return True, ""
