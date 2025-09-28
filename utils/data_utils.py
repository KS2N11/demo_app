import pandas as pd
import numpy as np
import streamlit as st
from typing import Dict, Any, List, Tuple
from config import DATA_VALIDATION, ERROR_MESSAGES

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
    
    # Clean and validate age (flexible - don't remove rows)
    if 'age' in cleaned_df.columns:
        cleaned_df['age'] = pd.to_numeric(cleaned_df['age'], errors='coerce')
        invalid_ages = cleaned_df['age'].isna().sum()
        if invalid_ages > 0:
            warnings.append(f"Found {invalid_ages} rows with invalid age data (kept as NaN)")
    
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

def compute_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute key metrics from any clinical trial data
    
    Args:
        df: Cleaned dataframe
        
    Returns:
        Dictionary of computed metrics
    """
    metrics = {}
    
    # Basic counts
    metrics['total_participants'] = len(df)
    
    # Auto-detect key columns
    location_col = find_best_column_match(df, ['location', 'site', 'center', 'facility', 'clinic'])
    metrics['total_sites'] = df[location_col].nunique() if location_col else 1
    
    # Eligibility metrics
    if 'meets_criteria' in df.columns:
        metrics['eligible_participants'] = df['meets_criteria'].sum()
        metrics['eligibility_rate'] = (df['meets_criteria'].sum() / len(df)) * 100 if len(df) > 0 else 0
        metrics['ineligible_participants'] = len(df) - df['meets_criteria'].sum()
    
    # Risk distribution
    if 'dropout_risk' in df.columns:
        risk_counts = df['dropout_risk'].value_counts()
        metrics['high_risk_count'] = risk_counts.get('High', 0)
        metrics['medium_risk_count'] = risk_counts.get('Medium', 0)
        metrics['low_risk_count'] = risk_counts.get('Low', 0)
        
        total_risk = len(df[df['dropout_risk'].notna()])
        if total_risk > 0:
            metrics['high_risk_percent'] = (metrics['high_risk_count'] / total_risk) * 100
            metrics['medium_risk_percent'] = (metrics['medium_risk_count'] / total_risk) * 100
            metrics['low_risk_percent'] = (metrics['low_risk_count'] / total_risk) * 100
    
    # Age statistics
    if 'age' in df.columns and df['age'].notna().sum() > 0:
        metrics['avg_age'] = df['age'].mean()
        metrics['median_age'] = df['age'].median()
        metrics['min_age'] = df['age'].min()
        metrics['max_age'] = df['age'].max()
        metrics['age_std'] = df['age'].std()
    
    # Site performance
    if 'location' in df.columns and 'meets_criteria' in df.columns:
        site_stats = df.groupby('location').agg({
            'participant_id': 'count',
            'meets_criteria': ['sum', 'mean']
        }).round(3)
        
        site_stats.columns = ['total_participants', 'eligible_participants', 'eligibility_rate']
        site_stats['eligibility_rate'] *= 100
        
        metrics['site_performance'] = site_stats.to_dict('index')
        
        # Best performing sites
        if len(site_stats) > 0:
            best_site = site_stats['eligibility_rate'].idxmax()
            metrics['best_performing_site'] = best_site
            metrics['best_site_rate'] = site_stats.loc[best_site, 'eligibility_rate']
    
    # Demographics
    if 'gender' in df.columns:
        gender_counts = df['gender'].value_counts()
        metrics['gender_distribution'] = gender_counts.to_dict()
    
    if 'bmi' in df.columns and df['bmi'].notna().sum() > 0:
        metrics['avg_bmi'] = df['bmi'].mean()
        metrics['bmi_categories'] = pd.cut(df['bmi'], 
                                         bins=[0, 18.5, 25, 30, 100], 
                                         labels=['Underweight', 'Normal', 'Overweight', 'Obese']).value_counts().to_dict()
    
    # Protocol deviations
    if 'protocol_deviation' in df.columns:
        metrics['protocol_deviations'] = df['protocol_deviation'].sum()
        metrics['deviation_rate'] = (df['protocol_deviation'].sum() / len(df)) * 100 if len(df) > 0 else 0
    
    # Follow-up status
    if 'followup_status' in df.columns:
        followup_counts = df['followup_status'].value_counts()
        metrics['followup_distribution'] = followup_counts.to_dict()
        metrics['active_participants'] = followup_counts.get('Active', 0)
        metrics['withdrawn_participants'] = followup_counts.get('Withdrawn', 0)
    
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
    Apply filters to the dataframe
    
    Args:
        df: Dataframe to filter
        filters: Dictionary of filter criteria
        
    Returns:
        Filtered dataframe
    """
    filtered_df = df.copy()
    
    for column, value in filters.items():
        if column in filtered_df.columns and value is not None:
            if isinstance(value, list):
                filtered_df = filtered_df[filtered_df[column].isin(value)]
            elif isinstance(value, tuple) and len(value) == 2:
                # Range filter
                filtered_df = filtered_df[
                    (filtered_df[column] >= value[0]) & 
                    (filtered_df[column] <= value[1])
                ]
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
