import pandas as pd
import numpy as np
import streamlit as st
from typing import Dict, Any, List, Tuple
from config import DATA_VALIDATION, ERROR_MESSAGES

def clean_and_validate_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Clean and validate uploaded data
    
    Args:
        df: Raw dataframe from uploaded CSV
        
    Returns:
        Tuple of (cleaned_df, list_of_warnings)
    """
    warnings = []
    cleaned_df = df.copy()
    
    # Check required columns
    required_cols = DATA_VALIDATION['required_columns']
    missing_cols = [col for col in required_cols if col not in cleaned_df.columns]
    
    if missing_cols:
        error_msg = ERROR_MESSAGES['missing_columns'].format(columns=', '.join(missing_cols))
        st.error(error_msg)
        raise ValueError(error_msg)
    
    # Clean participant_id
    if 'participant_id' in cleaned_df.columns:
        cleaned_df['participant_id'] = cleaned_df['participant_id'].astype(str)
        if cleaned_df['participant_id'].duplicated().any():
            warnings.append("Duplicate participant IDs found and removed")
            cleaned_df = cleaned_df.drop_duplicates(subset=['participant_id'])
    
    # Clean and validate age
    if 'age' in cleaned_df.columns:
        # Convert to numeric, replacing non-numeric values with NaN
        cleaned_df['age'] = pd.to_numeric(cleaned_df['age'], errors='coerce')
        
        # Remove rows with invalid ages
        age_min, age_max = DATA_VALIDATION['age_range']
        invalid_ages = (cleaned_df['age'] < age_min) | (cleaned_df['age'] > age_max) | cleaned_df['age'].isna()
        
        if invalid_ages.sum() > 0:
            warnings.append(f"Removed {invalid_ages.sum()} rows with invalid ages")
            cleaned_df = cleaned_df[~invalid_ages]
    
    # Clean meets_criteria column
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
    
    # Remove rows with too many missing required values
    required_cols_present = cleaned_df[required_cols].notna().sum(axis=1)
    min_required = len(required_cols) - 1  # Allow 1 missing required field
    insufficient_data = required_cols_present < min_required
    
    if insufficient_data.sum() > 0:
        warnings.append(f"Removed {insufficient_data.sum()} rows with insufficient required data")
        cleaned_df = cleaned_df[~insufficient_data]
    
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
    Compute key metrics from the cleaned data
    
    Args:
        df: Cleaned dataframe
        
    Returns:
        Dictionary of computed metrics
    """
    metrics = {}
    
    # Basic counts
    metrics['total_participants'] = len(df)
    metrics['total_sites'] = df['location'].nunique() if 'location' in df.columns else 0
    
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
