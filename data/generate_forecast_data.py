"""
Generate sample clinical trial data for testing forecasting capabilities
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_time_series_data():
    """Generate sample time-series data for clinical trial enrollment"""
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate dates for 12 months
    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(weeks=i) for i in range(52)]
    
    # Generate enrollment data with trend and seasonality
    base_enrollment = 20
    trend = np.linspace(0, 10, 52)  # Gradual increase
    seasonality = 5 * np.sin(2 * np.pi * np.arange(52) / 52)  # Annual cycle
    noise = np.random.normal(0, 3, 52)
    
    enrollment = base_enrollment + trend + seasonality + noise
    enrollment = np.maximum(enrollment, 0)  # Ensure non-negative
    
    # Generate cumulative data
    cumulative_enrollment = np.cumsum(enrollment)
    
    # Generate dropout data
    dropout_rate = 0.05 + 0.02 * np.random.random(52)  # 5-7% dropout rate
    dropouts = np.random.poisson(cumulative_enrollment * dropout_rate)
    dropouts = np.minimum(dropouts, cumulative_enrollment)  # Can't exceed enrollment
    
    # Generate site data
    sites = ['Site_A', 'Site_B', 'Site_C', 'Site_D', 'Site_E']
    site_data = []
    
    for i, date in enumerate(dates):
        for site in sites:
            site_enrollment = max(0, enrollment[i] // len(sites) + np.random.poisson(2))
            site_dropouts = max(0, dropouts[i] // len(sites) + np.random.poisson(1))
            
            site_data.append({
                'date': date,
                'site_location': site,
                'weekly_enrollment': site_enrollment,
                'weekly_dropouts': site_dropouts,
                'active_participants': max(0, site_enrollment - site_dropouts),
                'week_number': i + 1,
                'month_number': (i // 4) + 1,
                'quarter': (i // 13) + 1
            })
    
    return pd.DataFrame(site_data)

def generate_participant_level_data():
    """Generate individual participant data"""
    
    np.random.seed(42)
    
    # Number of participants
    n_participants = 500
    
    # Generate enrollment dates (spread over time)
    start_date = datetime(2023, 1, 1)
    enrollment_dates = [start_date + timedelta(days=np.random.randint(0, 365)) for _ in range(n_participants)]
    
    participants = []
    
    for i in range(n_participants):
        # Basic demographics
        age = np.random.normal(45, 15)
        age = max(18, min(80, age))  # Constrain to reasonable range
        
        gender = np.random.choice(['Male', 'Female'], p=[0.45, 0.55])
        site = np.random.choice(['Site_A', 'Site_B', 'Site_C', 'Site_D', 'Site_E'])
        
        # Risk factors
        bmi = np.random.normal(25, 4)
        bmi = max(16, min(45, bmi))
        
        # Determine dropout risk based on factors
        dropout_score = (
            0.1 * (age - 45) / 15 +  # Age effect
            0.1 * (bmi - 25) / 4 +   # BMI effect
            0.1 * np.random.normal(0, 1)  # Random factors
        )
        
        if dropout_score > 0.3:
            dropout_risk = 'High'
        elif dropout_score > 0:
            dropout_risk = 'Medium'
        else:
            dropout_risk = 'Low'
        
        # Treatment assignment
        treatment_group = np.random.choice(['Treatment', 'Control'], p=[0.6, 0.4])
        
        # Outcome measures
        baseline_score = np.random.normal(75, 10)
        
        # Follow-up data
        days_in_study = (datetime.now() - enrollment_dates[i]).days
        is_active = days_in_study < 365 and np.random.random() > 0.15  # 15% dropout overall
        
        completion_status = 'Active' if is_active else np.random.choice(['Completed', 'Withdrawn', 'Lost to Follow-up'], p=[0.5, 0.3, 0.2])
        
        participants.append({
            'participant_id': f'P{i+1:03d}',
            'enrollment_date': enrollment_dates[i],
            'age': round(age, 1),
            'gender': gender,
            'site_location': site,
            'bmi_value': round(bmi, 1),
            'dropout_risk': dropout_risk,
            'treatment_group': treatment_group,
            'baseline_score': round(baseline_score, 1),
            'days_in_study': days_in_study,
            'completion_status': completion_status,
            'is_eligible': np.random.choice([True, False], p=[0.85, 0.15]),
            'protocol_deviations': np.random.poisson(0.3),
            'visit_adherence': min(100, max(0, np.random.normal(85, 15)))
        })
    
    return pd.DataFrame(participants)

if __name__ == "__main__":
    # Generate and save sample datasets
    
    # Time series data
    ts_data = generate_time_series_data()
    ts_data.to_csv('enrollment_time_series.csv', index=False)
    print("Generated enrollment_time_series.csv")
    
    # Participant level data
    participant_data = generate_participant_level_data()
    participant_data.to_csv('participant_forecast_data.csv', index=False)
    print("Generated participant_forecast_data.csv")
    
    print("\nSample data generated successfully!")
    print(f"Time series data: {len(ts_data)} records")
    print(f"Participant data: {len(participant_data)} records")