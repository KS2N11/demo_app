import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_sample_data(num_participants=200, output_file='participants_sample.csv'):
    """
    Generate synthetic clinical trial participant data
    
    Args:
        num_participants: Number of participants to generate
        output_file: Output CSV filename
    """
    
    # Set random seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Define data parameters
    sites = ['Site A - New York', 'Site B - Los Angeles', 'Site C - Chicago', 
             'Site D - Houston', 'Site E - Phoenix', 'Site F - Philadelphia',
             'Site G - San Antonio', 'Site H - San Diego', 'Site I - Dallas',
             'Site J - San Jose']
    
    genders = ['Male', 'Female', 'Other']
    ethnicities = ['White', 'Black or African American', 'Hispanic or Latino', 
                   'Asian', 'Native American', 'Other', 'Not Reported']
    
    followup_statuses = ['Active', 'Withdrawn', 'Lost to Follow-up', 'Completed']
    
    # Generate participant data
    data = []
    
    for i in range(num_participants):
        participant_id = f"P{i+1:04d}"
        
        # Age distribution (skewed towards older adults for clinical trial)
        age = np.random.normal(55, 15)
        age = max(18, min(85, int(age)))  # Ensure age is within reasonable bounds
        
        # Gender distribution
        gender = np.random.choice(genders, p=[0.45, 0.53, 0.02])
        
        # Location (some sites are more active)
        location = np.random.choice(sites, p=[0.15, 0.12, 0.11, 0.10, 0.09, 
                                            0.09, 0.08, 0.08, 0.09, 0.09])
        
        # Ethnicity
        ethnicity = np.random.choice(ethnicities, p=[0.60, 0.13, 0.18, 0.06, 0.01, 0.01, 0.01])
        
        # BMI (normal distribution around healthy range)
        bmi = np.random.normal(26, 4)
        bmi = max(16, min(45, round(bmi, 1)))
        
        # Enrollment date (last 6 months)
        start_date = datetime.now() - timedelta(days=180)
        end_date = datetime.now() - timedelta(days=30)
        enrollment_date = start_date + timedelta(
            days=random.randint(0, (end_date - start_date).days)
        )
        
        # Eligibility criteria (age and BMI based with some randomness)
        base_eligibility_prob = 0.75
        if age < 25 or age > 75:
            base_eligibility_prob -= 0.2
        if bmi < 18.5 or bmi > 35:
            base_eligibility_prob -= 0.15
        
        meets_criteria = np.random.choice([True, False], p=[base_eligibility_prob, 1-base_eligibility_prob])
        
        # Dropout risk (correlated with age, BMI, and site)
        risk_score = 0
        if age > 65:
            risk_score += 0.3
        if bmi > 30 or bmi < 20:
            risk_score += 0.2
        if location in ['Site E - Phoenix', 'Site I - Dallas']:  # Some sites have higher dropout
            risk_score += 0.2
        
        risk_score += np.random.normal(0, 0.2)  # Add some randomness
        
        if risk_score > 0.5:
            dropout_risk = 'High'
        elif risk_score > 0.2:
            dropout_risk = 'Medium'
        else:
            dropout_risk = 'Low'
        
        # Protocol deviation (5-10% chance)
        protocol_deviation = np.random.choice([True, False], p=[0.08, 0.92])
        
        # Follow-up status (correlated with dropout risk)
        if dropout_risk == 'High':
            status_probs = [0.60, 0.25, 0.10, 0.05]
        elif dropout_risk == 'Medium':
            status_probs = [0.80, 0.12, 0.05, 0.03]
        else:
            status_probs = [0.90, 0.05, 0.03, 0.02]
        
        followup_status = np.random.choice(followup_statuses, p=status_probs)
        
        # Add participant to dataset
        participant = {
            'participant_id': participant_id,
            'age': age,
            'gender': gender,
            'location': location,
            'ethnicity': ethnicity,
            'bmi': bmi,
            'enrollment_date': enrollment_date.strftime('%Y-%m-%d'),
            'meets_criteria': meets_criteria,
            'dropout_risk': dropout_risk,
            'protocol_deviation': protocol_deviation,
            'followup_status': followup_status
        }
        
        data.append(participant)
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(data)
    
    # Add some derived metrics for demonstration
    df['days_since_enrollment'] = (datetime.now() - pd.to_datetime(df['enrollment_date'])).dt.days
    
    # Age groups for analysis
    df['age_group'] = pd.cut(df['age'], 
                            bins=[0, 30, 45, 60, 75, 100], 
                            labels=['18-30', '31-45', '46-60', '61-75', '75+'])
    
    # BMI categories
    df['bmi_category'] = pd.cut(df['bmi'],
                               bins=[0, 18.5, 25, 30, 100],
                               labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
    
    # Save to file
    output_path = f"data/{output_file}"
    df.to_csv(output_path, index=False)
    
    print(f"Generated {num_participants} participants and saved to {output_path}")
    print(f"Data shape: {df.shape}")
    print("\nSample data preview:")
    print(df.head())
    print("\nData summary:")
    print(f"Eligibility rate: {df['meets_criteria'].mean():.2%}")
    print(f"Risk distribution: {df['dropout_risk'].value_counts().to_dict()}")
    print(f"Site distribution: {len(df['location'].unique())} sites")
    print(f"Age range: {df['age'].min()}-{df['age'].max()} years")
    
    return df

def generate_additional_scenarios():
    """Generate additional sample datasets for different scenarios"""
    
    # High-performing site scenario
    print("\nGenerating high-performing site scenario...")
    df_high = generate_sample_data(150, 'participants_high_performance.csv')
    
    # Modify to simulate better performance
    df_high.loc[df_high['location'].isin(['Site A - New York', 'Site B - Los Angeles']), 'meets_criteria'] = True
    df_high.loc[df_high['location'].isin(['Site A - New York', 'Site B - Los Angeles']), 'dropout_risk'] = 'Low'
    df_high.to_csv('data/participants_high_performance.csv', index=False)
    
    # Multi-site comparison scenario
    print("Generating multi-site comparison scenario...")
    df_multi = generate_sample_data(300, 'participants_multi_site.csv')
    df_multi.to_csv('data/participants_multi_site.csv', index=False)

if __name__ == "__main__":
    # Generate main sample dataset
    df = generate_sample_data(200, 'participants_sample.csv')
    
    # Optional: Generate additional scenarios
    # generate_additional_scenarios()
    
    print("\nData generation complete!")
    print("\nTo use this data:")
    print("1. Run the Streamlit app: streamlit run app.py")
    print("2. Use the 'Load Sample Data' button in the sidebar")
    print("3. Or upload the generated CSV file manually")
