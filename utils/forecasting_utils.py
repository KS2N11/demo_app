"""
Forecasting and Prediction Utilities for Clinical Trial Analytics
Provides flexible forecasting capabilities that work with any dataset structure
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    try:
        from fbprophet import Prophet
        PROPHET_AVAILABLE = True
    except ImportError:
        PROPHET_AVAILABLE = False

try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.stattools import adfuller
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, accuracy_score

def detect_time_columns(df):
    """
    Automatically detect time-related columns in the dataset
    Returns list of potential time columns with their types
    """
    time_columns = []
    
    for col in df.columns:
        # Check if column name suggests time
        time_keywords = ['date', 'time', 'week', 'month', 'year', 'day', 'enrollment', 'visit', 'period']
        if any(keyword in col.lower() for keyword in time_keywords):
            # Try to parse as datetime
            try:
                pd.to_datetime(df[col], errors='raise')
                time_columns.append({'column': col, 'type': 'datetime', 'samples': df[col].dropna().head(3).tolist()})
            except:
                # Check if it's numeric (could be week/month numbers)
                if pd.api.types.is_numeric_dtype(df[col]):
                    time_columns.append({'column': col, 'type': 'numeric_time', 'samples': df[col].dropna().head(3).tolist()})
    
    # Also check for columns that might represent sequential data
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]) and col not in [tc['column'] for tc in time_columns]:
            # Check if values are sequential or date-like
            unique_vals = sorted(df[col].dropna().unique())
            if len(unique_vals) > 3:
                # Check if it looks like years (1990-2030)
                if all(1990 <= val <= 2030 for val in unique_vals if isinstance(val, (int, float))):
                    time_columns.append({'column': col, 'type': 'year', 'samples': unique_vals[:3]})
                # Check if it looks like months (1-12)
                elif all(1 <= val <= 12 for val in unique_vals if isinstance(val, (int, float))):
                    time_columns.append({'column': col, 'type': 'month', 'samples': unique_vals[:3]})
                # Check if it looks like weeks (1-53)
                elif all(1 <= val <= 53 for val in unique_vals if isinstance(val, (int, float))):
                    time_columns.append({'column': col, 'type': 'week', 'samples': unique_vals[:3]})
    
    return time_columns

def detect_target_columns(df):
    """
    Detect potential target variables for forecasting
    Returns columns that could represent enrollment, dropouts, participants, etc.
    """
    target_columns = []
    
    # Keywords that suggest target variables
    target_keywords = [
        'participant', 'enroll', 'dropout', 'withdraw', 'complete', 'finish',
        'count', 'number', 'total', 'active', 'inactive', 'screened', 'eligible',
        'recruit', 'retention', 'visit', 'adherence', 'compliance'
    ]
    
    for col in df.columns:
        col_lower = col.lower()
        
        # Check if column name suggests a target variable
        if any(keyword in col_lower for keyword in target_keywords):
            if pd.api.types.is_numeric_dtype(df[col]):
                target_columns.append({
                    'column': col,
                    'type': 'continuous',
                    'description': f"Numeric values (Range: {df[col].min():.1f} - {df[col].max():.1f})"
                })
            else:
                # Categorical target (for classification)
                unique_vals = df[col].nunique()
                if unique_vals <= 10:  # Reasonable for classification
                    target_columns.append({
                        'column': col,
                        'type': 'categorical',
                        'description': f"Categories: {list(df[col].unique())[:5]}"
                    })
    
    return target_columns

def prepare_time_series_data(df, time_col, target_col, aggregation='sum'):
    """
    Prepare data for time series forecasting
    """
    try:
        # Make a copy to avoid modifying original
        data = df[[time_col, target_col]].copy().dropna()
        
        # Check if target column is numeric
        if not pd.api.types.is_numeric_dtype(data[target_col]):
            # For categorical targets, convert to counts
            if aggregation in ['sum', 'mean']:
                aggregation = 'count'  # Force count for categorical data
            
            # If target is boolean-like, convert to numeric
            if data[target_col].dtype == 'bool' or set(data[target_col].unique()).issubset({True, False, 'True', 'False', 1, 0, '1', '0'}):
                data[target_col] = pd.to_numeric(data[target_col], errors='coerce')
            else:
                # For other categorical data, we'll count occurrences
                # This effectively creates a frequency time series
                pass  # Will be handled by aggregation logic below
        
        # Convert time column to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(data[time_col]):
            try:
                data[time_col] = pd.to_datetime(data[time_col])
            except:
                # If conversion fails, create a synthetic time series
                data = data.sort_values(time_col)
                data[time_col] = pd.date_range(start='2023-01-01', periods=len(data), freq='D')
        
        # Aggregate data by time period
        if aggregation == 'sum' and pd.api.types.is_numeric_dtype(data[target_col]):
            ts_data = data.groupby(time_col)[target_col].sum().reset_index()
        elif aggregation == 'mean' and pd.api.types.is_numeric_dtype(data[target_col]):
            ts_data = data.groupby(time_col)[target_col].mean().reset_index()
        else:  # count - works for both numeric and categorical
            ts_data = data.groupby(time_col)[target_col].count().reset_index()
        
        # Rename columns for Prophet compatibility
        ts_data.columns = ['ds', 'y']
        
        # Sort by date
        ts_data = ts_data.sort_values('ds').reset_index(drop=True)
        
        # Ensure y column is numeric
        ts_data['y'] = pd.to_numeric(ts_data['y'], errors='coerce')
        ts_data = ts_data.dropna()
        
        if len(ts_data) == 0:
            raise ValueError("No valid data points after processing")
        
        return ts_data
    
    except Exception as e:
        raise ValueError(f"Error preparing time series data: {str(e)}")

def forecast_with_prophet(ts_data, periods=30, freq='D'):
    """
    Generate forecasts using Facebook Prophet
    """
    if not PROPHET_AVAILABLE:
        raise ImportError("Prophet is not available. Please install it using: pip install prophet")
    
    try:
        # Initialize and fit Prophet model
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            interval_width=0.8
        )
        
        model.fit(ts_data)
        
        # Create future dataframe
        future = model.make_future_dataframe(periods=periods, freq=freq)
        forecast = model.predict(future)
        
        return {
            'model': model,
            'forecast': forecast,
            'historical': ts_data,
            'method': 'Prophet'
        }
    
    except Exception as e:
        raise ValueError(f"Prophet forecasting failed: {str(e)}")

def forecast_with_arima(ts_data, periods=30):
    """
    Generate forecasts using ARIMA model
    """
    if not STATSMODELS_AVAILABLE:
        raise ImportError("Statsmodels is not available. Please install it using: pip install statsmodels")
    
    try:
        # Prepare data
        y = ts_data['y'].values
        
        # Check stationarity and difference if needed
        result = adfuller(y)
        d = 0 if result[1] <= 0.05 else 1
        
        # Fit ARIMA model (using auto parameters for simplicity)
        model = ARIMA(y, order=(1, d, 1))
        fitted_model = model.fit()
        
        # Generate forecast
        forecast_result = fitted_model.forecast(steps=periods)
        conf_int = fitted_model.get_forecast(steps=periods).conf_int()
        
        # Create forecast dataframe
        last_date = ts_data['ds'].iloc[-1]
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=periods, freq='D')
        
        forecast_df = pd.DataFrame({
            'ds': future_dates,
            'yhat': forecast_result,
            'yhat_lower': conf_int.iloc[:, 0],
            'yhat_upper': conf_int.iloc[:, 1]
        })
        
        return {
            'model': fitted_model,
            'forecast': forecast_df,
            'historical': ts_data,
            'method': 'ARIMA'
        }
    
    except Exception as e:
        raise ValueError(f"ARIMA forecasting failed: {str(e)}")

def forecast_with_linear_trend(ts_data, periods=30):
    """
    Simple linear trend forecasting as fallback
    """
    try:
        # Convert dates to numeric for regression
        ts_data['ds_numeric'] = pd.to_numeric(ts_data['ds'])
        
        # Fit linear regression
        X = ts_data[['ds_numeric']]
        y = ts_data['y']
        
        model = LinearRegression()
        model.fit(X, y)
        
        # Generate future dates
        last_date = ts_data['ds'].iloc[-1]
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=periods, freq='D')
        future_numeric = pd.to_numeric(future_dates)
        
        # Predict
        predictions = model.predict(future_numeric.values.reshape(-1, 1))
        
        # Calculate simple confidence intervals (based on residuals)
        residuals = y - model.predict(X)
        std_error = np.std(residuals)
        
        forecast_df = pd.DataFrame({
            'ds': future_dates,
            'yhat': predictions,
            'yhat_lower': predictions - 1.96 * std_error,
            'yhat_upper': predictions + 1.96 * std_error
        })
        
        return {
            'model': model,
            'forecast': forecast_df,
            'historical': ts_data,
            'method': 'Linear Trend'
        }
    
    except Exception as e:
        raise ValueError(f"Linear trend forecasting failed: {str(e)}")

def create_forecast_chart(forecast_result, title="Forecast"):
    """
    Create interactive forecast visualization
    """
    historical = forecast_result['historical']
    forecast = forecast_result['forecast']
    method = forecast_result['method']
    
    fig = go.Figure()
    
    # Historical data
    fig.add_trace(go.Scatter(
        x=historical['ds'],
        y=historical['y'],
        mode='lines+markers',
        name='Historical Data',
        line=dict(color='blue', width=2)
    ))
    
    # Forecast
    fig.add_trace(go.Scatter(
        x=forecast['ds'],
        y=forecast['yhat'],
        mode='lines',
        name=f'{method} Forecast',
        line=dict(color='red', width=2, dash='dash')
    ))
    
    # Confidence interval
    if 'yhat_upper' in forecast.columns and 'yhat_lower' in forecast.columns:
        fig.add_trace(go.Scatter(
            x=forecast['ds'].tolist() + forecast['ds'].tolist()[::-1],
            y=forecast['yhat_upper'].tolist() + forecast['yhat_lower'].tolist()[::-1],
            fill='toself',
            fillcolor='rgba(255,0,0,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='Confidence Interval',
            hoverinfo="skip"
        ))
    
    fig.update_layout(
        title=f"{title} - {method}",
        xaxis_title="Date",
        yaxis_title="Value",
        hovermode='x unified',
        height=500
    )
    
    return fig

def train_prediction_model(df, target_col, feature_cols, model_type='regression'):
    """
    Train a prediction model for scenario analysis
    """
    try:
        # Prepare data
        features = df[feature_cols].copy()
        target = df[target_col].copy()
        
        # Handle missing values
        features = features.fillna(features.mean() if features.select_dtypes(include=[np.number]).shape[1] > 0 else features.mode().iloc[0])
        target = target.fillna(target.mean() if pd.api.types.is_numeric_dtype(target) else target.mode()[0])
        
        # Encode categorical variables
        for col in features.select_dtypes(include=['object', 'category']).columns:
            features[col] = pd.Categorical(features[col]).codes
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        if model_type == 'regression':
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train_scaled, y_train)
            predictions = model.predict(X_test_scaled)
            score = mean_absolute_error(y_test, predictions)
            score_name = "MAE"
        else:
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train_scaled, y_train)
            predictions = model.predict(X_test_scaled)
            score = accuracy_score(y_test, predictions)
            score_name = "Accuracy"
        
        return {
            'model': model,
            'scaler': scaler,
            'feature_cols': feature_cols,
            'score': score,
            'score_name': score_name,
            'model_type': model_type
        }
    
    except Exception as e:
        raise ValueError(f"Model training failed: {str(e)}")

def predict_scenario(model_info, scenario_values):
    """
    Make predictions for a given scenario
    """
    try:
        # Prepare scenario data
        scenario_df = pd.DataFrame([scenario_values])
        
        # Ensure all required features are present
        for col in model_info['feature_cols']:
            if col not in scenario_df.columns:
                scenario_df[col] = 0  # Default value
        
        # Select and order features
        scenario_features = scenario_df[model_info['feature_cols']]
        
        # Scale features
        scenario_scaled = model_info['scaler'].transform(scenario_features)
        
        # Make prediction
        prediction = model_info['model'].predict(scenario_scaled)[0]
        
        # Get feature importance
        importance = model_info['model'].feature_importances_
        feature_importance = dict(zip(model_info['feature_cols'], importance))
        
        return {
            'prediction': prediction,
            'feature_importance': feature_importance
        }
    
    except Exception as e:
        raise ValueError(f"Scenario prediction failed: {str(e)}")

def format_display_name(name: str) -> str:
    """Convert column name to user-friendly format without underscores"""
    if pd.isna(name) or name is None:
        return str(name)
    
    formatted = str(name).replace('_', ' ').title()
    formatted = formatted.replace(' Id', ' ID')
    formatted = formatted.replace(' Bmi', ' BMI')
    return formatted