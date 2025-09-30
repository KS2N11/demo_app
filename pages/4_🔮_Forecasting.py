"""
Forecasting & Prediction Page for Clinical Trial Analytics
Provides advanced forecasting capabilities and scenario analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from utils.forecasting_utils import (
    detect_time_columns, detect_target_columns, prepare_time_series_data,
    forecast_with_prophet, forecast_with_arima, forecast_with_linear_trend,
    create_forecast_chart, train_prediction_model, predict_scenario,
    format_display_name
)
from services.ai_summary import generate_forecast_insights, generate_scenario_analysis

# Page configuration
st.set_page_config(
    page_title="Forecasting & Prediction - Clinical Trial Analytics",
    page_icon="🔮",
    layout="wide"
)

def main():
    st.title("🔮 Forecasting & Prediction")
    st.markdown("Advanced forecasting and scenario analysis for clinical trials")
    
    # User Guidance Section
    with st.expander("📚 **How to Use This Page - Complete Guide**", expanded=False):
        st.markdown("""
        ### 🎯 **What Can You Do Here?**
        
        This page helps you **predict the future** of your clinical trial using your current data. No technical knowledge required!
        
        #### **Three Powerful Tools:**
        
        **1. 📈 Time Series Forecasting**
        - **What it does:** Predicts future enrollment, dropouts, or any metric over time
        - **Example:** "How many participants will we have next month?"
        - **Perfect for:** Planning timelines, resource allocation, regulatory submissions
        
        **2. 🔍 Scenario Analysis** 
        - **What it does:** Shows impact of changes like "What if we add 2 new sites?"
        - **Example:** Compare current plan vs. modified recruitment strategy
        - **Perfect for:** Strategic planning, risk assessment, budget planning
        
        **3. 🎯 Custom Predictions**
        - **What it does:** Predicts outcomes based on participant characteristics
        - **Example:** "Which participants are likely to drop out?"
        - **Perfect for:** Personalized interventions, risk management
        
        ---
        
        ### 🚀 **Quick Start Guide**
        
        #### **Step 1: Time Series Forecasting**
        1. **Select Time Column** - Choose your date/time data (automatically detected)
        2. **Select Target Variable** - Pick what you want to predict (enrollment, dropouts, etc.)
        3. **Choose Forecast Horizon** - How far into the future? (7-365 days)
        4. **Pick a Model:**
           - **Prophet (Recommended)** - Best for most business data, handles seasonality
           - **ARIMA** - Good for statistical analysis, handles trends well
           - **Linear Trend** - Simple and fast, good for stable data
        5. **Click "Generate Forecast"** - Get your prediction with confidence intervals
        6. **Read AI Insights** - Plain English explanation of your results
        
        #### **Step 2: Scenario Analysis** 
        1. **Generate a baseline forecast first** (Step 1 above)
        2. **Adjust parameters:**
           - **Growth Rate** - Increase/decrease overall trend (1.0 = no change)
           - **Seasonal Boost** - Account for seasonal effects (holidays, weather)
           - **External Impact** - Factor in marketing, competition, policy changes
        3. **Click "Apply Scenario"** - See the impact
        4. **Compare results** - Side-by-side charts show the difference
        5. **Read AI Analysis** - Understand what the changes mean
        
        #### **Step 3: Custom Predictions**
        1. **Select Target Variable** - What do you want to predict?
        2. **Choose Features** - What factors influence your target?
        3. **Click "Train Model"** - System learns patterns automatically
        4. **Adjust inputs** - Use sliders to see real-time predictions
        5. **View Feature Importance** - See which factors matter most
        
        ---
        
        ### 💡 **Real-World Examples**
        
        **📊 Enrollment Planning:**
        - Upload enrollment data → Forecast: "You'll reach 500 participants in 6 months"
        - Scenario: "Adding 2 sites reduces timeline by 3 weeks"
        
        **⚠️ Risk Management:**
        - Predict dropout risk → "15% of participants may drop out by month 6"
        - Identify high-risk participants for targeted interventions
        
        **💰 Budget Planning:**
        - Forecast resource needs → "You'll need 3 more staff members by Q3"
        - Compare cost scenarios for different recruitment strategies
        
        **📅 Timeline Estimation:**
        - Predict completion dates → "Trial will complete by March 2026"
        - Account for seasonal variations and site performance
        
        ---
        
        ### ❓ **Common Questions**
        
        **Q: What if I don't understand the AI insights?**
        A: The AI explains results in simple terms. Look for key phrases like "increasing trend" or "15% higher than baseline"
        
        **Q: How accurate are the predictions?**
        A: Accuracy depends on your data quality and patterns. The system shows confidence intervals to indicate uncertainty.
        
        **Q: What if my data doesn't have obvious time patterns?**
        A: The system creates synthetic time series if needed. Focus on the trends and relative changes rather than absolute dates.
        
        **Q: Can I trust these predictions for important decisions?**
        A: Use predictions as guidance alongside your clinical expertise. They're most valuable for identifying trends and comparing scenarios.
        
        ---
        
        ### 🔧 **Troubleshooting**
        
        **"No time columns detected"**
        - System will create synthetic dates for you
        - Your data will still work for trend analysis
        
        **"Forecast generation failed"**
        - Try different aggregation method (count works best for categorical data)
        - Ensure you have enough data points (minimum 2)
        - Check for missing values in your key columns
        
        **"No suitable target columns"**
        - Any numeric column can be a target
        - Categorical columns work too (system counts occurrences)
        - Look for columns related to participants, enrollment, outcomes
        
        """)
    
    # Check if data is available
    if 'processed_data' not in st.session_state or st.session_state.processed_data is None:
        st.warning("🚨 No data available. Please go to the main page and upload your clinical trial data.")
        
        if st.button("🔙 Go to Data Upload"):
            st.switch_page("app.py")
        return
    
    df = st.session_state.processed_data
    
    # Quick Tips Sidebar
    with st.sidebar:
        st.markdown("### 💡 Quick Tips")
        st.info("""
        **Getting Started:**
        1. Start with Time Series Forecasting
        2. Choose obvious time/target columns
        3. Use Prophet model (recommended)
        4. Read AI insights for interpretation
        
        **Best Practices:**
        - More data = better predictions
        - Check confidence intervals
        - Compare multiple scenarios
        - Validate against known outcomes
        """)
        
        st.markdown("### 🆘 Need Help?")
        if st.button("📖 Show Guide Again"):
            st.rerun()
    
    # Create tabs for different sections
    tab1, tab2, tab3 = st.tabs(["📈 Time Series Forecasting", "🔍 Scenario Analysis", "🎯 Custom Predictions"])
    
    with tab1:
        st.header("📈 Time Series Forecasting")
        st.markdown("**Goal:** Predict future values based on historical patterns")
        
        # Add contextual help
        st.info("🎯 **What this does:** Analyzes your historical data to predict future trends. "
               "Perfect for planning enrollment timelines, resource needs, or completion dates.")
        
        # Auto-detect time and target columns
        time_columns = detect_time_columns(df)
        target_columns = detect_target_columns(df)
        
        if not time_columns:
            st.warning("⚠️ No time-related columns detected. Creating synthetic time series...")
            # Create a synthetic time column based on row index
            df_with_time = df.copy()
            df_with_time['synthetic_date'] = pd.date_range(start='2023-01-01', periods=len(df), freq='D')
            time_columns = [{'column': 'synthetic_date', 'type': 'datetime', 'samples': ['2023-01-01', '2023-01-02', '2023-01-03']}]
            df = df_with_time
        
        if not target_columns:
            # Use numeric columns as potential targets
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            target_columns = [{'column': col, 'type': 'continuous', 'description': f"Numeric values"} for col in numeric_cols[:5]]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("⚙️ Forecast Configuration")
            
            # Time column selection
            if time_columns:
                time_options = {f"{format_display_name(tc['column'])} ({tc['type']})": tc['column'] for tc in time_columns}
                selected_time_display = st.selectbox(
                    "Time Column",
                    list(time_options.keys()),
                    help="Select the column representing time/date"
                )
                selected_time_col = time_options[selected_time_display]
            else:
                st.error("No suitable time columns found")
                return
            
            # Target column selection
            if target_columns:
                target_options = {f"{format_display_name(tc['column'])} - {tc['description']}": tc['column'] for tc in target_columns}
                selected_target_display = st.selectbox(
                    "Target Variable",
                    list(target_options.keys()),
                    help="Select what you want to forecast"
                )
                selected_target_col = target_options[selected_target_display]
            else:
                st.error("No suitable target columns found")
                return
            
            # Forecast parameters
            forecast_horizon = st.slider("Forecast Horizon (days)", 7, 365, 30)
            aggregation_method = st.selectbox(
                "Data Aggregation",
                ["sum", "mean", "count"],
                help="How to aggregate data points for the same time period"
            )
            
            # Model selection with explanations
            available_models = ["Linear Trend"]
            
            # Debug information
            prophet_available = False
            arima_available = False
            
            try:
                from prophet import Prophet
                available_models.insert(0, "Prophet (Recommended)")
                prophet_available = True
            except ImportError as e:
                try:
                    # Try legacy import
                    from fbprophet import Prophet
                    available_models.insert(0, "Prophet (Recommended)")
                    prophet_available = True
                except ImportError:
                    pass
            
            try:
                from statsmodels.tsa.arima.model import ARIMA
                available_models.append("ARIMA")
                arima_available = True
            except ImportError:
                pass
            
            # Show debug info
            with st.expander("🔧 Debug: Available Models"):
                st.write(f"Prophet available: {prophet_available}")
                st.write(f"ARIMA available: {arima_available}")
                st.write(f"Available models: {available_models}")
                if not prophet_available:
                    st.warning("⚠️ Prophet is not available. Install with: `pip install prophet`")
            
            selected_model = st.selectbox(
                "Forecasting Model", 
                available_models,
                help="Choose your forecasting method:\n"
                     "• Prophet (Recommended): Best for business data, handles seasonality\n"
                     "• ARIMA: Good for statistical analysis, handles trends\n"
                     "• Linear Trend: Simple and fast, good for stable data"
            )
            
            # Add model explanation
            if "Prophet" in selected_model:
                st.caption("🎯 **Prophet** is recommended for most clinical trial data as it handles seasonal patterns and holidays well.")
            elif "ARIMA" in selected_model:
                st.caption("📊 **ARIMA** is great for data with clear trends and is commonly used in statistical forecasting.")
            else:
                st.caption("📈 **Linear Trend** provides simple, interpretable forecasts and works well for stable growth patterns.")
            
            generate_forecast = st.button("🔮 Generate Forecast", type="primary", help="Click to create your forecast based on the settings above")
        
        with col2:
            st.subheader("📊 Data Preview")
            
            # Show data sample
            preview_data = df[[selected_time_col, selected_target_col]].copy()
            preview_data.columns = [format_display_name(col) for col in preview_data.columns]
            st.dataframe(preview_data.head(10), use_container_width=True)
            
            # Basic statistics
            st.subheader("📈 Target Variable Stats")
            
            # Check if target column is numeric
            if pd.api.types.is_numeric_dtype(df[selected_target_col]):
                target_stats = df[selected_target_col].describe()
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("Mean", f"{target_stats['mean']:.2f}")
                    st.metric("Min", f"{target_stats['min']:.2f}")
                with col_b:
                    st.metric("Max", f"{target_stats['max']:.2f}")
                    st.metric("Std Dev", f"{target_stats['std']:.2f}")
            else:
                # For categorical columns, show different stats
                unique_count = df[selected_target_col].nunique()
                most_common = df[selected_target_col].value_counts().index[0] if len(df[selected_target_col]) > 0 else "N/A"
                missing_count = df[selected_target_col].isna().sum()
                
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("Unique Values", f"{unique_count}")
                    st.metric("Most Common", str(most_common))
                with col_b:
                    st.metric("Missing Values", f"{missing_count}")
                    st.metric("Data Type", "Categorical")
        
        # Generate and display forecast
        if generate_forecast:
            try:
                with st.spinner("🔄 Generating forecast..."):
                    # Validate data first
                    if len(df) < 2:
                        raise ValueError("Need at least 2 data points for forecasting")
                    
                    # Check if we have enough data for the selected columns
                    valid_data = df[[selected_time_col, selected_target_col]].dropna()
                    if len(valid_data) < 2:
                        raise ValueError(f"Not enough valid data points. Found {len(valid_data)} valid rows after removing missing values.")
                    
                    # Prepare time series data
                    ts_data = prepare_time_series_data(df, selected_time_col, selected_target_col, aggregation_method)
                    
                    if len(ts_data) < 2:
                        raise ValueError("Insufficient data after aggregation. Try a different aggregation method or check your data.")
                    
                    # Generate forecast based on selected model
                    if "Prophet" in selected_model:
                        forecast_result = forecast_with_prophet(ts_data, forecast_horizon)
                    elif "ARIMA" in selected_model:
                        forecast_result = forecast_with_arima(ts_data, forecast_horizon)
                    else:  # Linear Trend
                        forecast_result = forecast_with_linear_trend(ts_data, forecast_horizon)
                    
                    # Store in session state
                    st.session_state.current_forecast = forecast_result
                    st.session_state.forecast_config = {
                        'time_col': selected_time_col,
                        'target_col': selected_target_col,
                        'horizon': forecast_horizon,
                        'model': selected_model
                    }
                
                st.success("✅ Forecast generated successfully!")
                st.info("🎉 **What's next?** Scroll down to see your forecast chart and AI insights, "
                       "or try the 'Scenario Analysis' tab to explore what-if scenarios!")
                
            except Exception as e:
                st.error(f"❌ Forecast generation failed: {str(e)}")
                
                # Provide specific suggestions based on error type
                error_msg = str(e).lower()
                if "insufficient data" in error_msg or "not enough" in error_msg:
                    st.info("💡 **Suggestions:**\n"
                           "- Upload more data points\n"
                           "- Try 'count' aggregation method\n"
                           "- Check for missing values in your time/target columns")
                elif "categorical" in error_msg or "non-numeric" in error_msg:
                    st.info("💡 **Suggestions:**\n"
                           "- Try using 'count' aggregation for categorical data\n"
                           "- Select a numeric target column for sum/mean aggregation\n"
                           "- Consider creating derived numeric columns from categorical data")
                else:
                    st.info("💡 **General suggestions:**\n"
                           "- Try a different forecasting model\n"
                           "- Check your data format and column types\n"
                           "- Ensure your time column has valid dates")
                    
                # Show data info for debugging
                with st.expander("🔍 Debug Information"):
                    st.write(f"**Selected time column:** {selected_time_col}")
                    st.write(f"**Selected target column:** {selected_target_col}")
                    st.write(f"**Time column type:** {df[selected_time_col].dtype}")
                    st.write(f"**Target column type:** {df[selected_target_col].dtype}")
                    st.write(f"**Valid data points:** {len(df[[selected_time_col, selected_target_col]].dropna())}")
                    st.write(f"**Target column sample values:** {df[selected_target_col].dropna().head().tolist()}")
        
        # Display forecast results
        if 'current_forecast' in st.session_state:
            st.markdown("---")
            st.subheader("📊 Forecast Results")
            
            forecast_result = st.session_state.current_forecast
            config = st.session_state.forecast_config
            
            # Create and display chart
            chart_title = f"{format_display_name(config['target_col'])} Forecast"
            forecast_chart = create_forecast_chart(forecast_result, chart_title)
            st.plotly_chart(forecast_chart, use_container_width=True)
            
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            forecast_data = forecast_result['forecast']
            if 'yhat' in forecast_data.columns:
                with col1:
                    avg_forecast = forecast_data['yhat'].mean()
                    st.metric("Average Forecast", f"{avg_forecast:.2f}")
                
                with col2:
                    if len(forecast_data) > 1:
                        trend = "📈 Increasing" if forecast_data['yhat'].iloc[-1] > forecast_data['yhat'].iloc[0] else "📉 Decreasing"
                    else:
                        trend = "➡️ Stable"
                    st.metric("Trend Direction", trend)
                
                with col3:
                    total_forecast = forecast_data['yhat'].sum()
                    st.metric("Total Forecast", f"{total_forecast:.2f}")
                
                with col4:
                    if 'yhat_upper' in forecast_data.columns and 'yhat_lower' in forecast_data.columns:
                        avg_uncertainty = (forecast_data['yhat_upper'] - forecast_data['yhat_lower']).mean()
                        st.metric("Avg Uncertainty", f"±{avg_uncertainty:.2f}")
            
            # AI Insights
            st.subheader("🤖 AI Insights")
            with st.spinner("🧠 Generating AI insights..."):
                try:
                    insights = generate_forecast_insights(forecast_result)
                    st.markdown(insights)
                except Exception as e:
                    st.warning(f"⚠️ Could not generate AI insights: {str(e)}")
    
    with tab2:
        st.header("🔍 Scenario Analysis")
        st.markdown("**Goal:** Compare 'what-if' scenarios to understand potential impacts")
        
        # Add contextual help
        st.info("🎯 **What this does:** Takes your baseline forecast and shows how changes affect outcomes. "
               "Perfect for evaluating new strategies, site additions, or external factors.")
        
        if 'current_forecast' not in st.session_state:
            st.warning("📋 Please generate a forecast first in the 'Time Series Forecasting' tab")
            st.markdown("💡 **Why?** Scenario analysis compares changes against your baseline forecast. "
                       "Once you have a baseline, you can explore different scenarios here.")
            return
        
        # Scenario builder
        st.subheader("🛠️ Scenario Builder")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Current Scenario (Baseline)**")
            
            # Show current forecast summary
            forecast_result = st.session_state.current_forecast
            config = st.session_state.forecast_config
            
            if 'yhat' in forecast_result['forecast'].columns:
                baseline_total = forecast_result['forecast']['yhat'].sum()
                st.metric("Baseline Total Forecast", f"{baseline_total:.2f}")
        
        with col2:
            st.markdown("**Modified Scenario**")
            
            # Scenario modification controls with enhanced guidance
            st.markdown("💡 **Adjust parameters to see impact:**")
            
            growth_factor = st.slider(
                "Growth Rate Adjustment",
                0.5, 2.0, 1.0, 0.1,
                help="Multiply forecast by this factor (1.0 = no change)\n"
                     "Examples: 1.2 = 20% increase, 0.8 = 20% decrease"
            )
            st.caption(f"📊 Current setting: {((growth_factor - 1) * 100):+.0f}% change from baseline")
            
            seasonal_boost = st.slider(
                "Seasonal Boost (%)",
                -50, 100, 0, 5,
                help="Additional seasonal effect as percentage\n"
                     "Examples: +20% for holiday boost, -30% for summer slowdown"
            )
            if seasonal_boost != 0:
                st.caption(f"🗓️ Seasonal effect: {seasonal_boost:+.0f}% {'boost' if seasonal_boost > 0 else 'reduction'}")
            
            external_impact = st.slider(
                "External Impact (%)",
                -30, 50, 0, 5,
                help="Impact of external factors\n"
                     "Examples: +30% for marketing campaign, -20% for competitor entry"
            )
            if external_impact != 0:
                st.caption(f"🌐 External factor: {external_impact:+.0f}% impact")
            
            # Examples section
            with st.expander("💡 **Real-World Examples**"):
                st.markdown("""
                **Adding New Sites:**
                - Growth Rate: 1.3 (30% increase)
                - Seasonal: 0% (no change)
                - External: +10% (improved infrastructure)
                
                **Marketing Campaign:**
                - Growth Rate: 1.0 (baseline)
                - Seasonal: +25% (campaign duration)
                - External: +15% (brand awareness)
                
                **Competitive Pressure:**
                - Growth Rate: 0.9 (10% decrease)
                - Seasonal: 0% (no change)
                - External: -20% (participants choose competitors)
                
                **Economic Downturn:**
                - Growth Rate: 0.8 (20% decrease)
                - Seasonal: -15% (reduced healthcare spending)
                - External: -10% (budget constraints)
                """)
            
            apply_scenario = st.button("🔄 Apply Scenario", type="primary", 
                                     help="Calculate the impact of your parameter changes")
        
        if apply_scenario:
            try:
                # Modify forecast based on scenario parameters
                modified_forecast = forecast_result['forecast'].copy()
                
                # Apply growth factor
                modified_forecast['yhat'] = modified_forecast['yhat'] * growth_factor
                
                # Apply seasonal boost (simplified)
                seasonal_multiplier = 1 + (seasonal_boost / 100)
                modified_forecast['yhat'] = modified_forecast['yhat'] * seasonal_multiplier
                
                # Apply external impact
                external_multiplier = 1 + (external_impact / 100)
                modified_forecast['yhat'] = modified_forecast['yhat'] * external_multiplier
                
                # Update confidence intervals if present
                if 'yhat_upper' in modified_forecast.columns:
                    modified_forecast['yhat_upper'] = modified_forecast['yhat_upper'] * growth_factor * seasonal_multiplier * external_multiplier
                    modified_forecast['yhat_lower'] = modified_forecast['yhat_lower'] * growth_factor * seasonal_multiplier * external_multiplier
                
                # Store modified scenario
                st.session_state.modified_scenario = {
                    'forecast': modified_forecast,
                    'parameters': {
                        'growth_factor': growth_factor,
                        'seasonal_boost': seasonal_boost,
                        'external_impact': external_impact
                    }
                }
                
                st.success("✅ Scenario applied successfully!")
                
            except Exception as e:
                st.error(f"❌ Error applying scenario: {str(e)}")
        
        # Display scenario comparison
        if 'modified_scenario' in st.session_state:
            st.markdown("---")
            st.subheader("📊 Scenario Comparison")
            
            baseline_forecast = forecast_result['forecast']
            modified_forecast = st.session_state.modified_scenario['forecast']
            
            # Create comparison chart
            fig = go.Figure()
            
            # Baseline
            fig.add_trace(go.Scatter(
                x=baseline_forecast['ds'],
                y=baseline_forecast['yhat'],
                mode='lines',
                name='Baseline Scenario',
                line=dict(color='blue', width=2)
            ))
            
            # Modified scenario
            fig.add_trace(go.Scatter(
                x=modified_forecast['ds'],
                y=modified_forecast['yhat'],
                mode='lines',
                name='Modified Scenario',
                line=dict(color='red', width=2, dash='dash')
            ))
            
            fig.update_layout(
                title="Scenario Comparison",
                xaxis_title="Date",
                yaxis_title="Value",
                hovermode='x unified',
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Comparison metrics
            col1, col2, col3 = st.columns(3)
            
            baseline_total = baseline_forecast['yhat'].sum()
            modified_total = modified_forecast['yhat'].sum()
            difference = modified_total - baseline_total
            percentage_change = (difference / baseline_total * 100) if baseline_total != 0 else 0
            
            with col1:
                st.metric("Baseline Total", f"{baseline_total:.2f}")
            
            with col2:
                st.metric("Modified Total", f"{modified_total:.2f}")
            
            with col3:
                st.metric("Difference", f"{difference:.2f}", f"{percentage_change:+.1f}%")
            
            # AI Scenario Analysis
            st.subheader("🤖 Scenario Impact Analysis")
            with st.spinner("🧠 Analyzing scenario impact..."):
                try:
                    scenario_params = st.session_state.modified_scenario['parameters']
                    analysis = generate_scenario_analysis(baseline_total, modified_total, scenario_params)
                    st.markdown(analysis)
                except Exception as e:
                    st.warning(f"⚠️ Could not generate scenario analysis: {str(e)}")
    
    with tab3:
        st.header("🎯 Custom Predictions")
        st.markdown("**Goal:** Create predictive models based on participant characteristics")
        
        # Add contextual help
        st.info("🎯 **What this does:** Trains AI models to predict outcomes based on factors you choose. "
               "Perfect for identifying at-risk participants or predicting individual outcomes.")
        
        # Add example guidance
        with st.expander("💡 Example Use Cases"):
            st.markdown("""
            **Predict Dropout Risk:**
            - Target: dropout_risk or completion_status
            - Features: age, bmi, site_location, treatment_group
            - Result: "High-risk participants are typically older with BMI > 30"
            
            **Predict Treatment Response:**
            - Target: baseline_score or outcome_measure
            - Features: age, gender, medical_history, baseline_values
            - Result: "Younger participants with lower baseline scores respond better"
            
            **Predict Site Performance:**
            - Target: enrollment_rate or retention_rate
            - Features: site_location, staff_count, patient_demographics
            - Result: "Urban sites with more staff perform 20% better"
            """)
        
        # Model training section
        st.subheader("🤖 Model Training")
        
        # Feature and target selection
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        all_cols = numeric_cols + categorical_cols
        
        if len(all_cols) < 2:
            st.warning("⚠️ Insufficient columns for predictive modeling")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Target variable selection
            target_options = {format_display_name(col): col for col in all_cols}
            selected_target_display = st.selectbox(
                "Target Variable (What to predict)",
                list(target_options.keys()),
                help="Select the outcome you want to predict"
            )
            selected_target = target_options[selected_target_display]
            
            # Determine model type
            is_categorical_target = selected_target in categorical_cols or df[selected_target].nunique() <= 10
            model_type = 'classification' if is_categorical_target else 'regression'
            st.info(f"🔧 Model Type: {model_type.title()}")
        
        with col2:
            # Feature selection
            available_features = [col for col in all_cols if col != selected_target]
            feature_options = {format_display_name(col): col for col in available_features}
            
            selected_features_display = st.multiselect(
                "Features (Predictors)",
                list(feature_options.keys()),
                default=list(feature_options.keys())[:5],
                help="Select variables to use for prediction"
            )
            selected_features = [feature_options[feat] for feat in selected_features_display]
        
        if len(selected_features) == 0:
            st.warning("⚠️ Please select at least one feature")
            return
        
        train_model = st.button("🎯 Train Prediction Model", type="primary", 
                               help="Train an AI model to learn patterns from your data")
        
        if train_model:
            try:
                with st.spinner("🤖 Training AI model... This may take a moment"):
                    model_info = train_prediction_model(df, selected_target, selected_features, model_type)
                    st.session_state.prediction_model = model_info
                
                st.success(f"✅ Model trained successfully!")
                st.info(f"📊 **Model Performance:** {model_info['score_name']}: {model_info['score']:.3f} "
                        f"({'Higher is better' if model_info['score_name'] == 'Accuracy' else 'Lower is better'})")
                
            except Exception as e:
                st.error(f"❌ Model training failed: {str(e)}")
                st.info("💡 **Try:** Select different features, ensure target has variation, check for missing data")
        
        # Prediction interface
        if 'prediction_model' in st.session_state:
            st.markdown("---")
            st.subheader("🔮 Make Predictions")
            
            model_info = st.session_state.prediction_model
            
            # Add guidance
            st.info("🎯 **How to use:** Adjust the sliders below to see how different factors affect your prediction. "
                   "This helps you understand what drives outcomes in your data.")
            
            # Create input interface
            st.markdown("**Adjust input values to see predictions:**")
            
            scenario_values = {}
            
            # Create input controls for each feature
            for feature in model_info['feature_cols']:
                display_name = format_display_name(feature)
                
                if feature in numeric_cols:
                    min_val = float(df[feature].min())
                    max_val = float(df[feature].max())
                    default_val = float(df[feature].mean())
                    
                    scenario_values[feature] = st.slider(
                        display_name,
                        min_val, max_val, default_val,
                        key=f"pred_{feature}"
                    )
                else:
                    unique_vals = df[feature].unique().tolist()
                    scenario_values[feature] = st.selectbox(
                        display_name,
                        unique_vals,
                        key=f"pred_{feature}"
                    )
            
            predict_button = st.button("🎯 Generate Prediction")
            
            if predict_button:
                try:
                    prediction_result = predict_scenario(model_info, scenario_values)
                    
                    # Display prediction
                    st.subheader("📊 Prediction Result")
                    
                    prediction_value = prediction_result['prediction']
                    if model_type == 'classification':
                        st.success(f"🎯 **Predicted Category:** {prediction_value}")
                    else:
                        st.success(f"🎯 **Predicted Value:** {prediction_value:.3f}")
                    
                    # Feature importance
                    st.subheader("📈 Feature Importance")
                    
                    importance_data = prediction_result['feature_importance']
                    importance_df = pd.DataFrame(
                        list(importance_data.items()),
                        columns=['Feature', 'Importance']
                    ).sort_values('Importance', ascending=True)
                    
                    # Format feature names
                    importance_df['Feature'] = importance_df['Feature'].apply(format_display_name)
                    
                    fig = px.bar(
                        importance_df,
                        x='Importance',
                        y='Feature',
                        orientation='h',
                        title="Feature Importance in Prediction"
                    )
                    
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"❌ Prediction failed: {str(e)}")

if __name__ == "__main__":
    main()