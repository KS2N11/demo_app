import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from utils.data_utils import compute_metrics, get_data_summary
from utils.chart_utils import create_eligibility_funnel, create_dropout_risk_chart, create_site_distribution, create_age_distribution_chart
from services.ai_summary import generate_summaries

# Page configuration
st.set_page_config(
    page_title="Dashboard - Clinical Trial Analytics",
    page_icon="📊",
    layout="wide"
)

def main():
    st.title("📊 Clinical Trial Dashboard")
    st.markdown("Comprehensive overview of your clinical trial data with key metrics and visualizations")
    
    # Check if data is available
    if 'processed_data' not in st.session_state or st.session_state.processed_data is None:
        st.warning("🚨 No data available. Please go to the main page and upload your clinical trial data.")
        
        if st.button("🔙 Go to Data Upload"):
            st.switch_page("app.py")
        return
    
    df = st.session_state.processed_data
    
    # Compute fresh metrics if not available
    if 'metrics' not in st.session_state or st.session_state.metrics is None:
        st.session_state.metrics = compute_metrics(df)
    
    metrics = st.session_state.metrics
    
    # Key Performance Indicators
    st.header("🎯 Key Performance Indicators")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_participants = metrics.get('total_participants', 0)
        st.metric(
            label="Total Participants",
            value=f"{total_participants:,}",
            help="Total number of participants enrolled in the study"
        )
    
    with col2:
        total_sites = metrics.get('total_sites', 0)
        st.metric(
            label="Study Sites",
            value=f"{total_sites:,}",
            help="Number of clinical sites participating in the study"
        )
    
    with col3:
        eligibility_rate = metrics.get('eligibility_rate', 0)
        st.metric(
            label="Eligibility Rate",
            value=f"{eligibility_rate:.1f}%",
            delta=f"{eligibility_rate - 75:.1f}% vs target",
            help="Percentage of screened participants who meet study criteria"
        )
    
    with col4:
        high_risk_count = metrics.get('high_risk_count', 0)
        high_risk_percent = metrics.get('high_risk_percent', 0)
        st.metric(
            label="High Risk Participants",
            value=f"{high_risk_count:,}",
            delta=f"{high_risk_percent:.1f}%",
            help="Number and percentage of participants with high dropout risk"
        )
    
    st.markdown("---")
    
    # Primary Charts
    st.header("📈 Primary Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Eligibility Funnel")
        funnel_chart = create_eligibility_funnel(df)
        st.plotly_chart(funnel_chart, use_container_width=True)
        
        # Add summary below chart
        eligible_count = metrics.get('eligible_participants', 0)
        st.info(f"📋 **Summary:** {eligible_count} of {total_participants} participants ({eligibility_rate:.1f}%) meet study criteria")
    
    with col2:
        st.subheader("Dropout Risk Distribution")
        risk_chart = create_dropout_risk_chart(df)
        st.plotly_chart(risk_chart, use_container_width=True)
        
        # Risk summary
        low_risk = metrics.get('low_risk_count', 0)
        medium_risk = metrics.get('medium_risk_count', 0)
        high_risk = metrics.get('high_risk_count', 0)
        st.info(f"⚠️ **Risk Breakdown:** Low: {low_risk}, Medium: {medium_risk}, High: {high_risk}")
    
    # Site Analysis
    st.subheader("🏢 Site Performance Analysis")
    site_chart = create_site_distribution(df)
    st.plotly_chart(site_chart, use_container_width=True)
    
    # Site performance table
    if 'site_performance' in metrics:
        st.subheader("📋 Site Performance Details")
        
        site_performance = pd.DataFrame.from_dict(metrics['site_performance'], orient='index')
        site_performance = site_performance.sort_values('eligibility_rate', ascending=False)
        
        # Color code the performance
        def color_performance(val):
            if val >= 80:
                color = 'background-color: #d4edda'  # Light green
            elif val >= 60:
                color = 'background-color: #fff3cd'  # Light yellow
            else:
                color = 'background-color: #f8d7da'  # Light red
            return color
        
        styled_df = site_performance.style.applymap(color_performance, subset=['eligibility_rate'])
        st.dataframe(styled_df, use_container_width=True)
    
    st.markdown("---")
    
    # Demographics Analysis
    st.header("👥 Demographics Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Age Distribution")
        age_chart = create_age_distribution_chart(df)
        st.plotly_chart(age_chart, use_container_width=True)
        
        if 'avg_age' in metrics:
            avg_age = metrics['avg_age']
            min_age = metrics.get('min_age', 0)
            max_age = metrics.get('max_age', 0)
            st.info(f"👤 **Age Stats:** Range {min_age:.0f}-{max_age:.0f} years, Average {avg_age:.1f} years")
    
    with col2:
        # Gender distribution (if available)
        if 'gender_distribution' in metrics:
            st.subheader("Gender Distribution")
            
            gender_data = metrics['gender_distribution']
            
            fig = go.Figure(data=[
                go.Pie(
                    labels=list(gender_data.keys()),
                    values=list(gender_data.values()),
                    hole=0.4
                )
            ])
            
            fig.update_layout(
                title="Gender Distribution",
                height=400,
                margin=dict(t=50, b=0, l=0, r=0)
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            # BMI distribution as alternative
            if 'bmi_categories' in metrics:
                st.subheader("BMI Categories")
                
                bmi_data = metrics['bmi_categories']
                
                fig = go.Figure(data=[
                    go.Bar(
                        x=list(bmi_data.keys()),
                        y=list(bmi_data.values())
                    )
                ])
                
                fig.update_layout(
                    title="BMI Category Distribution",
                    xaxis_title="BMI Category",
                    yaxis_title="Number of Participants",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Additional Metrics
    st.header("📊 Additional Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if 'protocol_deviations' in metrics:
            deviation_count = metrics['protocol_deviations']
            deviation_rate = metrics.get('deviation_rate', 0)
            st.metric(
                label="Protocol Deviations",
                value=f"{deviation_count:,}",
                delta=f"{deviation_rate:.1f}%",
                help="Number and rate of protocol deviations"
            )
    
    with col2:
        if 'active_participants' in metrics:
            active_count = metrics['active_participants']
            st.metric(
                label="Active Participants",
                value=f"{active_count:,}",
                help="Number of participants currently active in the study"
            )
    
    with col3:
        if 'withdrawn_participants' in metrics:
            withdrawn_count = metrics['withdrawn_participants']
            withdrawal_rate = (withdrawn_count / total_participants * 100) if total_participants > 0 else 0
            st.metric(
                label="Withdrawn",
                value=f"{withdrawn_count:,}",
                delta=f"{withdrawal_rate:.1f}%",
                help="Number and rate of withdrawn participants"
            )
    
    # Data Quality Section
    st.markdown("---")
    st.header("🔍 Data Quality Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Completeness")
        
        # Calculate completeness for each column
        completeness = {}
        for col in df.columns:
            completeness[col] = ((len(df) - df[col].isnull().sum()) / len(df)) * 100
        
        completeness_df = pd.DataFrame(list(completeness.items()), columns=['Column', 'Completeness (%)'])
        completeness_df = completeness_df.sort_values('Completeness (%)', ascending=True)
        
        # Create horizontal bar chart
        fig = go.Figure(go.Bar(
            x=completeness_df['Completeness (%)'],
            y=completeness_df['Column'],
            orientation='h',
            marker_color=['#d4edda' if x >= 95 else '#fff3cd' if x >= 80 else '#f8d7da' for x in completeness_df['Completeness (%)']]
        ))
        
        fig.update_layout(
            title="Data Completeness by Column",
            xaxis_title="Completeness (%)",
            height=300,
            margin=dict(l=150)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Data Summary")
        
        total_cells = len(df) * len(df.columns)
        missing_cells = df.isnull().sum().sum()
        completeness_overall = ((total_cells - missing_cells) / total_cells) * 100
        
        st.metric("Overall Completeness", f"{completeness_overall:.1f}%")
        st.metric("Total Records", f"{len(df):,}")
        st.metric("Total Fields", f"{len(df.columns):,}")
        st.metric("Missing Values", f"{missing_cells:,}")
        
        # Quality score
        if completeness_overall >= 95:
            quality_score = "Excellent ✅"
        elif completeness_overall >= 85:
            quality_score = "Good ✅"
        elif completeness_overall >= 70:
            quality_score = "Fair ⚠️"
        else:
            quality_score = "Poor ❌"
        
        st.metric("Data Quality", quality_score)

if __name__ == "__main__":
    main()
