import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import os

# Local imports
from utils.data_utils import clean_and_validate_data, compute_metrics, get_data_summary, validate_file_upload
from utils.chart_utils import create_eligibility_funnel, create_dropout_risk_chart, create_site_distribution
from services.ai_summary import generate_summaries, test_ai_connection
from utils.insights_engine import process_user_query

# Page configuration
st.set_page_config(
    page_title="Clinical Trial Analytics Demo",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
with open('assets/styles.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if 'data_uploaded' not in st.session_state:
        st.session_state.data_uploaded = False
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = None
    if 'metrics' not in st.session_state:
        st.session_state.metrics = None
    if 'ai_summaries' not in st.session_state:
        st.session_state.ai_summaries = None

def load_sample_data():
    """Load sample data if available"""
    sample_file = 'data/participants_sample.csv'
    if os.path.exists(sample_file):
        return pd.read_csv(sample_file)
    return None

def show_upload_section():
    """Display file upload section"""
    st.header("📁 Data Upload")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Choose a CSV file with participant data",
            type="csv",
            help="Upload a CSV file with participant data. Required columns: participant_id, age, location, meets_criteria, dropout_risk"
        )
        
        if uploaded_file is not None:
            # Validate file
            is_valid, error_msg = validate_file_upload(uploaded_file)
            
            if not is_valid:
                st.error(error_msg)
                return
            
            # Process uploaded file
            try:
                with st.spinner("Processing uploaded data..."):
                    df = pd.read_csv(uploaded_file)
                    cleaned_df, warnings = clean_and_validate_data(df)
                    
                    if warnings:
                        for warning in warnings:
                            st.warning(f"⚠️ {warning}")
                    
                    st.session_state.processed_data = cleaned_df
                    st.session_state.metrics = compute_metrics(cleaned_df)
                    st.session_state.data_uploaded = True
                    
                    st.success(f"✅ Successfully processed {len(cleaned_df)} participant records!")
                    
                    # Show data preview
                    st.subheader("Data Preview")
                    st.dataframe(cleaned_df.head(10), use_container_width=True)
                    
            except Exception as e:
                st.error(f"❌ Error processing file: {str(e)}")
                st.session_state.data_uploaded = False
    
    with col2:
        st.subheader("Sample Data")
        
        if st.button("📊 Load Sample Data", type="primary"):
            sample_df = load_sample_data()
            
            if sample_df is not None:
                try:
                    cleaned_df, warnings = clean_and_validate_data(sample_df)
                    
                    st.session_state.processed_data = cleaned_df
                    st.session_state.metrics = compute_metrics(cleaned_df)
                    st.session_state.data_uploaded = True
                    
                    st.success("✅ Sample data loaded successfully!")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Error loading sample data: {str(e)}")
            else:
                st.error("❌ Sample data not found. Run 'python data/generate_data.py' first.")
        
        st.info("💡 **Tip:** Use sample data to explore features without uploading your own file.")

def show_dashboard():
    """Display main dashboard with metrics and charts"""
    if not st.session_state.data_uploaded or st.session_state.processed_data is None:
        st.warning("🚨 Please upload data first to see the dashboard.")
        return
    
    df = st.session_state.processed_data
    metrics = st.session_state.metrics
    
    st.header("📊 Dashboard Overview")
    
    # Key Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Participants", 
            f"{metrics.get('total_participants', 0):,}",
            help="Total number of participants in the study"
        )
    
    with col2:
        st.metric(
            "Study Sites", 
            f"{metrics.get('total_sites', 0):,}",
            help="Number of clinical trial sites"
        )
    
    with col3:
        eligibility_rate = metrics.get('eligibility_rate', 0)
        st.metric(
            "Eligibility Rate", 
            f"{eligibility_rate:.1f}%",
            help="Percentage of screened participants who meet study criteria"
        )
    
    with col4:
        high_risk_count = metrics.get('high_risk_count', 0)
        st.metric(
            "High Risk Participants", 
            f"{high_risk_count:,}",
            help="Number of participants with high dropout risk"
        )
    
    st.markdown("---")
    
    # Charts Row
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Eligibility Funnel")
        funnel_chart = create_eligibility_funnel(df)
        st.plotly_chart(funnel_chart, use_container_width=True)
    
    with col2:
        st.subheader("⚠️ Dropout Risk Distribution")
        risk_chart = create_dropout_risk_chart(df)
        st.plotly_chart(risk_chart, use_container_width=True)
    
    # Site Distribution
    st.subheader("🏢 Site Distribution")
    site_chart = create_site_distribution(df)
    st.plotly_chart(site_chart, use_container_width=True)
    
    # AI Summaries
    st.subheader("🤖 AI-Powered Insights")
    
    if st.session_state.ai_summaries is None:
        with st.spinner("Generating AI summaries..."):
            data_summary = get_data_summary(df)
            st.session_state.ai_summaries = generate_summaries(data_summary, metrics)
    
    # Display summaries in tabs
    if st.session_state.ai_summaries:
        tab1, tab2, tab3 = st.tabs(["👔 Executive Summary", "🔬 Clinical Summary", "📢 Marketing Summary"])
        
        with tab1:
            st.markdown(st.session_state.ai_summaries.get('executive', 'Summary not available'))
        
        with tab2:
            st.markdown(st.session_state.ai_summaries.get('clinical', 'Summary not available'))
        
        with tab3:
            st.markdown(st.session_state.ai_summaries.get('marketing', 'Summary not available'))

def show_interactive_qa():
    """Display interactive Q&A section"""
    if not st.session_state.data_uploaded or st.session_state.processed_data is None:
        st.warning("🚨 Please upload data first to use the interactive Q&A.")
        return
    
    df = st.session_state.processed_data
    
    st.header("💬 Ask Questions About Your Data")
    st.markdown("Ask questions in natural language and get insights about your clinical trial data.")
    
    # Query input
    query = st.text_input(
        "Enter your question:",
        placeholder="e.g., 'Show me the age distribution by site' or 'Which site has the highest dropout risk?'",
        help="Ask about participant demographics, site performance, risk analysis, or any other aspect of your data."
    )
    
    if query:
        with st.spinner("Analyzing your question..."):
            try:
                result = process_user_query(query, df)
                
                if result.get('success', True):
                    st.subheader(result.get('title', 'Analysis Result'))
                    
                    if result['type'] == 'chart':
                        st.plotly_chart(result['content'], use_container_width=True)
                    elif result['type'] == 'table':
                        st.dataframe(result['content'], use_container_width=True)
                    else:  # text
                        st.markdown(result['content'])
                    
                    if 'explanation' in result:
                        st.info(f"💡 **Insight:** {result['explanation']}")
                else:
                    st.error(f"❌ {result.get('content', 'Analysis failed')}")
                    
            except Exception as e:
                st.error(f"❌ Error processing question: {str(e)}")
    
    # Example queries
    st.markdown("### 💡 Example Questions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Demographics:**
        - Show age distribution
        - Gender breakdown by site
        - Average age by risk level
        """)
    
    with col2:
        st.markdown("""
        **Performance:**
        - Site with most participants
        - Eligibility rates by location
        - Dropout risk analysis
        """)

def main():
    initialize_session_state()
    
    st.title("🏥 Clinical Trial Analytics Demo")
    st.markdown("Comprehensive analytics for clinical trial participant data with AI-powered insights")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("🎛️ Navigation")
        
        page = st.selectbox(
            "Choose a section:",
            ["📁 Data Upload", "📊 Dashboard", "💬 Interactive Q&A", "⚙️ Settings"],
            help="Navigate between different sections of the application"
        )
        
        st.markdown("---")
        
        # Data status
        if st.session_state.data_uploaded:
            st.success("✅ Data loaded successfully")
            if st.session_state.processed_data is not None:
                st.info(f"📊 {len(st.session_state.processed_data)} participants")
        else:
            st.warning("⚠️ No data loaded")
        
        st.markdown("---")
        
        # AI Status
        if st.button("🔗 Test AI Connection"):
            with st.spinner("Testing AI connection..."):
                status = test_ai_connection()
                if status['connected']:
                    st.success(f"✅ AI service connected ({status['service_type']})")
                else:
                    st.warning(f"⚠️ AI service unavailable: {status.get('error', 'Unknown error')}")
        
        st.markdown("---")
        
        # Reset button
        if st.button("🔄 Reset Application", type="secondary"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
    
    # Main content area
    if page == "📁 Data Upload":
        show_upload_section()
    elif page == "📊 Dashboard":
        show_dashboard()
    elif page == "💬 Interactive Q&A":
        show_interactive_qa()
    elif page == "⚙️ Settings":
        show_settings()

def show_settings():
    """Display application settings"""
    st.header("⚙️ Application Settings")
    
    st.subheader("🤖 AI Configuration")
    
    # AI service status
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Azure OpenAI:**")
        azure_configured = bool(os.getenv("AZURE_OPENAI_ENDPOINT") and os.getenv("AZURE_OPENAI_API_KEY"))
        st.write("✅ Configured" if azure_configured else "❌ Not configured")
    
    with col2:
        st.markdown("**OpenAI:**")
        openai_configured = bool(os.getenv("OPENAI_API_KEY"))
        st.write("✅ Configured" if openai_configured else "❌ Not configured")
    
    if not azure_configured and not openai_configured:
        st.warning("""
        ⚠️ **No AI services configured.** 
        
        To enable AI features, set environment variables:
        - For Azure OpenAI: `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`
        - For OpenAI: `OPENAI_API_KEY`
        """)
    
    st.markdown("---")
    
    st.subheader("📊 Data Information")
    
    if st.session_state.data_uploaded and st.session_state.processed_data is not None:
        df = st.session_state.processed_data
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Dataset Info:**")
            st.write(f"Rows: {len(df):,}")
            st.write(f"Columns: {len(df.columns)}")
            st.write(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024:.1f} KB")
        
        with col2:
            st.markdown("**Data Quality:**")
            missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
            st.write(f"Missing values: {missing_pct:.1f}%")
            st.write(f"Duplicates: {df.duplicated().sum()}")
            
        st.markdown("**Column Details:**")
        col_info = pd.DataFrame({
            'Column': df.columns,
            'Type': df.dtypes,
            'Non-null Count': df.count(),
            'Missing %': ((len(df) - df.count()) / len(df) * 100).round(1)
        })
        st.dataframe(col_info, use_container_width=True)
    else:
        st.info("Upload data to see dataset information.")
    
    # Sidebar
    with st.sidebar:
        st.header("Navigation")
        st.markdown("### Steps:")
        st.markdown("1. 📁 Upload your data")
        st.markdown("2. 📊 View dashboard")
        st.markdown("3. 🤖 Ask AI insights")
        st.markdown("4. 📈 Generate custom charts")
        
        st.markdown("---")
        
        # Sample data option
        if st.button("🎯 Load Sample Data"):
            sample_data = load_sample_data()
            if sample_data is not None:
                st.session_state.processed_data = sample_data
                st.session_state.data_uploaded = True
                st.session_state.metrics = compute_metrics(sample_data)
                st.success("Sample data loaded!")
                st.experimental_rerun()
    
    # Main content
    if not st.session_state.data_uploaded:
        show_upload_section()
    else:
        show_dashboard()

def load_sample_data():
    """Load sample data for demo purposes"""
    try:
        sample_path = "data/participants_sample.csv"
        if os.path.exists(sample_path):
            df = pd.read_csv(sample_path)
            return clean_and_validate_data(df)
        else:
            # Generate sample data if file doesn't exist
            from data.generate_data import generate_synthetic_data
            df = generate_synthetic_data(200)
            return clean_and_validate_data(df)
    except Exception as e:
        st.error(f"Error loading sample data: {e}")
        return None

def show_upload_section():
    """Display file upload section"""
    st.header("📁 Upload Participant Data")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### Expected CSV Format:
        - `participant_id`: Unique identifier
        - `age`: Participant age
        - `location`: Site/location
        - `meets_criteria`: Boolean (True/False or 1/0)
        - `dropout_risk`: Risk level (High/Medium/Low or numeric)
        - Additional columns are welcome!
        """)
    
    with col2:
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="Upload your clinical trial participant data"
        )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"File uploaded successfully! {len(df)} rows loaded.")
            
            # Preview data
            st.subheader("Data Preview")
            st.dataframe(df.head(10))
            
            # Process data
            with st.spinner("Processing data..."):
                processed_df = clean_and_validate_data(df)
                metrics = compute_metrics(processed_df)
                
                # Store in session state
                st.session_state.processed_data = processed_df
                st.session_state.metrics = metrics
                st.session_state.data_uploaded = True
                
                st.success("Data processed successfully!")
                st.experimental_rerun()
                
        except Exception as e:
            st.error(f"Error processing file: {e}")
            st.info("Please check your file format and try again.")

def show_dashboard():
    """Display the main dashboard"""
    st.header("📊 Clinical Trial Dashboard")
    
    df = st.session_state.processed_data
    metrics = st.session_state.metrics
    
    # Key Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Participants",
            f"{metrics['total_participants']:,}",
            delta=None
        )
    
    with col2:
        st.metric(
            "Eligibility Rate",
            f"{metrics['eligibility_rate']:.1f}%",
            delta=f"{metrics['eligibility_rate'] - 75:.1f}% vs target"
        )
    
    with col3:
        st.metric(
            "High Dropout Risk",
            f"{metrics['high_risk_count']:,}",
            delta=f"{metrics['high_risk_percentage']:.1f}% of total"
        )
    
    with col4:
        st.metric(
            "Active Sites",
            f"{metrics['unique_sites']:,}",
            delta=None
        )
    
    st.markdown("---")
    
    # Charts Section
    st.subheader("📈 Visual Analytics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Eligibility Funnel")
        funnel_fig = create_eligibility_funnel(df)
        st.plotly_chart(funnel_fig, use_container_width=True)
    
    with col2:
        st.markdown("#### Dropout Risk Distribution")
        risk_fig = create_dropout_risk_chart(df)
        st.plotly_chart(risk_fig, use_container_width=True)
    
    # Site Distribution
    st.markdown("#### Site Distribution")
    site_fig = create_site_distribution(df)
    st.plotly_chart(site_fig, use_container_width=True)
    
    # AI Summaries
    st.markdown("---")
    st.subheader("🤖 AI-Generated Insights")
    
    if st.session_state.ai_summaries is None:
        with st.spinner("Generating AI summaries..."):
            summaries = generate_summaries(df, metrics)
            st.session_state.ai_summaries = summaries
    
    summaries = st.session_state.ai_summaries
    
    tab1, tab2, tab3 = st.tabs(["Executive Summary", "Clinical Insights", "Marketing Report"])
    
    with tab1:
        st.markdown(summaries.get('executive', 'Summary not available'))
    
    with tab2:
        st.markdown(summaries.get('clinical', 'Summary not available'))
    
    with tab3:
        st.markdown(summaries.get('marketing', 'Summary not available'))
    
    # Interactive Q&A Section
    st.markdown("---")
    st.subheader("💬 Ask Questions About Your Data")
    
    user_query = st.text_input(
        "Ask a question:",
        placeholder="e.g., 'Show dropout risk by age group', 'Which site has the most eligible participants?'",
        help="Ask any question about your data and get instant insights!"
    )
    
    if user_query:
        with st.spinner("Analyzing your question..."):
            try:
                result = process_user_query(user_query, df)
                
                if result['type'] == 'chart':
                    st.plotly_chart(result['content'], use_container_width=True)
                elif result['type'] == 'text':
                    st.markdown(f"**Answer:** {result['content']}")
                elif result['type'] == 'data':
                    st.dataframe(result['content'])
                
                if 'explanation' in result:
                    st.info(f"💡 **Insight:** {result['explanation']}")
                    
            except Exception as e:
                st.error(f"Error processing query: {e}")
                st.info("Try rephrasing your question or ask something simpler.")

if __name__ == "__main__":
    main()