import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from services.ai_summary import generate_summaries, generate_custom_insight, test_ai_connection
from utils.data_utils import get_data_summary
from utils.insights_engine import process_user_query

# Page configuration
st.set_page_config(
    page_title="AI Insights - Clinical Trial Analytics",
    page_icon="🤖",
    layout="wide"
)

def main():
    st.title("🤖 AI-Powered Insights")
    st.markdown("Get intelligent analysis and insights from your clinical trial data using AI")
    
    # Check if data is available
    if 'processed_data' not in st.session_state or st.session_state.processed_data is None:
        st.warning("🚨 No data available. Please go to the main page and upload your clinical trial data.")
        
        if st.button("🔙 Go to Data Upload"):
            st.switch_page("app.py")
        return
    
    df = st.session_state.processed_data
    metrics = st.session_state.get('metrics', {})
    
    # Test AI connection
    # st.header("🔗 AI Service Status")
    
    # col1, col2 = st.columns([3, 1])
    
    # with col1:
    #     if st.button("🔍 Test AI Connection", type="secondary"):
    #         with st.spinner("Testing AI service connection..."):
    #             status = test_ai_connection()
                
    #             if status['connected']:
    #                 st.success(f"✅ AI service connected successfully! Using {status['service_type'].upper()} API")
    #             else:
    #                 st.error(f"❌ AI service connection failed: {status.get('error', 'Unknown error')}")
    #                 st.info("💡 AI features will use template responses when service is unavailable.")
    
    # with col2:
    #     # Show current AI status
    #     if 'ai_status' in st.session_state:
    #         if st.session_state.ai_status.get('connected'):
    #             st.success("✅ Connected")
    #         else:
    #             st.warning("⚠️ Unavailable")
    
    st.markdown("---")
    
    # AI-Generated Summaries
    st.header("📋 Executive Summaries")
    st.markdown("AI-generated reports tailored for different audiences")
    
    # Generate summaries if not already available
    if 'ai_summaries' not in st.session_state or st.session_state.ai_summaries is None:
        if st.button("🚀 Generate AI Summaries", type="primary"):
            with st.spinner("Generating comprehensive AI summaries..."):
                data_summary = get_data_summary(df)
                st.session_state.ai_summaries = generate_summaries(data_summary, metrics)
            st.success("✅ AI summaries generated successfully!")
            st.rerun()
    else:
        # Display existing summaries
        summaries = st.session_state.ai_summaries
        
        # Summary tabs
        tab1, tab2, tab3 = st.tabs(["👔 Executive Summary", "🔬 Clinical Summary", "📢 Marketing Summary"])
        
        with tab1:
            st.subheader("Executive Summary")
            st.markdown("*For leadership and stakeholder communication*")
            executive_summary = summaries.get('executive', 'Summary not available')
            st.markdown(executive_summary)
            
            # Copy functionality
            # st.code(executive_summary, language=None)
            
            if 'feedback_exec' not in st.session_state:
                st.session_state.feedback_exec = ""
            
            feedback_text = st.text_area("Provide feedback for improvement (optional):", 
                                        key="feedback_input_exec", 
                                        value=st.session_state.feedback_exec,
                                        height=70)
            
            if st.button("🔄 Regenerate Executive Summary", key="regen_exec"):
                with st.spinner("Regenerating executive summary..."):
                    data_summary = get_data_summary(df)
                    new_summaries = generate_summaries(data_summary, metrics, feedback=feedback_text if feedback_text else None)
                    st.session_state.ai_summaries['executive'] = new_summaries.get('executive')
                    st.session_state.feedback_exec = ""
                st.rerun()
        
        with tab2:
            st.subheader("Clinical Summary")
            st.markdown("*For clinical teams and medical professionals*")
            clinical_summary = summaries.get('clinical', 'Summary not available')
            st.markdown(clinical_summary)
            
            # Copy functionality
            # st.code(clinical_summary, language=None)
            
            if 'feedback_clin' not in st.session_state:
                st.session_state.feedback_clin = ""
            
            feedback_text = st.text_area("Provide feedback for improvement (optional):", 
                                        key="feedback_input_clin", 
                                        value=st.session_state.feedback_clin,
                                        height=70)
            
            if st.button("🔄 Regenerate Clinical Summary", key="regen_clin"):
                with st.spinner("Regenerating clinical summary..."):
                    data_summary = get_data_summary(df)
                    new_summaries = generate_summaries(data_summary, metrics, feedback=feedback_text if feedback_text else None)
                    st.session_state.ai_summaries['clinical'] = new_summaries.get('clinical')
                    st.session_state.feedback_clin = ""
                st.rerun()
        
        with tab3:
            st.subheader("Marketing Summary")
            st.markdown("*For marketing and communications teams*")
            marketing_summary = summaries.get('marketing', 'Summary not available')
            st.markdown(marketing_summary)
            
            # Copy functionality
            # st.code(marketing_summary, language=None)
            
            if 'feedback_mark' not in st.session_state:
                st.session_state.feedback_mark = ""
            
            feedback_text = st.text_area("Provide feedback for improvement (optional):", 
                                        key="feedback_input_mark", 
                                        value=st.session_state.feedback_mark,
                                        height=70)
            
            if st.button("🔄 Regenerate Marketing Summary", key="regen_mark"):
                with st.spinner("Regenerating marketing summary..."):
                    data_summary = get_data_summary(df)
                    new_summaries = generate_summaries(data_summary, metrics, feedback=feedback_text if feedback_text else None)
                    st.session_state.ai_summaries['marketing'] = new_summaries.get('marketing')
                    st.session_state.feedback_mark = ""
                st.rerun()
        
        # Reset summaries option
        st.markdown("---")
        if 'feedback_all' not in st.session_state:
            st.session_state.feedback_all = ""
        
        feedback_text = st.text_area("Provide feedback for all summaries (optional):",
                                    key="feedback_input_all",
                                    value=st.session_state.feedback_all,
                                    height=70)
        
        if st.button("🔄 Regenerate All Summaries"):
            with st.spinner("Regenerating all AI summaries..."):
                data_summary = get_data_summary(df)
                st.session_state.ai_summaries = generate_summaries(data_summary, metrics, feedback=feedback_text if feedback_text else None)
                st.session_state.feedback_all = ""
            st.success("✅ All summaries regenerated!")
            st.rerun()
    st.markdown("---")
    
    # Interactive AI Q&A
    st.header("💬 Interactive AI Analysis")
    st.markdown("Ask questions in natural language and get AI-powered insights")
    
    # Initialize chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Query input
    query = st.text_area(
        "Ask a question about your clinical trial data:",
        placeholder="e.g., 'What are the main risk factors for participant dropout?' or 'Which sites are performing best and why?'",
        height=100,
        help="Ask complex questions that require analysis and interpretation. AI will provide detailed insights based on your data."
    )
    
    col1, col2 = st.columns([1, 4])
    
    with col1:
        if st.button("🔍 Analyze", type="primary", disabled=not query.strip()):
            if query.strip():
                with st.spinner("🤖 AI is analyzing your question..."):
                    try:
                        # Process the query
                        result = process_user_query(query, df)
                        
                        # Add to chat history
                        st.session_state.chat_history.append({
                            'query': query,
                            'result': result,
                            'timestamp': pd.Timestamp.now()
                        })
                        
                        st.success("✅ Analysis complete!")
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"❌ Error during analysis: {str(e)}")
    
    with col2:
        if st.button("🗑️ Clear History"):
            st.session_state.chat_history = []
            st.rerun()
    
    # Display chat history
    if st.session_state.chat_history:
        st.subheader("💭 Analysis History")
        
        # Show recent analyses (most recent first)
        for i, chat in enumerate(reversed(st.session_state.chat_history[-5:])):  # Show last 5
            with st.expander(f"❓ {chat['query'][:100]}..." if len(chat['query']) > 100 else f"❓ {chat['query']}", expanded=i==0):
                st.markdown(f"**Question:** {chat['query']}")
                
                result = chat['result']
                
                if result.get('success', True):
                    st.markdown(f"**🤖 AI Analysis:**")
                    
                    if result['type'] == 'chart':
                        st.plotly_chart(result['content'], use_container_width=True)
                    elif result['type'] == 'table':
                        st.dataframe(result['content'], use_container_width=True)
                    elif result['type'] == 'text':
                        st.markdown(result['content'])
                    
                    if 'explanation' in result:
                        st.info(f"💡 **Insights:** {result['explanation']}")
                    
                    if 'title' in result:
                        st.caption(f"📊 {result['title']}")
                else:
                    st.error(f"❌ Analysis failed: {result.get('content', 'Unknown error')}")
                
                st.caption(f"⏰ {chat['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
    
    st.markdown("---")
    
    # Quick Analysis Suggestions
    st.header("💡 Quick Analysis Suggestions")
    st.markdown("Click on these suggested analyses to get instant insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Data Exploration")
        
        suggestions = [
            "Show me the age distribution by site",
            "Which site has the highest eligibility rate?",
            "What's the correlation between age and dropout risk?",
            "Compare enrollment rates across sites"
        ]
        
        for suggestion in suggestions:
            if st.button(f"🔍 {suggestion}", key=f"suggest1_{suggestion}"):
                with st.spinner("🤖 Running suggested analysis..."):
                    result = process_user_query(suggestion, df)
                    
                    st.session_state.chat_history.append({
                        'query': suggestion,
                        'result': result,
                        'timestamp': pd.Timestamp.now()
                    })
                st.rerun()
    
    with col2:
        st.subheader("🎯 Performance Analysis")
        
        performance_suggestions = [
            "Identify patterns in dropout risk factors",
            "Show participant flow by enrollment date",
            "Analyze protocol deviation patterns",
            "Compare demographic characteristics by risk level"
        ]
        
        for suggestion in performance_suggestions:
            if st.button(f"📈 {suggestion}", key=f"suggest2_{suggestion}"):
                with st.spinner("🤖 Running performance analysis..."):
                    result = process_user_query(suggestion, df)
                    
                    st.session_state.chat_history.append({
                        'query': suggestion,
                        'result': result,
                        'timestamp': pd.Timestamp.now()
                    })
                st.rerun()
    
    # Tips for better queries
    with st.expander("💡 Tips for Better AI Interactions"):
        st.markdown("""
        **For best results with AI analysis:**
        
        ✅ **Good Questions:**
        - "What factors correlate with high dropout risk?"
        - "Show me enrollment trends over time by site"
        - "Which age groups have the lowest eligibility rates?"
        - "Compare site performance metrics"
        
        ❌ **Less Effective:**
        - "Show data" (too vague)
        - "Make a chart" (no specific request)
        - "Help" (not specific)
        
        **Types of Analysis Available:**
        - 📊 **Charts & Visualizations:** Age distributions, site comparisons, risk analysis
        - 📋 **Data Tables:** Participant lists, site statistics, demographic breakdowns  
        - 📝 **Text Analysis:** Summaries, insights, recommendations
        - 🔍 **Pattern Recognition:** Correlations, trends, anomalies
        """)

if __name__ == "__main__":
    main()
