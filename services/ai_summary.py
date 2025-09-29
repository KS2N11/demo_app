import openai
import streamlit as st
from typing import Dict, Any, Optional
from config import (
    OPENAI_API_KEY, OPENAI_MODEL,
    EXECUTIVE_SUMMARY_PROMPT, CLINICAL_SUMMARY_PROMPT, MARKETING_SUMMARY_PROMPT,
    INSIGHT_GENERATION_PROMPT, TEMPLATE_RESPONSES, ERROR_MESSAGES
)

def setup_openai_client():
    """
    Set up OpenAI client
    
    Returns:
        Configured OpenAI client or None if configuration is missing
    """
    try:
        if OPENAI_API_KEY:
            client = openai.OpenAI(api_key=OPENAI_API_KEY)
            return client, "openai"
        else:
            st.warning("⚠️ No OpenAI API key found. AI features will use template responses.")
            return None, None
            
    except Exception as e:
        st.warning(f"⚠️ Error setting up OpenAI client: {str(e)}. Using template responses.")
        return None, None

def call_llm(client, client_type: str, messages: list, max_tokens: int = 1000) -> Optional[str]:
    """
    Make a call to the OpenAI Language Model
    
    Args:
        client: OpenAI client instance
        client_type: "openai"
        messages: List of message dictionaries
        max_tokens: Maximum tokens in response
        
    Returns:
        Generated text or None if error
    """
    try:
        if client_type == "openai":
            response = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.7
            )
        else:
            return None
            
        return response.choices[0].message.content
        
    except Exception as e:
        st.warning(f"⚠️ AI service error: {str(e)}. Using fallback response.")
        return None

def generate_summaries(data_summary: str, metrics: Dict[str, Any], feedback: Optional[str] = None) -> Dict[str, str]:
    """
    Generate AI-powered summaries for different audiences
    
    Args:
        data_summary: Text summary of the data
        metrics: Dictionary of computed metrics
        
    Returns:
        Dictionary with summary types as keys and generated content as values
    """
    summaries = {}
    client, client_type = setup_openai_client()
    
    # Prepare metrics string
    metrics_str = "\\n".join([f"{key}: {value}" for key, value in metrics.items() if isinstance(value, (int, float, str))])
    
    summary_types = {
        "executive": EXECUTIVE_SUMMARY_PROMPT,
        "clinical": CLINICAL_SUMMARY_PROMPT,
        "marketing": MARKETING_SUMMARY_PROMPT
    }
    
    for summary_type, prompt_template in summary_types.items():
        try:
            if client and client_type:
                # Use AI to generate summary
                prompt = prompt_template.format(
                    data_summary=data_summary,
                    metrics=metrics_str
                )
                if feedback:
                    prompt += f"\n\nUser Feedback: {feedback}\n\nPlease regenerate the summary taking this feedback into account."
                    
                messages = [
                    {"role": "system", "content": "You are an expert clinical research analyst."},
                    {"role": "user", "content": prompt}
                ]
                
                ai_response = call_llm(client, client_type, messages, max_tokens=800)
                
                if ai_response:
                    summaries[summary_type] = ai_response
                else:
                    # Use template response
                    summaries[summary_type] = get_template_summary(summary_type, metrics)
            else:
                # Use template response when AI is not available
                summaries[summary_type] = get_template_summary(summary_type, metrics)
                
        except Exception as e:
            # Fallback to template
            summaries[summary_type] = get_template_summary(summary_type, metrics)
            st.warning(f"Using template for {summary_type} summary due to error: {str(e)}")
    
    return summaries

def get_template_summary(summary_type: str, metrics: Dict[str, Any]) -> str:
    """
    Generate template-based summary when AI is not available
    
    Args:
        summary_type: Type of summary (executive, clinical, marketing)
        metrics: Dictionary of computed metrics
        
    Returns:
        Formatted template summary
    """
    try:
        template = TEMPLATE_RESPONSES.get(f'{summary_type}_summary', "Summary not available")
        
        # Format template with available metrics
        format_dict = {
            'total_participants': metrics.get('total_participants', 0),
            'site_count': metrics.get('total_sites', 0),
            'eligibility_rate': f"{metrics.get('eligibility_rate', 0):.1f}",
            'high_risk_percent': f"{metrics.get('high_risk_percent', 0):.1f}",
            'medium_risk_percent': f"{metrics.get('medium_risk_percent', 0):.1f}",
            'avg_age': f"{metrics.get('avg_age', 0):.1f}",
            'high_risk_count': metrics.get('high_risk_count', 0),
            'medium_high_risk_count': metrics.get('high_risk_count', 0) + metrics.get('medium_risk_count', 0),
            'top_site_rate': f"{metrics.get('best_site_rate', 0):.1f}",
            'top_sites': metrics.get('best_performing_site', 'Top performing sites')
        }
        
        return template.format(**format_dict)
        
    except Exception as e:
        return f"Summary generation error: {str(e)}"

def generate_custom_insight(query: str, analysis_results: str, data_context: str) -> str:
    """
    Generate custom insights based on user query and analysis results
    
    Args:
        query: User's original query
        analysis_results: Results from data analysis
        data_context: Context about the data
        
    Returns:
        Generated insight text
    """
    client, client_type = setup_openai_client()
    
    try:
        if client and client_type:
            prompt = INSIGHT_GENERATION_PROMPT.format(
                query=query,
                results=analysis_results,
                context=data_context
            )
            
            messages = [
                {"role": "system", "content": "You are an expert clinical research data analyst providing actionable insights."},
                {"role": "user", "content": prompt}
            ]
            
            ai_response = call_llm(client, client_type, messages, max_tokens=600)
            
            if ai_response:
                return ai_response
        
        # Fallback response
        return generate_template_insight(query, analysis_results)
        
    except Exception as e:
        return generate_template_insight(query, analysis_results)

def generate_template_insight(query: str, analysis_results: str) -> str:
    """
    Generate template-based insight when AI is not available
    
    Args:
        query: User's original query
        analysis_results: Results from analysis
        
    Returns:
        Template-based insight
    """
    return f"""
    **Analysis Results for: "{query}"**
    
    Based on the data analysis:
    
    {analysis_results}
    
    **Key Recommendations:**
    • Review the data patterns shown in the visualization above
    • Consider focusing on areas with the highest impact
    • Monitor trends over time for continuous improvement
    • Implement targeted interventions based on these findings
    
    *Note: This is a template response. Configure AI services for detailed insights.*
    """

def test_ai_connection() -> Dict[str, Any]:
    """
    Test AI service connection and return status
    
    Returns:
        Dictionary with connection status and details
    """
    client, client_type = setup_openai_client()
    
    result = {
        'connected': False,
        'service_type': None,
        'error': None
    }
    
    if not client:
        result['error'] = 'No API configuration found'
        return result
    
    try:
        # Test with a simple query
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, this is a connection test. Please respond with 'Connection successful'."}
        ]
        
        response = call_llm(client, client_type, messages, max_tokens=50)
        
        if response:
            result['connected'] = True
            result['service_type'] = client_type
        else:
            result['error'] = 'No response from AI service'
            
    except Exception as e:
        result['error'] = str(e)
    
    return result
