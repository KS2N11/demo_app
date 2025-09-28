import os
from typing import Dict, Any

# API Configuration
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2023-05-15")
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-35-turbo")

# OpenAI Configuration (fallback)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")

# Color schemes for charts
CHART_COLORS = {
    'primary': '#1f77b4',
    'secondary': '#ff7f0e',
    'success': '#2ca02c',
    'danger': '#d62728',
    'warning': '#ff7f0e',
    'info': '#17a2b8',
    'light': '#f8f9fa',
    'dark': '#343a40'
}

# Chart color palettes
COLOR_PALETTES = {
    'default': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'],
    'professional': ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#592941', '#F2CC8F'],
    'clinical': ['#0077BE', '#00A651', '#FFB81C', '#E31837', '#702F8A', '#F79100'],
    'risk': ['#2ca02c', '#ff7f0e', '#d62728']  # Green, Orange, Red
}

# Data validation settings
DATA_VALIDATION = {
    'required_columns': ['participant_id', 'age', 'location', 'meets_criteria', 'dropout_risk'],
    'optional_columns': ['gender', 'enrollment_date', 'ethnicity', 'bmi', 'protocol_deviation', 'followup_status'],
    'age_range': (18, 100),
    'valid_risk_levels': ['Low', 'Medium', 'High', 'low', 'medium', 'high', '1', '2', '3'],
    'valid_boolean_values': ['True', 'False', 'true', 'false', 'Yes', 'No', 'yes', 'no', '1', '0', 1, 0, True, False]
}

# AI Prompt Templates
EXECUTIVE_SUMMARY_PROMPT = """
You are a clinical research executive analyzing participant data. Create a concise executive summary focusing on:

1. Key Performance Indicators (KPIs)
2. Strategic insights and recommendations
3. Risk factors and mitigation strategies
4. Business impact and next steps

Data Summary:
{data_summary}

Key Metrics:
{metrics}

Provide a professional executive summary in 3-4 paragraphs.
"""

CLINICAL_SUMMARY_PROMPT = """
You are a clinical research specialist analyzing participant data. Create a detailed clinical summary focusing on:

1. Patient safety and compliance
2. Protocol adherence and deviations
3. Risk assessment and monitoring
4. Clinical recommendations

Data Summary:
{data_summary}

Key Metrics:
{metrics}

Provide a comprehensive clinical analysis in 3-4 paragraphs.
"""

MARKETING_SUMMARY_PROMPT = """
You are a clinical trial recruitment specialist analyzing participant data. Create a marketing-focused summary covering:

1. Recruitment performance and optimization
2. Site performance and geographic insights
3. Demographic analysis and targeting
4. Retention strategies and improvements

Data Summary:
{data_summary}

Key Metrics:
{metrics}

Provide actionable marketing insights in 3-4 paragraphs.
"""

QUERY_ANALYSIS_PROMPT = """
You are an expert data analyst. Analyze the user's query and determine the best way to respond with the available clinical trial data.

User Query: {query}

Available Data Columns: {columns}
Data Summary: {data_summary}

Determine:
1. What type of analysis is needed (chart, table, summary, comparison)
2. Which columns should be used
3. What chart type would be most appropriate (if applicable)
4. Any grouping or aggregation needed

Respond in JSON format:
{{
    "analysis_type": "chart|table|summary|comparison",
    "chart_type": "bar|line|scatter|pie|histogram|box|heatmap",
    "x_column": "column_name or null",
    "y_column": "column_name or null",
    "group_column": "column_name or null",
    "aggregation": "count|sum|mean|median|min|max",
    "title": "Descriptive title for the analysis",
    "explanation": "Brief explanation of what this analysis shows"
}}
"""

INSIGHT_GENERATION_PROMPT = """
You are an expert clinical research data analyst. Based on the user's query and the data analysis results, provide meaningful insights.

User Query: {query}
Analysis Results: {results}
Data Context: {context}

Provide:
1. Key findings from the analysis
2. Clinical or operational implications
3. Actionable recommendations
4. Any notable patterns or concerns

Keep the response concise and focused on actionable insights.
"""

# Template responses when AI is not available
TEMPLATE_RESPONSES = {
    'executive_summary': """
    **Executive Summary**
    
    Based on the clinical trial data analysis:
    
    • **Enrollment Performance**: {total_participants} participants across {site_count} sites
    • **Eligibility Rate**: {eligibility_rate}% of screened participants meet study criteria
    • **Risk Distribution**: {high_risk_percent}% high-risk, {medium_risk_percent}% medium-risk participants
    • **Site Performance**: Top performing sites show {top_site_rate}% eligibility rates
    
    **Key Recommendations**: Focus recruitment efforts on high-performing sites and implement additional screening protocols for risk mitigation.
    """,
    
    'clinical_summary': """
    **Clinical Analysis**
    
    Patient safety and protocol compliance overview:
    
    • **Demographics**: Average participant age {avg_age} years
    • **Risk Assessment**: {high_risk_count} participants identified as high dropout risk
    • **Protocol Compliance**: Monitoring required for {medium_high_risk_count} medium-to-high risk participants
    • **Geographic Distribution**: Participants enrolled across {site_count} clinical sites
    
    **Clinical Recommendations**: Implement enhanced monitoring protocols for high-risk participants and ensure consistent data collection across all sites.
    """,
    
    'marketing_summary': """
    **Recruitment Analysis**
    
    Marketing and recruitment performance insights:
    
    • **Recruitment Efficiency**: {eligibility_rate}% conversion from screening to enrollment
    • **Site Performance**: {top_sites} demonstrate highest recruitment success
    • **Target Demographics**: Average participant age {avg_age}, diverse geographic representation
    • **Retention Focus**: {high_risk_count} participants require retention strategies
    
    **Marketing Recommendations**: Expand recruitment in high-performing regions and develop targeted retention programs for at-risk participants.
    """
}

# Application settings
APP_SETTINGS = {
    'max_file_size': 200,  # MB
    'supported_formats': ['.csv'],
    'chart_height': 400,
    'chart_width': 600,
    'default_page_size': 50
}

# Error messages
ERROR_MESSAGES = {
    'file_too_large': 'File size exceeds maximum limit of {max_size}MB',
    'invalid_format': 'Please upload a CSV file',
    'missing_columns': 'Required columns missing: {columns}',
    'invalid_data': 'Invalid data detected in column: {column}',
    'api_error': 'AI service temporarily unavailable. Using template response.',
    'processing_error': 'Error processing data. Please check your file format.'
}