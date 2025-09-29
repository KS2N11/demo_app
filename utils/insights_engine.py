import pandas as pd
import numpy as np
import json
import re
from typing import Dict, Any, Tuple, List
import plotly.graph_objects as go
import plotly.express as px
from utils.chart_utils import create_custom_chart, create_age_distribution_chart, create_correlation_heatmap
from services.ai_summary import setup_openai_client, call_llm
from config import QUERY_ANALYSIS_PROMPT
import streamlit as st

def process_user_query(query: str, df: pd.DataFrame) -> Dict[str, Any]:
    """
    Process user's natural language query and return appropriate response
    
    Args:
        query: User's natural language question
        df: Cleaned dataframe
        
    Returns:
        Dictionary containing response type, content, and explanation
    """
    try:
        # Clean and normalize query
        query = query.strip().lower()
        
        # First, try pattern matching for common queries
        # pattern_result = match_common_patterns(query, df)
        # if pattern_result:
        #     print("Pattern matched result:",pattern_result)
        #     return pattern_result
        
        # If no pattern match, try AI-powered analysis
        ai_result = analyze_query_with_ai(query, df)
        print("AI analysis result:",ai_result)
        if ai_result:
            return ai_result
         
        # Fallback to simple text response
        return {
            'type': 'text',
            'content': "I understand you're asking about the data, but I couldn't determine the specific analysis needed. Could you try rephrasing your question?",
            'explanation': "Try asking specific questions like 'Show dropout risk by age group' or 'Which site has the most participants?'"
        }
        
    except Exception as e:
        return {
            'type': 'text',
            'content': f"Error processing query: {e}",
            'explanation': "Please try a simpler question."
        }

def match_common_patterns(query: str, df: pd.DataFrame) -> Dict[str, Any]:
    """
    Match common query patterns and return appropriate responses
    
    Args:
        query: User's query (lowercase)
        df: Cleaned dataframe
        
    Returns:
        Response dictionary or None if no pattern matches
    """
    try:
        # Age-related queries
        if any(word in query for word in ['age', 'older', 'younger', 'age group']):
            if 'dropout' in query or 'risk' in query:
                return create_age_risk_analysis(df)
            elif 'distribution' in query or 'histogram' in query:
                return {
                    'type': 'chart',
                    'content': create_age_distribution_chart(df),
                    'explanation': "Age distribution showing the spread of participant ages across the study."
                }
            else:
                return create_age_summary(df)
        
        # Site-related queries
        if any(word in query for word in ['site', 'location', 'center']):
            if 'best' in query or 'top' in query or 'highest' in query:
                return create_top_sites_analysis(df)
            elif 'worst' in query or 'lowest' in query or 'risk' in query:
                return create_risk_sites_analysis(df)
            else:
                return create_site_overview(df)
        
        # Dropout/Risk queries
        if any(word in query for word in ['dropout', 'risk', 'retention']):
            if 'site' in query or 'location' in query:
                return create_risk_by_site_analysis(df)
            elif 'age' in query:
                return create_age_risk_analysis(df)
            else:
                return create_overall_risk_analysis(df)
        
        # Eligibility queries
        if any(word in query for word in ['eligible', 'criteria', 'qualified', 'meet']):
            if 'site' in query:
                return create_eligibility_by_site(df)
            elif 'age' in query:
                return create_eligibility_by_age(df)
            else:
                return create_eligibility_overview(df)
        
        # Count/Number queries
        if any(word in query for word in ['how many', 'count', 'number']):
            return create_count_summary(df, query)
        
        # Comparison queries
        if any(word in query for word in ['compare', 'vs', 'versus', 'difference']):
            return create_comparison_analysis(df, query)
        
        return None
        
    except Exception as e:
        st.error(f"Error in pattern matching: {e}")
        return None

def analyze_query_with_ai(query: str, df: pd.DataFrame) -> Dict[str, Any]:
    """
    Use AI to analyze complex queries
    
    Args:
        query: User's query
        df: Cleaned dataframe
        
    Returns:
        Response dictionary or None if AI analysis fails
    """
    try:
        client, model = setup_openai_client()
        if not client or not model:
            return None
        
        # Prepare data context
        columns = list(df.columns)
        sample_data = df.head(3).to_dict('records')
        
        prompt = QUERY_ANALYSIS_PROMPT.format(
            query=query,
            columns=columns,
            data_summary=sample_data
        )
        
        # Fix: Create proper messages structure
        messages = [
            {"role": "system", "content": "You are a data analysis assistant."},
            {"role": "user", "content": prompt}
        ]
        
        response = call_llm(client, model, messages, max_tokens=300)
        
        print(f"AI Response: {response}")
        
        if not response:
            return None
        
        try:
            # Strip markdown code blocks if present
            cleaned_response = response.strip()
            if cleaned_response.startswith('```'):
                # Remove ```json or ``` at start and ``` at end
                lines = cleaned_response.split('\n')
                if lines[0].startswith('```'):
                    lines = lines[1:]  # Remove first line
                if lines[-1].strip() == '```':
                    lines = lines[:-1]  # Remove last line
                cleaned_response = '\n'.join(lines)
            
            print(f"Cleaned response: {cleaned_response}")
            
            # Try to parse JSON response
            analysis = json.loads(cleaned_response)
            print(f"Parsed analysis: {analysis}")
            
            result = execute_ai_analysis(analysis, df)
            print(f"Execution result: {result}")
            
            if result:
                return result
            else:
                print("execute_ai_analysis returned None")
                return {
                    'type': 'text',
                    'content': analysis.get('explanation', 'Analysis completed.'),
                    'explanation': "Based on your query and the available data."
                }
                
        except json.JSONDecodeError as je:
            print(f"JSON decode error: {je}")
            # If not JSON, treat as text explanation
            return {
                'type': 'text',
                'content': response,
                'explanation': "AI-generated insight based on your query."
            }
        
    except Exception as e:
        print(f"Exception in AI query analysis: {e}")
        import traceback
        traceback.print_exc()
        return None

def execute_ai_analysis(analysis: Dict[str, Any], df: pd.DataFrame) -> Dict[str, Any]:
    """
    Execute analysis based on AI-generated instructions
    
    Args:
        analysis: Parsed AI analysis instructions
        df: Cleaned dataframe
        
    Returns:
        Response dictionary
    """
    try:
        analysis_type = analysis.get('analysis_type', 'summary')
        print(f"Analysis type: {analysis_type}")
        
        # Handle chart and comparison types
        if analysis_type in ['chart', 'comparison']:
            chart_type = analysis.get('chart_type', 'bar')
            x_col = analysis.get('x_column')
            y_col = analysis.get('y_column')
            group_col = analysis.get('group_column')
            aggregation = analysis.get('aggregation')
            
            # Convert "null" strings to None
            if x_col == 'null':
                x_col = None
            if y_col == 'null':
                y_col = None
            if group_col == 'null':
                group_col = None
            
            print(f"Chart - x_col: {x_col}, y_col: {y_col}, group_col: {group_col}, aggregation: {aggregation}")
            
            # Validate x_col exists
            if x_col and x_col in df.columns:
                # If y_col doesn't exist or is for counting
                if not y_col or y_col not in df.columns:
                    # Create aggregated data
                    if aggregation == 'count':
                        agg_df = df.groupby(x_col).size().reset_index(name='count')
                        chart = create_custom_chart(agg_df, chart_type, x_col, 'count')
                    else:
                        # Use the first numeric column if available
                        numeric_cols = df.select_dtypes(include=['number']).columns
                        if len(numeric_cols) > 0:
                            y_col = numeric_cols[0]
                            chart = create_custom_chart(df, chart_type, x_col, y_col)
                        else:
                            agg_df = df.groupby(x_col).size().reset_index(name='count')
                            chart = create_custom_chart(agg_df, chart_type, x_col, 'count')
                elif y_col in df.columns:
                    # If aggregation specified, aggregate first
                    if aggregation and aggregation != 'null':
                        agg_df = df.groupby(x_col)[y_col].agg(aggregation).reset_index()
                        agg_df.columns = [x_col, f'{y_col}_{aggregation}']
                        chart = create_custom_chart(agg_df, chart_type, x_col, f'{y_col}_{aggregation}')
                    else:
                        chart = create_custom_chart(df, chart_type, x_col, y_col)
                else:
                    print(f"Column {y_col} not found")
                    return None
                
                return {
                    'type': 'chart',
                    'content': chart,
                    'explanation': analysis.get('explanation', analysis.get('title', 'Custom chart based on your query.'))
                }
            else:
                print(f"Column {x_col} not found in dataframe")
                return None
        
        elif analysis_type == 'table':
            x_col = analysis.get('x_column')
            y_col = analysis.get('y_column')
            group_col = analysis.get('group_column')
            aggregation = analysis.get('aggregation')
            
            # Convert "null" strings to None
            if x_col == 'null':
                x_col = None
            if y_col == 'null':
                y_col = None
            if group_col == 'null':
                group_col = None
            
            # Determine grouping column
            groupby_col = x_col or group_col
            
            print(f"Table - groupby_col: {groupby_col}, y_col: {y_col}, aggregation: {aggregation}")
            
            if groupby_col and groupby_col in df.columns:
                if aggregation and aggregation != 'null' and y_col and y_col in df.columns:
                    # Aggregate with specific function
                    result_df = df.groupby(groupby_col)[y_col].agg(aggregation).reset_index()
                    result_df.columns = [groupby_col, f'{y_col}_{aggregation}']
                elif aggregation == 'count' or not y_col:
                    # Simple count
                    result_df = df.groupby(groupby_col).size().reset_index(name='count')
                else:
                    # Multiple aggregations or just grouping
                    result_df = df.groupby(groupby_col).size().reset_index(name='count')
                
                print(f"Result dataframe shape: {result_df.shape}")
                
                return {
                    'type': 'table',
                    'content': result_df,
                    'explanation': analysis.get('explanation', analysis.get('title', 'Grouped data based on your query.'))
                }
            else:
                print(f"Column {groupby_col} not found in dataframe")
                return None
        
        elif analysis_type == 'summary':
            # Check if we have data parameters that we can use for analysis
            y_col = analysis.get('y_column')
            group_col = analysis.get('group_column')
            aggregation = analysis.get('aggregation')
            
            if y_col and y_col != 'null' and y_col in df.columns:
                # If we have valid columns for analysis, create a data summary
                if group_col and group_col != 'null' and group_col in df.columns:
                    # Group by the specified column
                    if aggregation and aggregation != 'null':
                        result_df = df.groupby(group_col)[y_col].agg(aggregation).reset_index()
                        result_df.columns = [group_col, f'{y_col}_{aggregation}']
                    else:
                        result_df = df.groupby(group_col)[y_col].value_counts().reset_index(name='count')
                    
                    return {
                        'type': 'table',
                        'content': result_df,
                        'explanation': analysis.get('explanation', analysis.get('title', 'Analysis completed.')),
                        'title': analysis.get('title')
                    }
                else:
                    # Just summarize the y_column
                    if aggregation and aggregation != 'null':
                        summary_data = df[y_col].agg(aggregation)
                    else:
                        summary_data = df[y_col].value_counts().reset_index()
                        summary_data.columns = [y_col, 'count']
                    
                    return {
                        'type': 'table',
                        'content': summary_data,
                        'explanation': analysis.get('explanation', analysis.get('title', 'Analysis completed.')),
                        'title': analysis.get('title')
                    }
            
            # If no valid data parameters, return text summary
            return {
                'type': 'text',
                'content': analysis.get('explanation', analysis.get('title', 'Analysis completed.')),
                'explanation': "Based on your query and the available data."
            }
        
        # Fallback
        print("Falling back to summary")
        return {
            'type': 'text',
            'content': analysis.get('explanation', analysis.get('title', 'Analysis completed.')),
            'explanation': "Based on your query and the available data."
        }
        
    except Exception as e:
        print(f"Error in execute_ai_analysis: {e}")
        import traceback
        traceback.print_exc()
        return None
# Specific analysis functions
def create_age_risk_analysis(df: pd.DataFrame) -> Dict[str, Any]:
    """Create age vs dropout risk analysis"""
    try:
        age_risk = df.groupby(['age_group', 'dropout_risk']).size().reset_index(name='count')
        
        fig = px.bar(
            age_risk, 
            x='age_group', 
            y='count', 
            color='dropout_risk',
            title='Dropout Risk by Age Group',
            color_discrete_map={'High': '#d62728', 'Medium': '#ff9800', 'Low': '#2ca02c'}
        )
        
        return {
            'type': 'chart',
            'content': fig,
            'explanation': f"Shows how dropout risk varies across age groups. Older participants may show different risk patterns."
        }
    except Exception as e:
        return {'type': 'text', 'content': f"Error creating age-risk analysis: {e}"}

def create_top_sites_analysis(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze top performing sites"""
    try:
        site_metrics = df.groupby('location').agg({
            'participant_id': 'count',
            'meets_criteria': 'sum'
        }).rename(columns={'participant_id': 'total', 'meets_criteria': 'eligible'})
        
        site_metrics['eligibility_rate'] = (site_metrics['eligible'] / site_metrics['total']) * 100
        top_sites = site_metrics.sort_values('eligibility_rate', ascending=False)
        
        return {
            'type': 'table',
            'content': top_sites,
            'explanation': f"Top performing site: {top_sites.index[0]} with {top_sites.iloc[0]['eligibility_rate']:.1f}% eligibility rate."
        }
    except Exception as e:
        return {'type': 'text', 'content': f"Error analyzing top sites: {e}"}

def create_risk_by_site_analysis(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze dropout risk by site"""
    try:
        risk_by_site = df.groupby(['location', 'dropout_risk']).size().reset_index(name='count')
        
        fig = px.bar(
            risk_by_site,
            x='location',
            y='count',
            color='dropout_risk',
            title='Dropout Risk Distribution by Site',
            color_discrete_map={'High': '#d62728', 'Medium': '#ff9800', 'Low': '#2ca02c'}
        )
        fig.update_xaxes(tickangle=45)
        
        return {
            'type': 'chart',
            'content': fig,
            'explanation': "Compare how dropout risk is distributed across different sites."
        }
    except Exception as e:
        return {'type': 'text', 'content': f"Error creating risk by site analysis: {e}"}

def create_eligibility_by_site(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze eligibility rates by site"""
    try:
        site_eligibility = df.groupby('location').agg({
            'meets_criteria': ['sum', 'count']
        })
        site_eligibility.columns = ['eligible', 'total']
        site_eligibility['rate'] = (site_eligibility['eligible'] / site_eligibility['total']) * 100
        site_eligibility = site_eligibility.sort_values('rate', ascending=False)
        
        fig = go.Figure(data=[
            go.Bar(x=site_eligibility.index, y=site_eligibility['rate'])
        ])
        fig.update_layout(
            title='Eligibility Rate by Site',
            xaxis_title='Site',
            yaxis_title='Eligibility Rate (%)'
        )
        
        return {
            'type': 'chart',
            'content': fig,
            'explanation': f"Best performing site: {site_eligibility.index[0]} ({site_eligibility.iloc[0]['rate']:.1f}% eligible)"
        }
    except Exception as e:
        return {'type': 'text', 'content': f"Error creating eligibility by site analysis: {e}"}

def create_count_summary(df: pd.DataFrame, query: str) -> Dict[str, Any]:
    """Create summary counts based on query"""
    try:
        total = len(df)
        eligible = len(df[df['meets_criteria'] == True])
        high_risk = len(df[df['dropout_risk'] == 'High'])
        sites = df['location'].nunique()
        
        summary = f"""
        **Participant Counts:**
        - Total participants: {total:,}
        - Eligible participants: {eligible:,}
        - High-risk participants: {high_risk:,}
        - Active sites: {sites}
        - Average age: {df['age'].mean():.1f} years
        """
        
        return {
            'type': 'text',
            'content': summary,
            'explanation': "Complete count summary of your clinical trial data."
        }
    except Exception as e:
        return {'type': 'text', 'content': f"Error creating count summary: {e}"}

def create_comparison_analysis(df: pd.DataFrame, query: str) -> Dict[str, Any]:
    """Create comparison analysis based on query"""
    try:
        # Compare sites by eligibility
        if 'site' in query or 'location' in query:
            site_comparison = df.groupby('location').agg({
                'participant_id': 'count',
                'meets_criteria': 'sum',
                'dropout_risk': lambda x: (x == 'High').sum()
            }).rename(columns={
                'participant_id': 'Total',
                'meets_criteria': 'Eligible',
                'dropout_risk': 'High_Risk'
            })
            
            site_comparison['Eligibility_Rate'] = (site_comparison['Eligible'] / site_comparison['Total']) * 100
            site_comparison['Risk_Rate'] = (site_comparison['High_Risk'] / site_comparison['Total']) * 100
            
            return {
                'type': 'table',
                'content': site_comparison,
                'explanation': "Comparison of sites showing participant counts, eligibility rates, and risk levels."
            }
        
        # Compare age groups
        elif 'age' in query:
            age_comparison = df.groupby('age_group').agg({
                'participant_id': 'count',
                'meets_criteria': 'sum',
                'dropout_risk': lambda x: (x == 'High').sum()
            })
            
            return {
                'type': 'table',
                'content': age_comparison,
                'explanation': "Comparison across age groups showing eligibility and risk patterns."
            }
        
        else:
            return {
                'type': 'text',
                'content': "Please specify what you'd like to compare (e.g., 'compare sites' or 'compare age groups').",
                'explanation': "Comparison requires specific categories to analyze."
            }
            
    except Exception as e:
        return {'type': 'text', 'content': f"Error creating comparison: {e}"}

def create_age_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Create age-related summary"""
    try:
        age_stats = df['age'].describe()
        age_groups = df['age_group'].value_counts()
        
        summary = f"""
        **Age Analysis:**
        - Average age: {age_stats['mean']:.1f} years
        - Median age: {age_stats['50%']:.1f} years
        - Age range: {age_stats['min']:.0f} - {age_stats['max']:.0f} years
        - Most common age group: {age_groups.index[0]} ({age_groups.iloc[0]} participants)
        """
        
        return {
            'type': 'text',
            'content': summary,
            'explanation': "Summary statistics about participant ages in your study."
        }
    except Exception as e:
        return {'type': 'text', 'content': f"Error creating age summary: {e}"}

def create_site_overview(df: pd.DataFrame) -> Dict[str, Any]:
    """Create site overview analysis"""
    try:
        site_overview = df.groupby('location').agg({
            'participant_id': 'count',
            'meets_criteria': ['sum', lambda x: (x.sum() / len(x)) * 100],
            'dropout_risk': lambda x: (x == 'High').sum(),
            'age': 'mean'
        }).round(1)
        
        site_overview.columns = ['Total_Participants', 'Eligible_Count', 'Eligibility_Rate', 'High_Risk_Count', 'Avg_Age']
        
        return {
            'type': 'table',
            'content': site_overview,
            'explanation': "Complete overview of all sites showing key performance metrics."
        }
    except Exception as e:
        return {'type': 'text', 'content': f"Error creating site overview: {e}"}

def create_overall_risk_analysis(df: pd.DataFrame) -> Dict[str, Any]:
    """Create overall risk analysis"""
    try:
        risk_dist = df['dropout_risk'].value_counts()
        risk_pct = df['dropout_risk'].value_counts(normalize=True) * 100
        
        # Risk by eligibility
        risk_eligible = pd.crosstab(df['dropout_risk'], df['meets_criteria'], normalize='columns') * 100
        
        summary = f"""
        **Dropout Risk Analysis:**
        - High risk: {risk_dist.get('High', 0)} participants ({risk_pct.get('High', 0):.1f}%)
        - Medium risk: {risk_dist.get('Medium', 0)} participants ({risk_pct.get('Medium', 0):.1f}%)
        - Low risk: {risk_dist.get('Low', 0)} participants ({risk_pct.get('Low', 0):.1f}%)
        
        Among eligible participants: {risk_eligible.loc['High', True] if 'High' in risk_eligible.index and True in risk_eligible.columns else 0:.1f}% are high risk
        """
        
        return {
            'type': 'text',
            'content': summary,
            'explanation': "Comprehensive analysis of dropout risk distribution in your study."
        }
    except Exception as e:
        return {'type': 'text', 'content': f"Error creating risk analysis: {e}"}

def create_eligibility_overview(df: pd.DataFrame) -> Dict[str, Any]:
    """Create eligibility overview"""
    try:
        total = len(df)
        eligible = len(df[df['meets_criteria'] == True])
        ineligible = total - eligible
        rate = (eligible / total) * 100
        
        # Eligibility by key factors
        by_age = df.groupby('age_group')['meets_criteria'].agg(['sum', 'count'])
        by_age['rate'] = (by_age['sum'] / by_age['count']) * 100
        
        summary = f"""
        **Eligibility Analysis:**
        - Total screened: {total:,}
        - Eligible: {eligible:,} ({rate:.1f}%)
        - Ineligible: {ineligible:,} ({100-rate:.1f}%)
        
        **Eligibility by Age Group:**
        {by_age['rate'].to_string()}
        """
        
        return {
            'type': 'text',
            'content': summary,
            'explanation': "Detailed breakdown of participant eligibility across different segments."
        }
    except Exception as e:
        return {'type': 'text', 'content': f"Error creating eligibility overview: {e}"}

def create_risk_sites_analysis(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze sites with highest risk"""
    try:
        site_risk = df.groupby('location').agg({
            'dropout_risk': lambda x: (x == 'High').sum(),
            'participant_id': 'count'
        })
        site_risk['risk_rate'] = (site_risk['dropout_risk'] / site_risk['participant_id']) * 100
        site_risk = site_risk.sort_values('risk_rate', ascending=False)
        
        highest_risk_site = site_risk.index[0]
        highest_risk_rate = site_risk.iloc[0]['risk_rate']
        
        return {
            'type': 'table',
            'content': site_risk,
            'explanation': f"Site with highest risk: {highest_risk_site} ({highest_risk_rate:.1f}% high-risk participants)"
        }
    except Exception as e:
        return {'type': 'text', 'content': f"Error analyzing risk sites: {e}"}