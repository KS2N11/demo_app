import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from config import CHART_COLORS

def get_original_columns(df: pd.DataFrame) -> List[str]:
    """Get only original columns, filtering out encoded columns"""
    return [col for col in df.columns if not col.startswith('_encoded_')]

def get_original_numeric_columns(df: pd.DataFrame) -> List[str]:
    """Get only original numeric columns"""
    original_cols = get_original_columns(df)
    return df[original_cols].select_dtypes(include=['number']).columns.tolist()

def get_original_categorical_columns(df: pd.DataFrame) -> List[str]:
    """Get only original categorical columns"""
    original_cols = get_original_columns(df)
    return df[original_cols].select_dtypes(include=['object', 'category']).columns.tolist()

def get_available_charts(df: pd.DataFrame) -> List[str]:
    """
    Determine which charts can be created based on available data
    
    Args:
        df: Input dataframe
        
    Returns:
        List of available chart types
    """
    available_charts = []
    
    # Get only original columns for analysis (exclude encoded ones)
    original_cols = get_original_columns(df)
    original_df = df[original_cols] if original_cols else df
    
    # Always available - basic distribution charts
    if not original_df.empty:
        available_charts.extend(['data_overview', 'column_distributions'])
    
    # Eligibility funnel - if eligibility column exists
    eligibility_cols = [col for col in original_cols if any(keyword in col.lower() 
                       for keyword in ['eligible', 'criteria', 'qualified', 'pass'])]
    if eligibility_cols:
        available_charts.append('eligibility_funnel')
    
    # Risk charts - if risk-related columns exist
    risk_cols = [col for col in original_cols if any(keyword in col.lower() 
                for keyword in ['risk', 'dropout', 'attrition'])]
    if risk_cols:
        available_charts.append('risk_distribution')
    
    # Site/location charts - if location columns exist
    location_cols = [col for col in original_cols if any(keyword in col.lower() 
                    for keyword in ['site', 'location', 'center', 'facility', 'clinic'])]
    if location_cols:
        available_charts.append('site_distribution')
    
    # Age/demographic charts - if numeric or convertible age columns exist
    numeric_cols = get_original_numeric_columns(df)
    age_cols_all = [col for col in original_cols if 'age' in col.lower()]
    if len(numeric_cols) > 0 or len(age_cols_all) > 0:
        available_charts.append('demographic_analysis')
    
    # One-hot encoded feature analysis - check for encoded columns
    encoded_cols = [col for col in df.columns if col.startswith('_encoded_')]
    if len(encoded_cols) > 2:
        available_charts.append('one_hot_analysis')
    
    # Correlation analysis - if enough numeric columns
    if len(numeric_cols) >= 2:
        available_charts.append('correlation_analysis')
    
    # Time-based charts - if date columns exist
    date_cols = df.select_dtypes(include=['datetime64']).columns
    if len(date_cols) > 0:
        available_charts.append('time_analysis')
    
    return available_charts

def create_eligibility_funnel(df: pd.DataFrame) -> go.Figure:
    """
    Create a dynamic eligibility funnel chart based on available data
    
    Args:
        df: Input dataframe
        
    Returns:
        Plotly figure object
    """
    try:
        total = len(df)
        
        # Find eligibility-related columns dynamically
        eligibility_cols = [col for col in df.columns if any(keyword in col.lower() 
                           for keyword in ['eligible', 'criteria', 'qualified', 'pass', 'meets'])]
        
        risk_cols = [col for col in df.columns if any(keyword in col.lower() 
                    for keyword in ['risk', 'dropout', 'attrition'])]
        
        funnel_stages = ["Total Participants"]
        funnel_values = [total]
        
        # Add eligibility stage if available
        if eligibility_cols:
            eligibility_col = eligibility_cols[0]
            # Handle different data types for eligibility
            if df[eligibility_col].dtype == 'bool':
                eligible = len(df[df[eligibility_col] == True])
            else:
                # Handle string/numeric eligibility values
                positive_values = df[eligibility_col].astype(str).str.lower().isin(['yes', 'true', '1', 'pass', 'qualified'])
                eligible = positive_values.sum()
            
            funnel_stages.append("Eligible")
            funnel_values.append(eligible)
            
            # Add low-risk stage if risk data is available
            if risk_cols and eligible > 0:
                risk_col = risk_cols[0]
                low_risk_mask = df[eligibility_col].astype(str).str.lower().isin(['yes', 'true', '1', 'pass', 'qualified'])
                if df[risk_col].dtype == 'object':
                    low_risk_mask = low_risk_mask & (df[risk_col].astype(str).str.lower() == 'low')
                else:
                    # Assume numeric risk where lower is better
                    low_risk_threshold = df[risk_col].quantile(0.33)  # Bottom 33% as "low risk"
                    low_risk_mask = low_risk_mask & (df[risk_col] <= low_risk_threshold)
                
                low_risk = low_risk_mask.sum()
                funnel_stages.append("Low Risk Eligible")
                funnel_values.append(low_risk)
        
        # If no eligibility data, create a simple distribution funnel
        if len(funnel_stages) == 1:
            # Use first categorical column for segmentation
            categorical_cols = df.select_dtypes(include=['object']).columns
            if len(categorical_cols) > 0:
                first_cat_col = categorical_cols[0]
                top_category = df[first_cat_col].value_counts().index[0]
                top_count = df[first_cat_col].value_counts().iloc[0]
                funnel_stages.append(f"Top Category ({first_cat_col})")
                funnel_values.append(top_count)
        
        fig = go.Figure(go.Funnel(
            y=funnel_stages,
            x=funnel_values,
            textinfo="value+percent initial",
            marker=dict(
                color=[CHART_COLORS['info'], CHART_COLORS['success'], CHART_COLORS['primary']][:len(funnel_stages)]
            )
        ))
        
        fig.update_layout(
            title="Participant Flow Analysis",
            font_size=12,
            height=400,
            margin=dict(l=80, r=80, t=100, b=80),
            # Prevent label overlap
            showlegend=False
        )
        
        return fig
        
    except Exception as e:
        # Return empty figure on error
        fig = go.Figure()
        fig.add_annotation(text=f"Unable to create funnel chart with available data", 
                          xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig

def create_dropout_risk_chart(df: pd.DataFrame) -> go.Figure:
    """
    Create a dynamic risk/category distribution chart based on available data
    
    Args:
        df: Input dataframe
        
    Returns:
        Plotly figure object
    """
    try:
        # Find risk-related columns dynamically
        risk_cols = [col for col in df.columns if any(keyword in col.lower() 
                    for keyword in ['risk', 'dropout', 'attrition', 'level'])]
        
        if risk_cols:
            risk_col = risk_cols[0]
            risk_counts = df[risk_col].value_counts()
            chart_title = f"{risk_col.replace('_', ' ').title()} Distribution"
            x_title = risk_col.replace('_', ' ').title()
        else:
            # Fall back to first categorical column
            categorical_cols = df.select_dtypes(include=['object']).columns
            if len(categorical_cols) > 0:
                risk_col = categorical_cols[0]
                risk_counts = df[risk_col].value_counts()
                chart_title = f"{risk_col.replace('_', ' ').title()} Distribution"
                x_title = risk_col.replace('_', ' ').title()
            else:
                # Create a simple count chart
                chart_title = "Data Distribution"
                x_title = "Categories"
                risk_counts = pd.Series([len(df)], index=['Total Records'])
        
        # Dynamic color assignment
        colors = {
            'High': CHART_COLORS['danger'],
            'Medium': CHART_COLORS['warning'], 
            'Low': CHART_COLORS['success'],
            'Yes': CHART_COLORS['success'],
            'No': CHART_COLORS['danger'],
            'True': CHART_COLORS['success'],
            'False': CHART_COLORS['danger']
        }
        
        # Generate colors for all categories
        chart_colors = []
        color_palette = [CHART_COLORS['primary'], CHART_COLORS['info'], CHART_COLORS['success'], 
                        CHART_COLORS['warning'], CHART_COLORS['danger']]
        
        for i, category in enumerate(risk_counts.index):
            if str(category) in colors:
                chart_colors.append(colors[str(category)])
            else:
                chart_colors.append(color_palette[i % len(color_palette)])
        
        fig = go.Figure(data=[
            go.Bar(
                x=risk_counts.index,
                y=risk_counts.values,
                marker_color=chart_colors,
                text=risk_counts.values,
                textposition='auto',
            )
        ])
        
        fig.update_layout(
            title=chart_title,
            xaxis_title=x_title,
            yaxis_title="Number of Participants",
            height=400,
            showlegend=False,
            margin=dict(l=80, r=80, t=100, b=80),
            # Prevent label overlap
            xaxis=dict(
                tickangle=45 if len(risk_counts) > 5 else 0,
                automargin=True
            ),
            yaxis=dict(
                automargin=True
            )
        )
        
        return fig
        
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(text=f"Unable to create distribution chart with available data", 
                          xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig

def create_site_distribution(df: pd.DataFrame) -> go.Figure:
    """
    Create a dynamic site/location distribution chart based on available data
    
    Args:
        df: Input dataframe
        
    Returns:
        Plotly figure object
    """
    try:
        # Find location-related columns dynamically
        location_cols = [col for col in df.columns if any(keyword in col.lower() 
                        for keyword in ['site', 'location', 'center', 'facility', 'clinic', 'hospital'])]
        
        if not location_cols:
            # Fall back to first categorical column with reasonable number of unique values
            categorical_cols = df.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                if 2 <= df[col].nunique() <= 20:  # Reasonable number of categories for visualization
                    location_cols = [col]
                    break
        
        if not location_cols:
            # Create a simple summary chart
            fig = go.Figure(data=[
                go.Bar(x=['Total Records'], y=[len(df)], 
                      marker_color=CHART_COLORS['primary'])
            ])
            fig.update_layout(
                title="Dataset Summary",
                xaxis_title="Dataset",
                yaxis_title="Number of Records",
                height=400
            )
            return fig
        
        location_col = location_cols[0]
        
        # Find participant ID column
        id_cols = [col for col in df.columns if any(keyword in col.lower() 
                  for keyword in ['id', 'participant', 'subject', 'patient'])]
        id_col = id_cols[0] if id_cols else df.columns[0]
        
        # Find eligibility column
        eligibility_cols = [col for col in df.columns if any(keyword in col.lower() 
                           for keyword in ['eligible', 'criteria', 'qualified', 'pass', 'meets'])]
        
        # Create basic site metrics
        site_metrics = df.groupby(location_col).agg({
            id_col: 'count'
        }).rename(columns={id_col: 'total'})
        
        # Add eligibility metrics if available
        if eligibility_cols:
            eligibility_col = eligibility_cols[0]
            if df[eligibility_col].dtype == 'bool':
                site_metrics['eligible'] = df.groupby(location_col)[eligibility_col].sum()
            else:
                # Handle string/numeric eligibility values
                eligible_mask = df[eligibility_col].astype(str).str.lower().isin(['yes', 'true', '1', 'pass', 'qualified'])
                site_metrics['eligible'] = df[eligible_mask].groupby(location_col).size().reindex(site_metrics.index, fill_value=0)
            
            site_metrics['eligibility_rate'] = (site_metrics['eligible'] / site_metrics['total']) * 100
        
        site_metrics = site_metrics.sort_values('total', ascending=True)
        
        fig = go.Figure()
        
        # Total participants bar
        fig.add_trace(go.Bar(
            y=site_metrics.index,
            x=site_metrics['total'],
            name=f'Total by {location_col.replace("_", " ").title()}',
            marker_color=CHART_COLORS['info'],
            orientation='h'
        ))
        
        # Eligible participants bar (if eligibility data exists)
        if 'eligible' in site_metrics.columns:
            fig.add_trace(go.Bar(
                y=site_metrics.index,
                x=site_metrics['eligible'],
                name='Eligible Participants',
                marker_color=CHART_COLORS['success'],
                orientation='h'
            ))
        
        fig.update_layout(
            title=f"Distribution by {location_col.replace('_', ' ').title()}",
            xaxis_title="Number of Participants",
            yaxis_title=location_col.replace('_', ' ').title(),
            height=max(400, len(site_metrics) * 50),
            barmode='overlay' if 'eligible' in site_metrics.columns else 'group',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(l=120, r=80, t=100, b=80),
            # Prevent label overlap for y-axis labels
            yaxis=dict(
                automargin=True,
                tickangle=0
            ),
            xaxis=dict(
                automargin=True
            )
        )
        
        return fig
        
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(text=f"Unable to create location distribution chart", 
                          xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig

def create_dynamic_numeric_chart(df: pd.DataFrame) -> go.Figure:
    """
    Create a dynamic chart for numeric data (age, BMI, etc.) - handles categorical age data
    
    Args:
        df: Input dataframe
        
    Returns:
        Plotly figure object
    """
    try:
        # First try to find truly numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        # Check for age columns that might be categorical but convertible
        age_cols_all = [col for col in df.columns if 'age' in col.lower()]
        
        target_col = None
        chart_data = None
        
        # Prioritize age columns first
        if age_cols_all:
            for col in age_cols_all:
                if col in numeric_cols:
                    # Already numeric age column
                    target_col = col
                    chart_data = df[col].dropna()
                    break
                else:
                    # Check if it's now numeric after intelligent conversion
                    if pd.api.types.is_numeric_dtype(df[col]):
                        target_col = col
                        chart_data = df[col].dropna()
                        break
        
        # If no age column found, use first numeric column
        if target_col is None and len(numeric_cols) > 0:
            target_col = numeric_cols[0]
            chart_data = df[target_col].dropna()
        
        # If still no data, create a categorical chart for age
        if target_col is None and age_cols_all:
            target_col = age_cols_all[0]
            age_counts = df[target_col].value_counts()
            
            fig = go.Figure(data=[
                go.Bar(
                    x=age_counts.index,
                    y=age_counts.values,
                    marker_color=CHART_COLORS['primary'],
                    text=age_counts.values,
                    textposition='auto'
                )
            ])
            
            fig.update_layout(
                title=f"{target_col.replace('_', ' ').title()} Distribution",
                xaxis_title=target_col.replace('_', ' ').title(),
                yaxis_title="Count",
                height=400,
                showlegend=False
            )
            return fig
        
        if chart_data is None or len(chart_data) == 0:
            fig = go.Figure()
            fig.add_annotation(text="No suitable data available for numeric analysis", 
                              xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            return fig
        
        fig = go.Figure()
        
        # Create histogram for numeric data
        fig.add_trace(go.Histogram(
            x=chart_data,
            nbinsx=min(30, chart_data.nunique()),
            marker_color=CHART_COLORS['primary'],
            name=target_col.replace('_', ' ').title()
        ))
        
        fig.update_layout(
            title=f"{target_col.replace('_', ' ').title()} Distribution",
            xaxis_title=target_col.replace('_', ' ').title(),
            yaxis_title="Frequency",
            height=400,
            showlegend=False
        )
        
        return fig
        
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(text="Unable to create distribution chart", 
                          xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig

def create_data_overview_chart(df: pd.DataFrame) -> go.Figure:
    """
    Create a comprehensive data overview chart
    
    Args:
        df: Input dataframe
        
    Returns:
        Plotly figure object
    """
    try:
        # Create summary statistics
        total_rows = len(df)
        total_columns = len(df.columns)
        numeric_columns = len(df.select_dtypes(include=[np.number]).columns)
        categorical_columns = len(df.select_dtypes(include=['object']).columns)
        missing_data_pct = (df.isnull().sum().sum() / (total_rows * total_columns)) * 100
        
        # Create a summary bar chart
        categories = ['Total Rows', 'Total Columns', 'Numeric Columns', 'Text Columns']
        values = [total_rows, total_columns, numeric_columns, categorical_columns]
        
        fig = go.Figure(data=[
            go.Bar(
                x=categories,
                y=values,
                marker_color=[CHART_COLORS['primary'], CHART_COLORS['info'], 
                             CHART_COLORS['success'], CHART_COLORS['warning']],
                text=values,
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title=f"Dataset Overview ({missing_data_pct:.1f}% Missing Data)",
            xaxis_title="Data Metrics",
            yaxis_title="Count",
            height=400,
            showlegend=False
        )
        
        return fig
        
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(text="Unable to create data overview chart", 
                          xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig

def create_one_hot_analysis_chart(df: pd.DataFrame) -> go.Figure:
    """
    Create a chart for one-hot encoded categorical variables
    
    Args:
        df: Input dataframe with one-hot encoded columns
        
    Returns:
        Plotly figure object
    """
    try:
        # Find one-hot encoded columns (binary columns with meaningful names)
        binary_cols = []
        for col in df.columns:
            if df[col].dtype in ['int64', 'float64'] and set(df[col].dropna().unique()).issubset({0, 1}):
                # Check if column name suggests it's one-hot encoded
                if '_' in col and not col.lower().endswith(('_id', '_number', '_count')):
                    binary_cols.append(col)
        
        if len(binary_cols) == 0:
            fig = go.Figure()
            fig.add_annotation(text="No one-hot encoded features found", 
                              xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            return fig
        
        # Create a stacked bar chart showing feature distributions
        categories = []
        counts = []
        
        for col in binary_cols[:10]:  # Limit to first 10 to avoid overcrowding
            positive_count = df[col].sum()
            categories.append(col.replace('_', ' ').title())
            counts.append(positive_count)
        
        fig = go.Figure(data=[
            go.Bar(
                x=categories,
                y=counts,
                marker_color=CHART_COLORS['primary'],
                text=counts,
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title="One-Hot Encoded Feature Distribution",
            xaxis_title="Features",
            yaxis_title="Count (Positive Cases)",
            height=400,
            showlegend=False,
            margin=dict(l=80, r=80, t=100, b=120),
            # Prevent label overlap
            xaxis=dict(
                tickangle=-45,
                automargin=True
            ),
            yaxis=dict(
                automargin=True
            )
        )
        
        return fig
        
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(text="Unable to create one-hot analysis chart", 
                          xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig

def create_correlation_matrix_chart(df: pd.DataFrame) -> go.Figure:
    """
    Create a correlation matrix for numeric and converted categorical data
    
    Args:
        df: Input dataframe
        
    Returns:
        Plotly figure object
    """
    try:
        # Get numeric columns (including converted categorical ones)
        numeric_df = df.select_dtypes(include=[np.number])
        
        if len(numeric_df.columns) < 2:
            fig = go.Figure()
            fig.add_annotation(text="Need at least 2 numeric columns for correlation analysis", 
                              xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            return fig
        
        # Calculate correlation matrix
        corr_matrix = numeric_df.corr()
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=corr_matrix.round(2).values,
            texttemplate="%{text}",
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title="Feature Correlation Matrix",
            height=max(400, len(corr_matrix.columns) * 30),
            width=max(400, len(corr_matrix.columns) * 30)
        )
        
        return fig
        
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(text="Unable to create correlation matrix", 
                          xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig

def create_age_distribution_chart(df: pd.DataFrame) -> go.Figure:
    """
    Create age distribution histogram
    
    Args:
        df: Cleaned dataframe
        
    Returns:
        Plotly figure object
    """
    try:
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=df['age'],
            nbinsx=20,
            marker_color=CHART_COLORS['primary'],
            opacity=0.7,
            name='Age Distribution'
        ))
        
        fig.update_layout(
            title="Age Distribution of Participants",
            xaxis_title="Age",
            yaxis_title="Count",
            height=400,
            showlegend=False
        )
        
        return fig
        
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(text=f"Error creating age chart: {e}", 
                          xref="paper", yref="paper", x=0.5, y=0.5)
        return fig

def create_custom_chart(df: pd.DataFrame, chart_type: str, x_col: str, y_col: str = None, 
                       group_col: str = None, agg_func: str = 'count') -> go.Figure:
    """
    Create custom charts based on user requirements
    
    Args:
        df: Cleaned dataframe
        chart_type: Type of chart ('bar', 'line', 'pie', 'scatter', 'histogram')
        x_col: Column for x-axis
        y_col: Column for y-axis (optional)
        group_col: Column for grouping/coloring (optional)
        agg_func: Aggregation function ('count', 'sum', 'mean', 'median')
        
    Returns:
        Plotly figure object
    """
    try:
        fig = go.Figure()
        
        if chart_type == 'bar':
            if y_col and y_col in df.columns:
                # Aggregate data
                if group_col and group_col in df.columns:
                    grouped = df.groupby([x_col, group_col])[y_col].agg(agg_func).reset_index()
                    for group in grouped[group_col].unique():
                        group_data = grouped[grouped[group_col] == group]
                        fig.add_trace(go.Bar(
                            x=group_data[x_col],
                            y=group_data[y_col],
                            name=str(group)
                        ))
                else:
                    grouped = df.groupby(x_col)[y_col].agg(agg_func).reset_index()
                    fig.add_trace(go.Bar(
                        x=grouped[x_col],
                        y=grouped[y_col],
                        marker_color=CHART_COLORS['primary']
                    ))
            else:
                # Count by category
                counts = df[x_col].value_counts()
                fig.add_trace(go.Bar(
                    x=counts.index,
                    y=counts.values,
                    marker_color=CHART_COLORS['primary']
                ))
        
        elif chart_type == 'pie':
            counts = df[x_col].value_counts()
            fig.add_trace(go.Pie(
                labels=counts.index,
                values=counts.values,
                hole=0.3
            ))
        
        elif chart_type == 'histogram':
            fig.add_trace(go.Histogram(
                x=df[x_col],
                marker_color=CHART_COLORS['primary'],
                opacity=0.7
            ))
        
        elif chart_type == 'scatter' and y_col:
            if group_col and group_col in df.columns:
                # Handle categorical color mapping properly
                df_copy = df.copy()
                df_copy[group_col] = df_copy[group_col].astype(str)  # Ensure categorical as string
                unique_groups = df_copy[group_col].unique()
                
                for i, group in enumerate(unique_groups):
                    group_data = df_copy[df_copy[group_col] == group]
                    fig.add_trace(go.Scatter(
                        x=group_data[x_col],
                        y=group_data[y_col],
                        mode='markers',
                        name=str(group),
                        marker=dict(color=CHART_COLORS['primary'] if i == 0 else CHART_COLORS['info'])
                    ))
            else:
                fig.add_trace(go.Scatter(
                    x=df[x_col],
                    y=df[y_col],
                    mode='markers',
                    marker=dict(color=CHART_COLORS['primary'])
                ))
        
        elif chart_type == 'line' and y_col:
            if group_col and group_col in df.columns:
                for group in df[group_col].unique():
                    group_data = df[df[group_col] == group]
                    fig.add_trace(go.Scatter(
                        x=group_data[x_col],
                        y=group_data[y_col],
                        mode='lines+markers',
                        name=str(group)
                    ))
            else:
                grouped = df.groupby(x_col)[y_col].agg(agg_func).reset_index()
                fig.add_trace(go.Scatter(
                    x=grouped[x_col],
                    y=grouped[y_col],
                    mode='lines+markers',
                    marker_color=CHART_COLORS['primary']
                ))
        
        # Update layout
        title = f"{chart_type.title()} Chart: {x_col}"
        if y_col:
            title += f" vs {y_col}"
        if group_col:
            title += f" by {group_col}"
            
        fig.update_layout(
            title=title,
            xaxis_title=x_col.replace('_', ' ').title(),
            yaxis_title=y_col.replace('_', ' ').title() if y_col else 'Count',
            height=400,
            margin=dict(l=80, r=80, t=100, b=80),
            # Prevent label overlap
            xaxis=dict(
                tickangle=45 if chart_type in ['bar', 'line'] and df[x_col].nunique() > 5 else 0,
                automargin=True
            ),
            yaxis=dict(
                automargin=True
            )
        )
        
        return fig
        
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(text=f"Error creating custom chart: {e}", 
                          xref="paper", yref="paper", x=0.5, y=0.5)
        return fig

def create_correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    """
    Create correlation heatmap for numeric columns with user-friendly names
    
    Args:
        df: Cleaned dataframe
        
    Returns:
        Plotly figure object
    """
    try:
        # Get only original numeric columns (filter out encoded ones)
        original_cols = get_original_columns(df)
        numeric_cols = df[original_cols].select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            fig = go.Figure()
            fig.add_annotation(text="Not enough numeric columns for correlation analysis", 
                              xref="paper", yref="paper", x=0.5, y=0.5, font=dict(size=14))
            fig.update_layout(height=400, width=600)
            return fig
        
        # Create correlation matrix
        corr_matrix = df[numeric_cols].corr()
        
        # Create user-friendly column names for display
        friendly_names = [col.replace('_', ' ').title() for col in numeric_cols]
        
        # Calculate dynamic size based on number of columns
        size = max(500, len(numeric_cols) * 50)
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=friendly_names,
            y=friendly_names,
            colorscale='RdBu',
            zmid=0,
            text=np.round(corr_matrix.values, 2),
            texttemplate="%{text}",
            textfont={"size": max(8, min(12, 100 // len(numeric_cols)))},
            hoverongaps=False,
            colorbar=dict(
                title="Correlation",
                titleside="right"
            )
        ))
        
        fig.update_layout(
            title=dict(
                text="Feature Correlation Matrix",
                x=0.5,
                font=dict(size=16)
            ),
            height=size,
            width=size,
            margin=dict(l=100, r=100, t=80, b=100),
            xaxis=dict(
                tickangle=45,
                side='bottom'
            ),
            yaxis=dict(
                tickangle=0
            )
        )
        
        return fig
        
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(text=f"Error creating heatmap: {e}", 
                          xref="paper", yref="paper", x=0.5, y=0.5, font=dict(size=14))
        fig.update_layout(height=400, width=600)
        return fig