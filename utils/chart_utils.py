import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from typing import Dict, Any, List
from config import CHART_COLORS

def create_eligibility_funnel(df: pd.DataFrame) -> go.Figure:
    """
    Create an eligibility funnel chart
    
    Args:
        df: Cleaned dataframe
        
    Returns:
        Plotly figure object
    """
    try:
        total = len(df)
        eligible = len(df[df['meets_criteria'] == True])
        low_risk = len(df[(df['meets_criteria'] == True) & (df['dropout_risk'] == 'Low')])
        
        fig = go.Figure(go.Funnel(
            y = ["Total Screened", "Eligible", "Low Risk Eligible"],
            x = [total, eligible, low_risk],
            textinfo = "value+percent initial",
            marker = dict(
                color = [CHART_COLORS['info'], CHART_COLORS['success'], CHART_COLORS['primary']]
            )
        ))
        
        fig.update_layout(
            title="Eligibility Funnel",
            font_size=12,
            height=400
        )
        
        return fig
        
    except Exception as e:
        # Return empty figure on error
        fig = go.Figure()
        fig.add_annotation(text=f"Error creating funnel chart: {e}", 
                          xref="paper", yref="paper", x=0.5, y=0.5)
        return fig

def create_dropout_risk_chart(df: pd.DataFrame) -> go.Figure:
    """
    Create dropout risk distribution chart
    
    Args:
        df: Cleaned dataframe
        
    Returns:
        Plotly figure object
    """
    try:
        risk_counts = df['dropout_risk'].value_counts()
        
        colors = {
            'High': CHART_COLORS['danger'],
            'Medium': CHART_COLORS['warning'], 
            'Low': CHART_COLORS['success']
        }
        
        fig = go.Figure(data=[
            go.Bar(
                x=risk_counts.index,
                y=risk_counts.values,
                marker_color=[colors.get(risk, CHART_COLORS['primary']) for risk in risk_counts.index],
                text=risk_counts.values,
                textposition='auto',
            )
        ])
        
        fig.update_layout(
            title="Dropout Risk Distribution",
            xaxis_title="Risk Level",
            yaxis_title="Number of Participants",
            height=400,
            showlegend=False
        )
        
        return fig
        
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(text=f"Error creating risk chart: {e}", 
                          xref="paper", yref="paper", x=0.5, y=0.5)
        return fig

def create_site_distribution(df: pd.DataFrame) -> go.Figure:
    """
    Create site distribution chart with eligibility rates
    
    Args:
        df: Cleaned dataframe
        
    Returns:
        Plotly figure object
    """
    try:
        site_metrics = df.groupby('location').agg({
            'participant_id': 'count',
            'meets_criteria': 'sum'
        }).rename(columns={
            'participant_id': 'total',
            'meets_criteria': 'eligible'
        })
        
        site_metrics['eligibility_rate'] = (site_metrics['eligible'] / site_metrics['total']) * 100
        site_metrics = site_metrics.sort_values('total', ascending=True)
        
        fig = go.Figure()
        
        # Total participants bar
        fig.add_trace(go.Bar(
            y=site_metrics.index,
            x=site_metrics['total'],
            name='Total Participants',
            marker_color=CHART_COLORS['info'],
            orientation='h'
        ))
        
        # Eligible participants bar
        fig.add_trace(go.Bar(
            y=site_metrics.index,
            x=site_metrics['eligible'],
            name='Eligible Participants',
            marker_color=CHART_COLORS['success'],
            orientation='h'
        ))
        
        fig.update_layout(
            title="Participants by Site",
            xaxis_title="Number of Participants",
            yaxis_title="Site",
            height=max(400, len(site_metrics) * 50),
            barmode='overlay',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
        
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(text=f"Error creating site chart: {e}", 
                          xref="paper", yref="paper", x=0.5, y=0.5)
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
            color = df[group_col] if group_col and group_col in df.columns else None
            fig.add_trace(go.Scatter(
                x=df[x_col],
                y=df[y_col],
                mode='markers',
                marker=dict(
                    color=color,
                    colorscale='viridis' if color is not None else None
                ),
                text=df[group_col] if group_col else None
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
        )
        
        return fig
        
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(text=f"Error creating custom chart: {e}", 
                          xref="paper", yref="paper", x=0.5, y=0.5)
        return fig

def create_correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    """
    Create correlation heatmap for numeric columns
    
    Args:
        df: Cleaned dataframe
        
    Returns:
        Plotly figure object
    """
    try:
        # Select only numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            fig = go.Figure()
            fig.add_annotation(text="Not enough numeric columns for correlation analysis", 
                              xref="paper", yref="paper", x=0.5, y=0.5)
            return fig
        
        corr_matrix = df[numeric_cols].corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=np.round(corr_matrix.values, 2),
            texttemplate="%{text}",
            textfont={"size": 10},
        ))
        
        fig.update_layout(
            title="Correlation Heatmap",
            height=500,
            width=500
        )
        
        return fig
        
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(text=f"Error creating heatmap: {e}", 
                          xref="paper", yref="paper", x=0.5, y=0.5)
        return fig