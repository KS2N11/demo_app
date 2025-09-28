import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from utils.chart_utils import create_custom_chart, create_correlation_heatmap
from utils.data_utils import filter_data

# Page configuration
st.set_page_config(
    page_title="Custom Charts - Clinical Trial Analytics",
    page_icon="📈",
    layout="wide"
)

def main():
    st.title("📈 Custom Chart Builder")
    st.markdown("Create custom visualizations from your clinical trial data")
    
    # Check if data is available
    if 'processed_data' not in st.session_state or st.session_state.processed_data is None:
        st.warning("🚨 No data available. Please go to the main page and upload your clinical trial data.")
        
        if st.button("🔙 Go to Data Upload"):
            st.switch_page("app.py")
        return
    
    df = st.session_state.processed_data
    
    st.markdown("---")
    
    # Sidebar for chart configuration
    with st.sidebar:
        st.header("🎨 Chart Configuration")
        
        # Chart type selection
        chart_type = st.selectbox(
            "Chart Type",
            ["Bar Chart", "Pie Chart", "Histogram", "Scatter Plot", "Box Plot", "Line Chart", "Correlation Heatmap"],
            help="Select the type of visualization you want to create"
        )
        
        st.markdown("---")
        
        # Column selection based on chart type
        numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
        all_columns = df.columns.tolist()
        
        if chart_type == "Correlation Heatmap":
            st.info("Correlation heatmap will use all numeric columns automatically.")
            x_column = None
            y_column = None
            group_column = None
            
        elif chart_type in ["Bar Chart", "Pie Chart"]:
            x_column = st.selectbox("Category Column", categorical_columns + numeric_columns)
            
            if chart_type == "Bar Chart":
                y_column = st.selectbox(
                    "Value Column (optional)", 
                    ["Count"] + numeric_columns,
                    help="Leave as 'Count' to count occurrences, or select a numeric column to aggregate"
                )
                if y_column == "Count":
                    y_column = None
                
                group_column = st.selectbox(
                    "Group By (optional)",
                    [None] + categorical_columns,
                    help="Optional: Group bars by another categorical variable"
                )
            else:
                y_column = None
                group_column = None
                
        elif chart_type == "Histogram":
            x_column = st.selectbox("Numeric Column", numeric_columns)
            y_column = None
            group_column = st.selectbox(
                "Color By (optional)",
                [None] + categorical_columns,
                help="Optional: Color histogram by categorical variable"
            )
            
        elif chart_type == "Scatter Plot":
            x_column = st.selectbox("X-axis", numeric_columns)
            y_column = st.selectbox("Y-axis", numeric_columns)
            group_column = st.selectbox(
                "Color By (optional)",
                [None] + categorical_columns,
                help="Optional: Color points by categorical variable"
            )
            
        elif chart_type == "Box Plot":
            x_column = st.selectbox("Numeric Column", numeric_columns)
            y_column = None
            group_column = st.selectbox(
                "Group By (optional)",
                [None] + categorical_columns,
                help="Optional: Create separate box plots for each group"
            )
            
        elif chart_type == "Line Chart":
            x_column = st.selectbox("X-axis", all_columns)
            y_column = st.selectbox("Y-axis", numeric_columns)
            group_column = st.selectbox(
                "Group By (optional)",
                [None] + categorical_columns,
                help="Optional: Create separate lines for each group"
            )
        
        st.markdown("---")
        
        # Aggregation function (for applicable charts)
        if chart_type == "Bar Chart" and y_column:
            agg_function = st.selectbox(
                "Aggregation Function",
                ["mean", "sum", "count", "median", "min", "max"],
                help="How to aggregate the selected value column"
            )
        else:
            agg_function = "count"
        
        st.markdown("---")
        
        # Data filters
        st.subheader("🔍 Data Filters")
        
        filters = {}
        
        # Age filter
        if 'age' in df.columns:
            age_range = st.slider(
                "Age Range",
                min_value=int(df['age'].min()),
                max_value=int(df['age'].max()),
                value=(int(df['age'].min()), int(df['age'].max()))
            )
            if age_range != (int(df['age'].min()), int(df['age'].max())):
                filters['age'] = age_range
        
        # Site filter
        if 'location' in df.columns:
            sites = df['location'].unique().tolist()
            selected_sites = st.multiselect(
                "Study Sites",
                sites,
                default=sites,
                help="Select which sites to include in the analysis"
            )
            if len(selected_sites) != len(sites):
                filters['location'] = selected_sites
        
        # Risk filter
        if 'dropout_risk' in df.columns:
            risk_levels = df['dropout_risk'].unique().tolist()
            selected_risks = st.multiselect(
                "Dropout Risk Levels",
                risk_levels,
                default=risk_levels,
                help="Select which risk levels to include"
            )
            if len(selected_risks) != len(risk_levels):
                filters['dropout_risk'] = selected_risks
        
        # Eligibility filter
        if 'meets_criteria' in df.columns:
            eligibility_filter = st.selectbox(
                "Eligibility Status",
                ["All", "Eligible Only", "Not Eligible Only"]
            )
            if eligibility_filter == "Eligible Only":
                filters['meets_criteria'] = [True]
            elif eligibility_filter == "Not Eligible Only":
                filters['meets_criteria'] = [False]
    
    # Main content area
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Apply filters
        filtered_df = filter_data(df, filters) if filters else df
        
        st.subheader(f"📊 {chart_type}")
        
        if len(filtered_df) == 0:
            st.warning("⚠️ No data matches the selected filters. Please adjust your filter criteria.")
            return
        
        # Generate chart
        try:
            if chart_type == "Correlation Heatmap":
                fig = create_correlation_heatmap(filtered_df)
            else:
                # Map chart types to internal names
                chart_type_mapping = {
                    "Bar Chart": "bar",
                    "Pie Chart": "pie", 
                    "Histogram": "histogram",
                    "Scatter Plot": "scatter",
                    "Box Plot": "box",
                    "Line Chart": "line"
                }
                
                internal_chart_type = chart_type_mapping.get(chart_type, "bar")
                
                fig = create_custom_chart(
                    filtered_df, 
                    internal_chart_type, 
                    x_column, 
                    y_column, 
                    group_column, 
                    agg_function
                )
            
            # Display chart
            st.plotly_chart(fig, use_container_width=True)
            
            # Chart insights
            st.subheader("📝 Chart Insights")
            
            insights = generate_chart_insights(filtered_df, chart_type, x_column, y_column, group_column)
            st.markdown(insights)
            
        except Exception as e:
            st.error(f"❌ Error creating chart: {str(e)}")
            st.info("Please check your column selections and try again.")
    
    with col2:
        # Chart information panel
        st.subheader("ℹ️ Chart Info")
        
        st.metric("Total Records", f"{len(filtered_df):,}")
        
        if filters:
            st.metric("Filtered Records", f"{len(df) - len(filtered_df):,} excluded")
        
        if x_column:
            st.write(f"**X-axis:** {x_column}")
        if y_column:
            st.write(f"**Y-axis:** {y_column}")
        if group_column:
            st.write(f"**Grouping:** {group_column}")
        
        st.markdown("---")
        
        # Quick stats for selected columns
        if x_column and x_column in filtered_df.columns:
            st.subheader(f"📊 {x_column} Stats")
            
            if pd.api.types.is_numeric_dtype(filtered_df[x_column]):
                stats = filtered_df[x_column].describe()
                st.write(f"Mean: {stats['mean']:.2f}")
                st.write(f"Median: {stats['50%']:.2f}")
                st.write(f"Std Dev: {stats['std']:.2f}")
                st.write(f"Range: {stats['min']:.2f} - {stats['max']:.2f}")
            else:
                value_counts = filtered_df[x_column].value_counts()
                st.write(f"Unique Values: {len(value_counts)}")
                st.write("Top Values:")
                for value, count in value_counts.head(3).items():
                    st.write(f"  • {value}: {count}")
    
    st.markdown("---")
    
    # Preset chart gallery
    st.header("🎨 Preset Chart Gallery")
    st.markdown("Quick access to commonly used visualizations")
    
    preset_col1, preset_col2, preset_col3 = st.columns(3)
    
    with preset_col1:
        st.subheader("👥 Demographics")
        
        if st.button("Age by Risk Level", key="age_risk"):
            create_preset_chart("age_by_risk", df)
        
        if st.button("Site Distribution", key="site_dist"):
            create_preset_chart("site_distribution", df)
        
        if st.button("Gender Breakdown", key="gender_breakdown"):
            create_preset_chart("gender_breakdown", df)
    
    with preset_col2:
        st.subheader("📈 Performance")
        
        if st.button("Eligibility by Site", key="eligibility_site"):
            create_preset_chart("eligibility_by_site", df)
        
        if st.button("Risk Distribution", key="risk_dist"):
            create_preset_chart("risk_distribution", df)
        
        if st.button("Enrollment Timeline", key="enrollment_timeline"):
            create_preset_chart("enrollment_timeline", df)
    
    with preset_col3:
        st.subheader("🔍 Analysis")
        
        if st.button("BMI vs Age", key="bmi_age"):
            create_preset_chart("bmi_vs_age", df)
        
        if st.button("Protocol Deviations", key="protocol_dev"):
            create_preset_chart("protocol_deviations", df)
        
        if st.button("Follow-up Status", key="followup_status"):
            create_preset_chart("followup_status", df)

def generate_chart_insights(df, chart_type, x_column, y_column, group_column):
    """Generate insights based on the chart configuration"""
    
    insights = []
    
    if chart_type == "Correlation Heatmap":
        numeric_cols = df.select_dtypes(include=['number']).columns
        insights.append(f"📊 Showing correlations between {len(numeric_cols)} numeric variables")
        
        if len(numeric_cols) >= 2:
            # Find strongest correlations
            corr_matrix = df[numeric_cols].corr()
            # Get upper triangle of correlation matrix
            upper_triangle = corr_matrix.where(
                pd.DataFrame(True, index=corr_matrix.index, columns=corr_matrix.columns).values != 
                pd.DataFrame(True, index=corr_matrix.index, columns=corr_matrix.columns).T.values
            )
            
            # Find max correlation (excluding diagonal)
            max_corr = upper_triangle.abs().max().max()
            if not pd.isna(max_corr):
                insights.append(f"🔍 Strongest correlation: {max_corr:.3f}")
    
    elif x_column:
        if pd.api.types.is_numeric_dtype(df[x_column]):
            mean_val = df[x_column].mean()
            median_val = df[x_column].median()
            insights.append(f"📈 {x_column}: Mean = {mean_val:.2f}, Median = {median_val:.2f}")
            
            if abs(mean_val - median_val) > 0.1 * mean_val:
                if mean_val > median_val:
                    insights.append("📊 Data shows positive skew (tail extends to higher values)")
                else:
                    insights.append("📊 Data shows negative skew (tail extends to lower values)")
        else:
            unique_vals = df[x_column].nunique()
            most_common = df[x_column].mode()[0] if len(df[x_column].mode()) > 0 else "N/A"
            insights.append(f"🏷️ {x_column}: {unique_vals} unique values, most common: {most_common}")
    
    if group_column:
        group_counts = df[group_column].value_counts()
        insights.append(f"👥 Grouped by {group_column}: {len(group_counts)} categories")
        
        # Check for imbalanced groups
        min_group = group_counts.min()
        max_group = group_counts.max()
        if max_group > 3 * min_group:
            insights.append("⚠️ Note: Groups are imbalanced - consider this when interpreting results")
    
    if len(insights) == 0:
        insights.append("📊 Chart displays the distribution and patterns in your selected data")
    
    return "\n\n".join([f"• {insight}" for insight in insights])

def create_preset_chart(preset_type, df):
    """Create and display preset charts"""
    
    try:
        if preset_type == "age_by_risk" and 'age' in df.columns and 'dropout_risk' in df.columns:
            fig = go.Figure()
            
            for risk in df['dropout_risk'].unique():
                risk_data = df[df['dropout_risk'] == risk]['age']
                fig.add_trace(go.Box(y=risk_data, name=risk))
            
            fig.update_layout(
                title="Age Distribution by Dropout Risk",
                yaxis_title="Age",
                xaxis_title="Risk Level"
            )
            
        elif preset_type == "eligibility_by_site" and 'location' in df.columns and 'meets_criteria' in df.columns:
            site_eligibility = df.groupby('location')['meets_criteria'].agg(['count', 'sum', 'mean']).reset_index()
            site_eligibility.columns = ['Site', 'Total', 'Eligible', 'Rate']
            site_eligibility['Rate'] *= 100
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=site_eligibility['Site'],
                y=site_eligibility['Rate'],
                text=[f"{rate:.1f}%" for rate in site_eligibility['Rate']],
                textposition='auto'
            ))
            
            fig.update_layout(
                title="Eligibility Rate by Site",
                xaxis_title="Study Site",
                yaxis_title="Eligibility Rate (%)"
            )
        
        else:
            st.warning(f"Preset chart '{preset_type}' not available with current data")
            return
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error creating preset chart: {str(e)}")

if __name__ == "__main__":
    main()