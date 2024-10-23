import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

def apply_layer(loss, deductible, limit):
    """Apply insurance layer to loss amount"""
    return min(limit, max(loss - deductible, 0))

def analyze_cat_losses(data, threshold=0, deductible=0, limit=float('inf')):
    # Filter data based on threshold
    filtered_data = data[data['Loss'] > threshold].copy()
    
    # Apply insurance layer if specified
    if deductible > 0 or limit < float('inf'):
        filtered_data['LayeredLoss'] = filtered_data['Loss'].apply(
            lambda x: apply_layer(x, deductible, limit)
        )
    else:
        filtered_data['LayeredLoss'] = filtered_data['Loss']
    
    # Calculate years of exposure
    total_years = filtered_data['Year'].nunique()
    
    # 1. Frequency Analysis
    frequency_by_year = (filtered_data.groupby('Year')
                        .agg({
                            'Loss': ['count', 'sum', 'mean', 'median'],
                            'LayeredLoss': ['sum', 'mean', 'median']
                        })
                        .round(2))
    frequency_by_year.columns = ['EventCount', 'TotalLoss', 'AvgLoss', 'MedianLoss',
                               'LayeredTotalLoss', 'LayeredAvgLoss', 'LayeredMedianLoss']
    frequency_by_year = frequency_by_year.reset_index()
    
    # Calculate average frequency
    avg_freq = filtered_data['Loss'].count() / total_years
    freq_stats = {
        'Average Annual Frequency': avg_freq,
        'Max Annual Events': frequency_by_year['EventCount'].max(),
        'Min Annual Events': frequency_by_year['EventCount'].min(),
        'Std Dev of Annual Events': frequency_by_year['EventCount'].std()
    }

    # 2. Severity Analysis
    severity_stats = (filtered_data.groupby('Year')
                     .agg({
                         'Loss': ['min', lambda x: x.quantile(0.25), 'median',
                                 lambda x: x.quantile(0.75), 'max', 'sum'],
                         'LayeredLoss': ['min', lambda x: x.quantile(0.25), 'median',
                                       lambda x: x.quantile(0.75), 'max', 'sum']
                     })
                     .round(2))
    severity_stats.columns = ['MinLoss', 'Q1Loss', 'MedianLoss', 'Q3Loss', 'MaxLoss', 'TotalLoss',
                            'LayeredMinLoss', 'LayeredQ1Loss', 'LayeredMedianLoss', 
                            'LayeredQ3Loss', 'LayeredMaxLoss', 'LayeredTotalLoss']
    severity_stats = severity_stats.reset_index()

    # 3. Peril Analysis
    peril_summary = (filtered_data.groupby('Peril')
                    .agg({
                        'Loss': ['count', 'sum', 'mean', 'median'],
                        'LayeredLoss': ['sum', 'mean', 'median']
                    })
                    .round(2))
    peril_summary.columns = ['EventCount', 'TotalLoss', 'AvgLoss', 'MedianLoss',
                           'LayeredTotalLoss', 'LayeredAvgLoss', 'LayeredMedianLoss']
    peril_summary = peril_summary.reset_index().sort_values('TotalLoss', ascending=False)

    return frequency_by_year, severity_stats, peril_summary, freq_stats, filtered_data

def create_plots(frequency_by_year, severity_stats, peril_summary, data, threshold, deductible, limit):
    # Frequency Plot
    freq_fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    freq_fig.add_trace(
        go.Bar(x=frequency_by_year['Year'], y=frequency_by_year['EventCount'],
               name="Event Count", marker_color='steelblue'),
        secondary_y=False
    )
    
    freq_fig.add_trace(
        go.Scatter(x=frequency_by_year['Year'], y=frequency_by_year['LayeredTotalLoss'],
                  name="Total Layered Loss", line=dict(color='red')),
        secondary_y=True
    )
    
    freq_fig.update_layout(
        title="Catastrophe Event Frequency and Layered Loss by Year",
        xaxis_title="Year",
        yaxis_title="Number of Events",
        yaxis2_title="Total Loss"
    )

    # Loss Distribution Plot
    fig_dist = make_subplots(rows=2, cols=1, subplot_titles=('Original Losses', 'Layered Losses'))
    
    fig_dist.add_trace(
        go.Histogram(x=data['Loss'], name='Original Losses', 
                    marker_color='steelblue', opacity=0.7),
        row=1, col=1
    )
    
    fig_dist.add_trace(
        go.Histogram(x=data['LayeredLoss'], name='Layered Losses',
                    marker_color='red', opacity=0.7),
        row=2, col=1
    )
    
    fig_dist.update_layout(height=600, title_text="Distribution of Losses")

    # Peril Loss Plot
    peril_plot = px.bar(
        peril_summary,
        x='Peril',
        y=['TotalLoss', 'LayeredTotalLoss'],
        title='Total Losses by Peril',
        barmode='group',
        labels={'value': 'Loss Amount', 'Peril': 'Peril Type', 
                'variable': 'Loss Type'}
    )

    return freq_fig, fig_dist, peril_plot

# Streamlit App
st.set_page_config(page_title="Catastrophe Loss Analysis", layout="wide")

st.title("Catastrophe Loss Analysis Dashboard")

# File upload
uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])

if uploaded_file is not None:
    try:
        # Load data
        data = pd.read_csv(uploaded_file)
        
        # Display raw data sample
        st.subheader("Raw Data Sample")
        st.write(data.head())
        
        # Ensure numeric types
        data['Year'] = pd.to_numeric(data['Year'], errors='coerce')
        data['Loss'] = pd.to_numeric(data['Loss'], errors='coerce')
        
        # Remove any rows with NA values
        data = data.dropna(subset=['Year', 'Loss', 'Peril'])
        
        # Analysis parameters in columns
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Threshold selector
            min_loss = data['Loss'].min()
            max_loss = data['Loss'].max()
            threshold = st.slider(
                "Select Loss Threshold",
                min_value=float(min_loss),
                max_value=float(max_loss),
                value=float(min_loss),
                format="%.2f"
            )
        
        with col2:
            # Deductible selector
            deductible = st.slider(
                "Select Deductible",
                min_value=0.0,
                max_value=float(max_loss),
                value=0.0,
                format="%.2f"
            )
        
        with col3:
            # Limit selector
            limit = st.slider(
                "Select Limit",
                min_value=float(min_loss),
                max_value=float(max_loss * 2),
                value=float(max_loss),
                format="%.2f"
            )
        
        # Perform analysis
        frequency_by_year, severity_stats, peril_summary, freq_stats, analyzed_data = analyze_cat_losses(
            data, threshold, deductible, limit
        )
        
        # Create plots
        freq_fig, loss_dist, peril_plot = create_plots(
            frequency_by_year, severity_stats, peril_summary, analyzed_data, 
            threshold, deductible, limit
        )
        
        # Display frequency statistics
        st.subheader("Frequency Statistics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Average Annual Events", f"{freq_stats['Average Annual Frequency']:.2f}")
        with col2:
            st.metric("Maximum Annual Events", f"{freq_stats['Max Annual Events']:.0f}")
        with col3:
            st.metric("Minimum Annual Events", f"{freq_stats['Min Annual Events']:.0f}")
        with col4:
            st.metric("Std Dev of Annual Events", f"{freq_stats['Std Dev of Annual Events']:.2f}")
        
        # Display summary statistics
        st.subheader("Loss Statistics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Original Losses")
            st.write(f"Total Loss: ${analyzed_data['Loss'].sum():,.2f}")
            st.write(f"Average Loss: ${analyzed_data['Loss'].mean():,.2f}")
        
        with col2:
            st.subheader("Layered Losses")
            st.write(f"Total Loss: ${analyzed_data['LayeredLoss'].sum():,.2f}")
            st.write(f"Average Loss: ${analyzed_data['LayeredLoss'].mean():,.2f}")
        
        with col3:
            st.subheader("Most Frequent Peril")
            top_peril = peril_summary.iloc[0]
            st.write(f"Type: {top_peril['Peril']}")
            st.write(f"Events: {top_peril['EventCount']:.0f}")
        
        # Display plots
        st.plotly_chart(freq_fig, use_container_width=True)
        st.plotly_chart(loss_dist, use_container_width=True)
        st.plotly_chart(peril_plot, use_container_width=True)
        
        # Display detailed tables
        st.subheader("Detailed Analysis Tables")
        tab1, tab2, tab3 = st.tabs(["Frequency Analysis", "Severity Analysis", "Peril Analysis"])
        
        with tab1:
            st.dataframe(frequency_by_year)
        with tab2:
            st.dataframe(severity_stats)
        with tab3:
            st.dataframe(peril_summary)
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.error("Please ensure your CSV file contains the required columns: Year, Loss, and Peril")
else:
    st.info("Please upload a CSV file to begin the analysis.")
