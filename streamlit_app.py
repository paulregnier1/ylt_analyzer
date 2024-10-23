import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

def analyze_cat_losses(data, threshold=0):
    # Display column names for debugging
    st.write("Available columns:", data.columns.tolist())
    
    # Ensure required columns exist
    required_columns = {'Year', 'Loss', 'Peril'}
    missing_columns = required_columns - set(data.columns)
    
    if missing_columns:
        st.error(f"Missing required columns: {missing_columns}")
        st.stop()
    
    # Filter data based on threshold
    filtered_data = data[data['Loss'] > threshold].copy()
    
    # 1. Frequency Analysis
    frequency_by_year = (filtered_data.groupby('Year')
                        .agg({
                            'Loss': ['count', 'sum', 'mean', 'median']
                        })
                        .round(2))
    frequency_by_year.columns = ['EventCount', 'TotalLoss', 'AvgLoss', 'MedianLoss']
    frequency_by_year = frequency_by_year.reset_index()

    # 2. Severity Analysis
    severity_stats = (filtered_data.groupby('Year')
                     .agg({
                         'Loss': ['min', lambda x: x.quantile(0.25), 'median',
                                 lambda x: x.quantile(0.75), 'max', 'sum']
                     })
                     .round(2))
    severity_stats.columns = ['MinLoss', 'Q1Loss', 'MedianLoss', 'Q3Loss', 'MaxLoss', 'TotalLoss']
    severity_stats = severity_stats.reset_index()

    # 3. Peril Analysis
    peril_summary = (filtered_data.groupby('Peril')
                    .agg({
                        'Loss': ['count', 'sum', 'mean', 'median']
                    })
                    .round(2))
    peril_summary.columns = ['EventCount', 'TotalLoss', 'AvgLoss', 'MedianLoss']
    peril_summary = peril_summary.reset_index().sort_values('TotalLoss', ascending=False)

    return frequency_by_year, severity_stats, peril_summary

def create_plots(frequency_by_year, severity_stats, peril_summary, data, threshold):
    # Frequency Plot
    freq_fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    freq_fig.add_trace(
        go.Bar(x=frequency_by_year['Year'], y=frequency_by_year['EventCount'],
               name="Event Count", marker_color='steelblue'),
        secondary_y=False
    )
    
    freq_fig.add_trace(
        go.Scatter(x=frequency_by_year['Year'], y=frequency_by_year['TotalLoss'],
                  name="Total Loss", line=dict(color='red')),
        secondary_y=True
    )
    
    freq_fig.update_layout(
        title="Catastrophe Event Frequency and Total Loss by Year",
        xaxis_title="Year",
        yaxis_title="Number of Events",
        yaxis2_title="Total Loss"
    )

    # Loss Distribution Plot
    loss_dist = px.histogram(
        data[data['Loss'] > threshold], x='Loss',
        title='Distribution of Losses',
        labels={'Loss': 'Loss Amount', 'count': 'Frequency'},
        nbins=30
    )
    loss_dist.update_traces(marker_color='steelblue', opacity=0.7)

    # Peril Loss Plot
    peril_plot = px.bar(
        peril_summary,
        x='Peril', y='TotalLoss',
        title='Total Losses by Peril',
        labels={'TotalLoss': 'Total Loss', 'Peril': 'Peril Type'}
    )
    peril_plot.update_traces(marker_color='steelblue')

    return freq_fig, loss_dist, peril_plot

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
        
        # Verify required columns
        required_columns = {'Year', 'Loss', 'Peril'}
        if not all(col in data.columns for col in required_columns):
            st.error(f"CSV file must contain these columns: {required_columns}")
            st.stop()
        
        # Ensure numeric types
        data['Year'] = pd.to_numeric(data['Year'], errors='coerce')
        data['Loss'] = pd.to_numeric(data['Loss'], errors='coerce')
        
        # Remove any rows with NA values
        data = data.dropna(subset=['Year', 'Loss', 'Peril'])
        
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
        
        # Perform analysis
        frequency_by_year, severity_stats, peril_summary = analyze_cat_losses(data, threshold)
        
        # Create plots
        freq_fig, loss_dist, peril_plot = create_plots(
            frequency_by_year, severity_stats, peril_summary, data, threshold
        )
        
        # Display summary statistics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Events Above Threshold")
            st.write(f"Total Events: {len(data[data['Loss'] > threshold])}")
            st.write(f"Total Loss: ${data[data['Loss'] > threshold]['Loss'].sum():,.2f}")
        
        with col2:
            st.subheader("Average Metrics")
            st.write(f"Average Loss: ${data[data['Loss'] > threshold]['Loss'].mean():,.2f}")
            st.write(f"Median Loss: ${data[data['Loss'] > threshold]['Loss'].median():,.2f}")
        
        with col3:
            st.subheader("Most Frequent Peril")
            top_peril = peril_summary.iloc[0]
            st.write(f"Type: {top_peril['Peril']}")
            st.write(f"Total Loss: ${top_peril['TotalLoss']:,.2f}")
        
        # Display plots
        st.plotly_chart(freq_fig, use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(loss_dist, use_container_width=True)
        with col2:
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
