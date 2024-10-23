import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import scipy.stats as stats  # Import scipy.stats for statistical fitting

def analyze_cat_losses(data, threshold=0):
    # Filter data based on threshold
    filtered_data = data[data['Loss'] > threshold].copy()
    
    # Calculate years of exposure
    total_years = filtered_data['Year'].nunique()
    
    # 1. Frequency Analysis
    frequency_by_year = (filtered_data.groupby('Year')
                        .agg({
                            'Loss': ['count', 'sum', 'mean', 'median']
                        })
                        .round(2))
    frequency_by_year.columns = ['EventCount', 'TotalLoss', 'AvgLoss', 'MedianLoss']
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

    return frequency_by_year, severity_stats, peril_summary, freq_stats, filtered_data

def compute_oep_aep(shape_param, scale_param, avg_frequency, max_loss, num_simulations=10000):
    # Define return periods
    return_periods = [1, 2, 5, 10, 25, 50, 100, 250]
    
    # Compute Single Loss for Return Periods (OEP)
    single_loss = {}
    for rp in return_periods:
        probability = 1 - 1/rp
        loss = stats.pareto.ppf(probability, shape_param, loc=0, scale=scale_param)
        single_loss[rp] = loss
    single_loss_table = pd.DataFrame({
        'Return Period (Years)': return_periods,
        'Single Loss Threshold': [single_loss[rp] for rp in return_periods]
    })

    # Compute Aggregate Yearly Loss for Return Periods (AEP) via Monte Carlo Simulation
    # Simulate aggregate losses
    event_counts = np.random.poisson(lam=avg_frequency, size=num_simulations)
    aggregate_losses = np.zeros(num_simulations)
    
    # Identify simulations with at least one event
    non_zero_indices = event_counts > 0
    non_zero_event_counts = event_counts[non_zero_indices]
    
    if shape_param is not None and scale_param is not None and np.any(non_zero_indices):
        unique_counts, counts = np.unique(non_zero_event_counts, return_counts=True)
        for uc, cnt in zip(unique_counts, counts):
            if uc > 0:
                # Sample uc losses for cnt simulations
                sampled_losses = stats.pareto.rvs(shape_param, loc=0, scale=scale_param, size=(cnt, uc))
                # Sum losses for each simulation
                aggregate_losses[non_zero_indices][event_counts[non_zero_indices] == uc] += sampled_losses.sum(axis=1)
    
    # Define thresholds for AEP (based on return periods)
    aep_thresholds = []
    for rp in return_periods:
        probability = 1 - 1/rp
        loss = np.percentile(aggregate_losses, probability*100)
        aep_thresholds.append(loss)
    aggregate_loss_table = pd.DataFrame({
        'Return Period (Years)': return_periods,
        'Aggregate Loss Threshold': aep_thresholds
    })
    
    return single_loss_table, aggregate_loss_table

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

    # ECDF Plot
    fig_ecdf = px.ecdf(data, x='Loss', title='Empirical Cumulative Distribution of Losses')
    
    # Fit Pareto Type I Distribution starting at the threshold
    positive_losses = data['Loss']
    
    # Fit the Pareto distribution with scale fixed at the threshold
    # The 'b' parameter is the shape parameter (alpha)
    try:
        # Ensure that all losses are greater than or equal to the threshold
        losses_for_fitting = positive_losses[positive_losses >= threshold]
        if len(losses_for_fitting) < 2:
            raise ValueError("Not enough data points above the threshold to fit the Pareto distribution.")
        
        # Fit the Pareto distribution; fix loc=0 and scale=threshold
        params = stats.pareto.fit(losses_for_fitting, floc=0, fscale=threshold)
        shape_param = params[0]  # alpha
        loc_param = params[1]
        scale_param = params[2]
    except Exception as e:
        st.error(f"Error fitting Pareto distribution: {e}")
        shape_param, loc_param, scale_param = None, None, None
    
    # Initialize alpha
    alpha = None
    
    if shape_param is not None:
        # Generate theoretical CDF from the fitted distribution up to max_loss
        max_loss = data['Loss'].max()
        x = np.linspace(threshold, max_loss, 1000)  # x from threshold to maximum loss
        cdf_fitted = stats.pareto.cdf(x, shape_param, loc=loc_param, scale=scale_param)
    
        # Add the theoretical CDF to the ECDF plot
        fig_ecdf.add_trace(
            go.Scatter(
                x=x,
                y=cdf_fitted,
                mode='lines',
                name='Fitted Pareto CDF',
                line=dict(color='red')
            )
        )
    
        # Update x-axis range to 0-20M
        fig_ecdf.update_xaxes(range=[0, 20000000])
        fig_ecdf.update_layout(
            xaxis_title='Loss Amount',
            yaxis_title='Cumulative Probability',
            legend_title='Legend'
        )
        
        # Assign alpha
        alpha = shape_param
    else:
        st.warning("Pareto distribution could not be fitted to the data.")
    
    # Peril Loss Plot
    peril_plot = px.bar(
        peril_summary,
        x='Peril',
        y='TotalLoss',
        title='Total Losses by Peril',
        labels={'TotalLoss': 'Loss Amount', 'Peril': 'Peril Type'}
    )

    return freq_fig, fig_ecdf, peril_plot, alpha

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
        
        # Analysis parameters
        st.subheader("Analysis Parameters")
        col1 = st.columns(1)[0]
        
        with col1:
            # Threshold selector as number input
            min_loss = data['Loss'].min()
            max_loss = data['Loss'].max()
            threshold = st.number_input(
                "Enter Loss Threshold",
                min_value=float(min_loss),
                max_value=float(max_loss),
                value=float(min_loss),
                format="%.2f"
            )
        
        # Perform analysis
        frequency_by_year, severity_stats, peril_summary, freq_stats, analyzed_data = analyze_cat_losses(
            data, threshold
        )
        
        # Create plots and get the alpha parameter
        freq_fig, ecdf_fig, peril_plot, alpha = create_plots(
            frequency_by_year, severity_stats, peril_summary, analyzed_data, 
            threshold
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
            st.subheader("Total Losses")
            st.write(f"Total Loss: ${analyzed_data['Loss'].sum():,.2f}")
            st.write(f"Average Loss: ${analyzed_data['Loss'].mean():,.2f}")
            if alpha is not None:
                st.metric("Fitted Pareto Alpha (Shape)", f"{alpha:.4f}")
        
        with col2:
            st.subheader("Most Frequent Peril")
            if not peril_summary.empty:
                top_peril = peril_summary.iloc[0]
                st.write(f"Type: {top_peril['Peril']}")
                st.write(f"Events: {top_peril['EventCount']:.0f}")
            else:
                st.write("No data available.")
        
        with col3:
            if alpha is not None:
                st.subheader("Pareto Distribution")
                st.write(f"**Alpha (Shape Parameter):** {alpha:.4f}")
            else:
                st.subheader("Pareto Distribution")
                st.write("Pareto distribution not fitted.")
        
        # Display plots
        st.plotly_chart(freq_fig, use_container_width=True)
        st.plotly_chart(ecdf_fig, use_container_width=True)
        st.plotly_chart(peril_plot, use_container_width=True)
        
        # Compute OEP and AEP tables if Pareto was fitted
        if alpha is not None:
            with st.spinner("Computing Single Loss and Aggregate Yearly Loss for Return Periods..."):
                single_loss_table, aggregate_loss_table = compute_oep_aep(
                    shape_param=alpha,
                    scale_param=threshold,
                    avg_frequency=freq_stats['Average Annual Frequency'],
                    max_loss=data['Loss'].max(),
                    num_simulations=10000  # Adjust number of simulations as needed
                )
        else:
            single_loss_table, aggregate_loss_table = pd.DataFrame(), pd.DataFrame()
        
        # Display detailed tables
        st.subheader("Detailed Analysis Tables")
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Frequency Analysis", 
            "Severity Analysis", 
            "Peril Analysis", 
            "Single Loss (Return Periods)", 
            "Aggregate Yearly Loss (Return Periods)"
        ])
        
        with tab1:
            st.dataframe(frequency_by_year)
        with tab2:
            st.dataframe(severity_stats)
        with tab3:
            st.dataframe(peril_summary)
        with tab4:
            st.write("### Single Loss for Given Return Periods")
            st.dataframe(single_loss_table)
        with tab5:
            st.write("### Aggregate Yearly Loss for Given Return Periods")
            st.dataframe(aggregate_loss_table)
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.error("Please ensure your CSV file contains the required columns: Year, Loss, and Peril")
else:
    st.info("Please upload a CSV file to begin the analysis.")
