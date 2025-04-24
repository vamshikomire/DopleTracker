import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
import time
from model import predict_class
from utils import validate_csv, process_signal
from database import save_classification_result
from history import display_classification_history

# Add sample data path for demo purposes
SAMPLE_DATA_PATH = "test_micro_doppler_signals.csv"

# Set page configuration
st.set_page_config(
    page_title="Micro-Doppler Target Classification",
    page_icon="🎯",
    layout="wide",
)

# Create a navigation sidebar 
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Classification", "History"])

# Main title based on selected page
if page == "Classification":
    st.title("Micro-Doppler Target Classification")

# Function to process and display the signal data
def process_and_display_signal(data_source, is_file=True):
    # Container for the main content
    main_container = st.container()
    
    with main_container:
        # Create a progress indicator
        progress = st.progress(0)
        status_text = st.empty()
        
        try:
            # Update progress
            status_text.text("Validating data...")
            progress.progress(10)
            time.sleep(0.5)  # Simulate processing time
            
            # Different handling based on source type (file upload or sample data)
            if is_file:
                # Show file details
                file_size = data_source.size / (1024 * 1024)  # Convert to MB
                st.write(f"**File name:** {data_source.name} ({file_size:.2f}MB)")
                
                # Validate CSV format
                is_valid, error_msg = validate_csv(data_source)
                
                if not is_valid:
                    st.error(f"Invalid file: {error_msg}")
                    return
                    
                # Read the CSV file
                df = pd.read_csv(data_source, header=None)
            else:
                # Read the sample data file
                df = pd.read_csv(data_source, header=None)
                st.write("**Using sample data for demonstration**")
                
            # Process the signal
            status_text.text("Processing signal...")
            progress.progress(30)
            
            # Process the signal data
            signal_data = process_signal(df)
            
            # Update progress
            status_text.text("Analyzing micro-Doppler signature...")
            progress.progress(60)
            
            # Display time-domain signal
            st.subheader("Signal Data (Time-Domain)")
            
            # Create a time-domain plot using Plotly
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=signal_data['time'],
                    y=signal_data['amplitude'],
                    mode='lines',
                    line=dict(color='#1e88e5', width=1),
                ))
            
            # Update layout
            fig.update_layout(
                height=300,
                margin=dict(l=0, r=0, t=0, b=0),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(
                    title="Time",
                    showgrid=True,
                    gridcolor='rgba(255,255,255,0.1)',
                ),
                yaxis=dict(
                    title="Amplitude",
                    showgrid=True,
                    gridcolor='rgba(255,255,255,0.1)',
                ),
            )
            
            # Display the plot
            st.plotly_chart(fig, use_container_width=True)
            
            # Update progress
            status_text.text("Classifying target...")
            progress.progress(90)
            
            # Perform classification
            result, confidence = predict_class(signal_data)
            
            # Display classification result
            progress.progress(100)
            status_text.text("Classification complete!")
            
            # Show results in a nice format
            st.subheader("Classification Result")
            
            # Create columns for better layout
            col1, col2 = st.columns(2)
            
            with col1:
                if result == "drone":
                    st.markdown(f"### 🛸 Drone Detected")
                else:
                    st.markdown(f"### 🐦 Bird Detected")
            
            with col2:
                st.metric("Confidence", f"{confidence:.2f}%")
            
            # Additional details about the classification
            st.subheader("Analysis Details")
            st.write("""
            The classification is based on micro-Doppler signature analysis. 
            Drones typically exhibit distinct patterns due to their rotating propellers, 
            while birds show different signatures from their wing movements.
            """)
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

# Main content based on selected page
if page == "Classification":
    # Subtitle with emojis
    st.markdown("### 🛸 Identify if the Signal is from a Drone or a Bird")
    
    # Instruction text
    st.write("Upload a signal file to classify it based on Micro-Doppler signatures.")
    
    # Add sample data option
    col1, col2 = st.columns([1, 1])
    with col1:
        # File uploader
        st.subheader("Upload Signal File (CSV)")
        uploaded_file = st.file_uploader(
            "Drag and drop file here",
            type=["csv"],
            help="Upload a CSV file containing time-domain signal data. Maximum file size: 200MB."
        )
    with col2:
        st.subheader("Or Use Sample Data")
        use_sample_data = st.button("Load Sample Data", type="primary")
        st.caption("Click to load provided test data for demonstration")
    
    # Main classification content logic
    if uploaded_file is not None:
        # File size validation (200MB limit as shown in the image)
        file_size = uploaded_file.size / (1024 * 1024)  # Convert to MB
        if file_size > 200:
            st.error(f"File size ({file_size:.2f}MB) exceeds the 200MB limit.")
        else:
            try:
                # Process and classify the signal
                process_and_display_signal(uploaded_file, is_file=True)
                
                # Save to database
                # We avoid re-reading the file and just use the result from the process function
                df = pd.read_csv(uploaded_file, header=None)
                signal_data = process_signal(df)
                result, confidence = predict_class(signal_data)
                filename = uploaded_file.name
                if save_classification_result(filename, False, result, confidence):
                    st.success("Classification result saved to history!")
                else:
                    st.warning("Classification result was not saved to history due to a database error.")
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
            
    # Handle sample data button
    elif use_sample_data:
        try:
            # Process and classify the sample data
            process_and_display_signal(SAMPLE_DATA_PATH, is_file=False)
            
            # Save to database (get the result directly)
            df = pd.read_csv(SAMPLE_DATA_PATH, header=None)
            signal_data = process_signal(df)
            result, confidence = predict_class(signal_data)
            if save_classification_result("Sample Data", True, result, confidence):
                st.success("Classification result saved to history!")
            else:
                st.warning("Classification result was not saved to history due to a database error.")
        except Exception as e:
            st.error(f"Error processing sample data: {str(e)}")

elif page == "History":
    # Display the classification history page
    display_classification_history()

# Add information about the application (only on Classification page)
if page == "Classification":
    with st.expander("About this application"):
        st.write("""
        ### Micro-Doppler Target Classification
        
        This application uses machine learning to analyze signal data and classify 
        whether the target is a drone or a bird based on micro-Doppler signatures.
        
        **How it works:**
        1. Upload a CSV file containing signal data
        2. The application processes the time-domain signal
        3. Features are extracted from the micro-Doppler signature
        4. A machine learning model classifies the target
        5. Results are displayed with confidence level
        6. Classification history is stored in database
        
        **Technical details:**
        - Signal processing using NumPy and Pandas
        - Machine learning with Scikit-learn
        - Visualization with Plotly
        - PostgreSQL database for storing results
        - Web application built with Streamlit
        """)

# Footer
st.markdown("---")
st.markdown("""
    <div style="text-align: center;">
        <p>Micro-Doppler Target Classification System</p>
        <div style="margin-top: 15px; padding: 10px; background-color: rgba(70, 70, 70, 0.1); border-radius: 5px; display: inline-block;">
            <p style="font-size: 0.95em; margin: 0; font-weight: bold;">Developed by:</p>
            <p style="font-size: 0.9em; margin-top: 8px;">Vamshi Komire, 217Y1A3357</p>
            <p style="font-size: 0.9em; margin-top: 5px;">Sri Charan Reddy, 217Y1A3352</p>
        </div>
    </div>
    """,
            unsafe_allow_html=True)
