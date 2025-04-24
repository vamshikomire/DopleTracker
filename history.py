import streamlit as st
import pandas as pd
import datetime
from database import get_all_classification_results, get_recent_classification_results, clear_classification_history

def display_classification_history():
    """
    Display the classification history page with results from the database
    """
    st.title("Classification History")
    
    # Show loading spinner while fetching data
    with st.spinner("Fetching classification history..."):
        # Get classification results from the database
        results = get_all_classification_results()
    
    if not results:
        if st.button("Refresh History"):
            st.rerun()
        st.info("No classification history found. Start classifying signals to see your history.")
        return
    
    # Add a refresh button
    if st.button("🔄 Refresh History"):
        st.rerun()
    
    # Create a dataframe from the results
    data = []
    try:
        for result in results:
            data.append({
                "ID": result.id,
                "Date": result.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "Source": "Sample Data" if result.is_sample_data else f"File: {result.filename}",
                "Classification": result.classification.title(),
                "Confidence": f"{result.confidence:.2f}%",
                "Notes": result.notes or ""
            })
    except Exception as e:
        st.error(f"Error processing results: {str(e)}")
        data = []
    
    df = pd.DataFrame(data)
    
    # Create statistics section
    st.subheader("Statistics")
    col1, col2, col3 = st.columns(3)
    
    total_classifications = len(results)
    drone_count = sum(1 for r in results if r.classification.lower() == "drone")
    bird_count = sum(1 for r in results if r.classification.lower() == "bird")
    
    with col1:
        st.metric("Total Classifications", total_classifications)
    with col2:
        st.metric("Drone Detections", drone_count)
    with col3:
        st.metric("Bird Detections", bird_count)
    
    # Display the table of results
    st.subheader("Classification History")
    st.dataframe(
        df,
        column_config={
            "ID": st.column_config.NumberColumn("ID", width="small"),
            "Date": st.column_config.DatetimeColumn("Date/Time", format="MMM DD, YYYY, hh:mm a"),
            "Classification": st.column_config.TextColumn("Classification"),
            "Confidence": st.column_config.ProgressColumn(
                "Confidence",
                help="Classification confidence level",
                format="%f",
                min_value=0,
                max_value=100,
            ),
        },
        use_container_width=True,
        hide_index=True,
    )
    
    # Create a filter section
    st.subheader("Filter by Classification")
    
    selected_filter = st.radio(
        "Show only:",
        ["All", "Drone", "Bird"],
        horizontal=True,
    )
    
    if selected_filter != "All":
        filtered_df = df[df["Classification"].str.lower() == selected_filter.lower()]
        if len(filtered_df) > 0:
            st.dataframe(
                filtered_df,
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info(f"No {selected_filter} classifications found.")
    
    # Add an option to clear history
    st.write("---")
    if st.button("Clear History", type="secondary"):
        # Confirm dialog using session state
        confirm_clear = st.warning("Are you sure you want to clear all classification history? This cannot be undone.")
        col1, col2 = st.columns([1, 5])
        with col1:
            if st.button("Yes, Clear All"):
                if clear_classification_history():
                    st.success("Classification history cleared successfully.")
                    st.rerun()  # Refresh the page
                else:
                    st.error("An error occurred while clearing the history.")
        with col2:
            if st.button("Cancel"):
                st.rerun()  # Refresh the page
