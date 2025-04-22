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
