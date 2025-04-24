import pandas as pd
import numpy as np
import io

def validate_csv(file_obj):
    """
    Validate if the uploaded file is a properly formatted CSV file containing signal data.
    
    Parameters:
    -----------
    file_obj : UploadedFile object
        The file uploaded by the user
        
    Returns:
    --------
    is_valid : bool
        True if the file is valid, False otherwise
    error_msg : str
        Error message if the file is invalid, empty string otherwise
    """
    # Reset file pointer to the beginning
    file_obj.seek(0)
    
    try:
        # Try to read the file as CSV
        # Handle files without headers (like our test data)
        df = pd.read_csv(file_obj, header=None)
        
        # Reset file pointer again for future use
        file_obj.seek(0)
        
        # Check if the dataframe is empty
        if df.empty:
            return False, "The file is empty."
        
        # Check for minimum number of rows
        if len(df) < 5:
            return False, "The file contains too few data points (minimum 5 required)."
        
        # Ensure we have numerical data
        try:
            # Convert all columns to numeric, coercing errors to NaN
            df = df.apply(pd.to_numeric, errors='coerce')
            
            # Check if we have too many NaN values
            if df.isna().sum().sum() > 0.5 * df.size:
                return False, "Too many non-numeric values in the file."
        except Exception:
            return False, "File contains invalid data format. Please ensure data is numeric."
        
        return True, ""
        
    except pd.errors.ParserError:
        return False, "The file is not a valid CSV file."
    except pd.errors.EmptyDataError:
        return False, "The file is empty."
    except Exception as e:
        return False, f"Error validating file: {str(e)}"

def process_signal(df):
    """
    Process the signal data from the dataframe.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the signal data
        
    Returns:
    --------
    signal_data : dict
        Dictionary containing 'time' and 'amplitude' arrays
    """
    # For our test data with no headers, we need a different approach
    
    # First, ensure all data is numeric
    df = df.apply(pd.to_numeric, errors='coerce')
    
    # For test data, where we have multiple rows of signal data
    # We'll use the first row as the signal for now
    # In a more sophisticated app, we would process all rows or let the user select
    
    # Generate a time axis (0 to n-1)
    time = np.arange(len(df.columns))
    
    # Get the first row as amplitude data
    amplitude = df.iloc[0].values
    
    # Normalize amplitude to range approximately from -1 to 1
    # This is helpful for consistent visualization and feature extraction
    max_amp = max(abs(np.nanmax(amplitude)), abs(np.nanmin(amplitude)))
    if max_amp > 0:
        amplitude = amplitude / max_amp
    
    # Return the processed signal data
    return {
        'time': time,
        'amplitude': amplitude,
        'raw_df': df  # Include the raw DataFrame for reference if needed
    }
