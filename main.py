import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def time_it(func: callable) -> callable:
    """
    Decorator function to measure the execution time of a function.

    Args:
    func: The function to be executed

    Returns:
    wrapper: The wrapper function
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Execution time: {end_time - start_time} seconds")
        return result
    return wrapper

def load_data(filepath):
    """Loads a CSV file into a Pandas DataFrame."""
    try:
        df = pd.read_csv(filepath)
        print("Data loaded successfully.")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def clean_data(df):
    """Cleans the dataset by handling missing values and duplicates."""
    df.drop_duplicates(inplace=True)
    df.fillna(df.mean(numeric_only=True), inplace=True)
    print("Data cleaned successfully.")
    return df

def get_summary_statistics(df):
    """Returns summary statistics of the dataset."""
    return df.describe()

def plot_histogram(df, column):
    """Plots a histogram for a specified numerical column."""
    if column in df.columns and np.issubdtype(df[column].dtype, np.number):
        plt.hist(df[column], bins=20, edgecolor='black')
        plt.xlabel(column)
        plt.ylabel("Frequency")
        plt.title(f"Histogram of {column}")
        plt.show()
    else:
        print("Invalid column for histogram.")

def correlation_matrix(df):
    """Generates and displays a correlation matrix heatmap."""
    import seaborn as sns
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title("Correlation Matrix")
    plt.show()

def detect_outliers(df, column):
    """
    Detects outliers in a specified column using the IQR method.
    
    Args:
    df: The input DataFrame
    column: The column to detect outliers in

    Returns:
    outliers: DataFrame containing the outliers
    """
    if column in df.columns and np.issubdtype(df[column].dtype, np.number):
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        outliers = df[(df[column] < (Q1 - 1.5 * IQR)) | (df[column] > (Q3 + 1.5 * IQR))]
        return outliers
    else:
        print("Invalid column for outlier detection.")
        return None

def main():
    """Main function to execute the script."""
    filepath = "./data/data.csv"  # Modify as needed
    df = load_data(filepath)
    if df is not None:
        df = clean_data(df)
        print(get_summary_statistics(df))
        
        column_to_plot = df.columns[0]  # Change column index as needed
        plot_histogram(df, column_to_plot)
        
        # Time how long correlation matrix takes
        correlation_matrix_timed = time_it(correlation_matrix)
        correlation_matrix_timed(df)
        
        column_for_outliers = df.columns[0]  # Change column index as needed
        outliers = detect_outliers(df, column_for_outliers)
        if outliers is not None:
            print("Detected outliers:")
            print(outliers)

if __name__ == "__main__":
    main()
