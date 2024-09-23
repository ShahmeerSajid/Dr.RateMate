
import pandas as pd
import csv
import logging
from typing import List, Optional

# Set up logging for better error tracking and debugging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_existing_csv(file_path: str) -> Optional[pd.DataFrame]:
    """
    Load existing CSV file into a DataFrame.
    
    This function uses a more efficient method to read CSV files by specifying
    the data types of columns, which can significantly speed up the loading process
    for large files.
    """
    try:
        # Specify dtypes for faster reading and less memory usage
        dtypes = {'Review': str}  # Adjust this based on your actual column names and types
        return pd.read_csv(file_path, dtype=dtypes)
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
    except pd.errors.EmptyDataError:
        logging.error(f"Empty CSV file: {file_path}")
    except Exception as e:
        logging.error(f"Error loading CSV: {str(e)}")
    return None

def create_new_comments_df(comments: List[str]) -> pd.DataFrame:
    """
    Create a DataFrame from new comments list.
    
    This function is optimized to create a DataFrame directly from a list,
    which is more efficient than creating it from a dictionary.
    """
    return pd.DataFrame(comments, columns=['Review'])

def append_and_save(existing_df: pd.DataFrame, new_df: pd.DataFrame, output_path: str) -> None:
    """
    Append new comments to existing DataFrame and save to CSV.
    
    This function uses pd.concat which is more efficient for combining DataFrames
    than using append. It also uses the 'to_csv' method with optimized parameters
    for faster writing and to ensure proper quoting of text data.
    """
    try:
        # Use concat instead of append for better performance
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        
        # Use efficient CSV writing options
        combined_df.to_csv(output_path, index=False, quoting=csv.QUOTE_NONNUMERIC, 
                           escapechar='\\', doublequote=False)
        logging.info(f"Updated file saved to {output_path}")
    except Exception as e:
        logging.error(f"Error saving CSV: {str(e)}")

def main():
    """
    Main function to orchestrate the CSV update process.
    
    This function demonstrates the use of the if __name__ == "__main__" idiom,
    which allows the script to be imported as a module without running the main code.
    """
    # File paths
    input_file_path = './combined_course_reviews.csv'
    output_file_path = './combined_course_reviews_updated.csv'

    # New comments as a list (truncated for brevity)
    new_comments = [
        "The professor's lectures were incredibly disorganized and hard to follow. They often seemed unprepared and lacked a clear structure. The course content was equally terrible, with outdated materials and assignments that seemed to have no real purpose.",
        "This was by far the worst course I have ever taken. The professor was not only unengaging but also seemed completely uninterested in teaching. Their explanations were often confusing and lacked depth.",
        # ... Add more comments here
    ]

    # Process the data
    existing_df = load_existing_csv(input_file_path)
    if existing_df is not None:
        new_comments_df = create_new_comments_df(new_comments)
        append_and_save(existing_df, new_comments_df, output_file_path)
    else:
        logging.error("Failed to load existing CSV. Process aborted.")

if __name__ == "__main__":
    """
    This conditional is used to check whether this script is being run directly or being imported.
    If the script is being run directly, __name__ is set to "__main__",
    and the main() function will be called.
    This allows for the script to be imported and used as a module without automatically running main().
    """
    main()






