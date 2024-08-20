import pandas as pd

# Load the dataset
file_path = './combined_course_reviews.csv'

#file_path = '/Users/shahmeer/Desktop/Sentimental Analysis ML Project/combined_course_reviews.csv'# Update this path as necessary
df = pd.read_csv(file_path)

# Define the columns to count values for
rating_columns = ['Prof Rating', 'Course Rating']

# Initialize a dictionary to store the counts
rating_counts = {col: {i: 0 for i in range(-1, 6)} for col in rating_columns}

# Count the occurrences of each rating value in each column
for col in rating_columns:
    counts = df[col].value_counts().to_dict()
    for rating, count in counts.items():
        rating_counts[col][rating] = count

# Print the results
for col in rating_columns:
    print(f"\nCounts for {col}:")
    for rating in range(-1, 6):
        print(f"{rating}: {rating_counts[col][rating]}")
