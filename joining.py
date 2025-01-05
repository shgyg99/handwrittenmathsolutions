import os
import pandas as pd

# Path to the folder containing the CSV files
folder_path = "C:\\Users\\KARBAR\\myProject"  # Replace with the path to the folder containing your CSV files
output_file = "merged_sorted.csv"  # Name of the output file

# List all CSV files in the folder
csv_files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]

# Initialize an empty DataFrame to merge all files
merged_df = pd.DataFrame()

# Read and concatenate all CSV files
for file in csv_files:
    file_path = os.path.join(folder_path, file)
    temp_df = pd.read_csv(file_path, on_bad_lines="skip")
    merged_df = pd.concat([merged_df, temp_df], ignore_index=True)  # Append to the merged DataFrame

# Extract the image number from the first column (e.g., "im (number)")
merged_df["image_number"] = merged_df.iloc[:, 0].str.extract(r"im_\s*\((\d+)\)", expand=False).astype(int)

# Sort the DataFrame by the extracted image number
sorted_df = merged_df.sort_values(by="image_number")

# Drop the temporary column used for sorting
sorted_df = sorted_df.drop(columns=["image_number"])

# Save the sorted and merged DataFrame to a new CSV file
sorted_df.to_csv(output_file, index=False)

print(f"CSV files have been successfully merged and sorted. Output file: {output_file}")
