import os
import csv
import re

def remove_sz_suffix_from_csv_files(directory_path):
    """
    Removes .sz or .SZ suffix from stock codes in all CSV files in the given directory.
    """
    csv_files = [f for f in os.listdir(directory_path) if f.lower().endswith('.csv')]
    
    for filename in csv_files:
        file_path = os.path.join(directory_path, filename)
        
        # Read the original CSV file
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # Use regex to find and replace .sz or .SZ at the end of stock codes
        # This pattern looks for .sz or .SZ at the end of numbers in the first column
        updated_content = re.sub(r'(\d+)\.(SZ|sz)', r'\1', content)
        
        # Write the updated content back to the file
        with open(file_path, 'w', encoding='utf-8', newline='') as file:
            file.write(updated_content)
        
        print(f"Processed {filename}")

if __name__ == "__main__":
    directory_path = r"C:\Users\Administrator\Desktop\cyzt\data"
    remove_sz_suffix_from_csv_files(directory_path)
    print("All CSV files have been processed to remove .sz suffix from stock codes.")