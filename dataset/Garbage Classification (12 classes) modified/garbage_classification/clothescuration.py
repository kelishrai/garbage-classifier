import os

def delete_files(folder_path, num_files):
    # Get list of files in the folder
    files = os.listdir(folder_path)
    # Ensure there are enough files to delete
    if len(files) < num_files:
        print("Error: Not enough files in the folder to delete.")
        return
    # Delete the specified number of files
    for i in range(num_files):
        file_path = os.path.join(folder_path, files[i])
        os.remove(file_path)
        print(f"Deleted file: {file_path}")

# Specify the folder path and number of files to delete
folder_path = "clothes"
num_files_to_delete = 4000

# Check if the folder exists
if os.path.exists(folder_path):
    delete_files(folder_path, num_files_to_delete)
else:
    print("Error: Folder 'clothes' does not exist in the current directory.")
