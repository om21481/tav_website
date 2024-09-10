import zipfile
from data import DataDownloader, DataGenerator, Analysis
import os

def create_zip(directory_path, zip_filename):
    """Create a zip file from the contents of a directory."""
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_STORED) as zipf:
        for root, _, files in os.walk(directory_path):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, directory_path)
                zipf.write(file_path, arcname)

def list_directories(folder_path):
    all_entries = os.listdir(folder_path)
    directories = [entry for entry in all_entries if os.path.isdir(os.path.join(folder_path, entry))]
    return directories

# GSE40680
gse_id = input("Enter gse id: ")

downloader = DataDownloader(gse_id)

file_path = downloader.dataDownloader()
print(file_path)
datagenerator = DataGenerator(file_path)
generated_path = datagenerator.dataGenerator()
print(generated_path)
generated_path="/Users/omgarg/Desktop/projects/techmedbuddy/tav_website/website/GSE40680"
folders = list_directories(generated_path)
folders_path = [os.path.join(generated_path, i) for i in folders]
print(folders_path)
print("COmpleted")
# for i in folders_path:
#     process = Analysis(i, gse_id)
#     process.process_directory()

# # Ensure directory path is not empty or invalid
# # if not os.path.isdir(directory_path):
# #     return "Invalid directory", 400

# # # Create a zip file of the directory contents
# zip_filename = f"{gse_id}.zip"
# create_zip(file_path, zip_filename)