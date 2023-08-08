from dotenv import load_dotenv
# Load environment variables from the .env file
load_dotenv()

# Import the required function for downloading from S3
from download import download_from_s3
# Download files from the specified S3 bucket
download_from_s3()

# Import the required functions for splitting the downloaded dataset
from split_dataset import process_and_save_dataset
# Define the folder path containing the PDF files
pdf_folder_path = os.environ['LOCAL_DIRECTORY']
# Process the PDF files and save the resulting DataFrame to a CSV file
df = process_and_save_dataset(pdf_folder_path, file_name=os.environ['DATASET_NAME'])

# Import the required function for processing the documents
from load import process_documents
# Process the documents using the specified function
process_documents()
