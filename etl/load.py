# Importing necessary modules for processing documents
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings, SentenceTransformerEmbeddings
import pandas as pd
from dotenv import load_dotenv
import os

# Load environment variables from the .env file
load_dotenv()

def process_documents():
    """ Processes documents from a CSV file and stores them in a Chroma database.

    This function reads the dataset from the specified file path, extracts the documents,
    and the metadata, and then stores them in a Chroma database with embeddings.
    """
    # Construct the file path using environment variables
    file_path = os.path.join(os.environ['LOCAL_DIRECTORY'], os.environ['DATASET_NAME'])
    
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)

    # Extract the document column as text and convert to strings
    splits = df.document.tolist()
    splits_strings = [str(element) for element in splits]

    # Extract the rest of the columns as metadata
    metadata = df.drop(columns='document', axis=1)
    metadata = metadata.to_dict(orient='records')

    # Ensure the text and metadata have the same length
    assert len(splits_strings) == len(metadata)

    # Initialize the embeddings using a specific model
    embeddings = HuggingFaceEmbeddings(model_name="multi-qa-MiniLM-L6-cos-v1")

    # Create a Chroma database from the texts, metadata, and embeddings
    db = Chroma.from_texts(
        texts=splits_strings,
        metadatas=metadata,
        persist_directory=os.environ['PERSIST_DIRECTORY'],
        embedding=embeddings,
        collection_name=os.environ['COLLECTION']
        )

# Usage example
"""
if __name__ == '__main__':
    process_documents()
"""
