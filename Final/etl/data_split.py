from haystack.nodes import PDFToTextConverter, PreProcessor
import numpy as np
import pandas as pd
import os
import text_normalizer
import config
import unicodedata

# Add the is_pua function here
def is_pua(c):
    return unicodedata.category(c) == 'Co'


class PDFProcessor:
    """
    A class to convert PDF files to text and preprocess the text content.
    """
    def __init__(self, remove_numeric_tables=True, valid_languages=["en"], split_length=1000, split_overlap=200):
        """
        Initialize the PDFProcessor object.

        Args:
            remove_numeric_tables (bool): Flag to remove numeric tables during PDF to text conversion.
            valid_languages (list): List of valid languages for PDF conversion.
            split_length (int): Maximum length of each split during text preprocessing.
            split_overlap (int): Overlap length between splits during text preprocessing.
        """

        self.converter = PDFToTextConverter(remove_numeric_tables=remove_numeric_tables, valid_languages=valid_languages)
        self.preprocessor = PreProcessor(clean_empty_lines=True, clean_whitespace=True, clean_header_footer=False,
                                         split_by="word", split_length=split_length, split_overlap=split_overlap,
                                         split_respect_sentence_boundary=True, add_page_number=True)

    def process_pdf(self, file_path):
        """
        Convert a PDF file to text, preprocess it, and return processed Haystack document objects.

        Parameters:
            file_path (str): The path to the PDF file.

        Returns:
            list of Document: A list of processed Haystack Document objects containing text content and metadata.
        """
        try:
            # Convert the PDF to text
            doc_pdf = self.converter.convert(file_path=file_path, meta=None)[0]

            # Preprocess the text
            docs_default = self.preprocessor.process([doc_pdf])

            # Process and append the text content to the 'processed_docs' list
            processed_docs = [doc.content for doc in docs_default]

            # Apply the is_pua filter to remove private use area characters
            processed_docs = ["".join([char for char in content if not is_pua(char)]) for content in processed_docs]

            processed_docs = text_normalizer.normalize_corpus(processed_docs, html_stripping=True,
                                                              contraction_expansion=True,
                                                              accented_char_removal=True,
                                                              text_lower_case=True, text_stemming=False,
                                                              text_lemmatization=False,
                                                              special_char_removal=False,
                                                              remove_digits=False, stopword_removal=False)

            for i in range(len(docs_default)):
                # Normalize text content from the Document object
                docs_default[i].content = processed_docs[i]

            return docs_default
        except Exception as e:
            print(f"Error processing PDF: {file_path}")
            print(f"Error message: {e}")
            return []


class Dataset:
    """
    A class to load and process a folder containing PDFs into a pandas DataFrame.
    """
    def __init__(self, pdf_folder_path):
        """
        Initialize the Dataset object.

        Args:
            pdf_folder_path (str): Path to the folder containing the PDF files.
        """
        self.pdf_folder_path = pdf_folder_path
        self.pdf_processor = PDFProcessor()

    def load_and_process_folder(self):
        """
        Load and process the PDF files in the specified folder.

        Returns:
            pandas.DataFrame: Processed dataset containing Haystack document info, content, and corresponding PDF names.
        """
        # Get a list of PDF files in the specified folder
        pdf_files = [file for file in os.listdir(self.pdf_folder_path) if file.lower().endswith(".pdf")]

        # Initialize an empty list to store the processed data
        data = []

        # Loop through each PDF file in the folder
        for pdf_file in pdf_files:
            # Get the full file path of the PDF
            file_path = os.path.join(self.pdf_folder_path, pdf_file)

            # Get useful info from the name
            # Remove the file extension (.pdf) to get just the filename
            filename_without_extension = os.path.splitext(pdf_file)[0]
            # Split the filename using underscores
            parts_name = filename_without_extension.split('_')
            exchange = parts_name[0]
            symbol = parts_name[1]
            year = parts_name[2]

            # Process the PDF file using the PDFProcessor object
            processed_docs = self.pdf_processor.process_pdf(file_path)

            # Loop through each processed document obtained from the PDF
            for doc in processed_docs:
                # Extract relevant information from the processed document
                doc_info = {
                    "name": pdf_file,          # Name of the PDF file
                    "exchange": exchange,      # Exchange type
                    "symbol": symbol,          # Name of the company
                    "year": year,              # Year of the report
                    "document": doc.content,    # Content of the processed document
                    "meta": doc.meta,          # Meta information of the processed document (Haystack document info)
                }

                # Append the extracted document information to the 'data' list
                data.append(doc_info)

        # Create a pandas DataFrame from the 'data' list
        df = pd.DataFrame(data)
        return df


def process_and_save_dataset(pdf_folder_path, file_name='data.csv'):
    """
    Process PDF files in the specified folder and save the processed data to a CSV file.

    Args:
        pdf_folder_path (str): Path to the folder containing the PDF files.
        file_name (str): Name of the file where the pandas DataFrame will be saved.

    Returns:
        None
    """
    try:
        # Process the dataset
        pdf_folder_processor = Dataset(pdf_folder_path)
        processed_df = pdf_folder_processor.load_and_process_folder()

        # Read the "nasdaq_screener_1689942465749.csv" file
        nasdaq_df = pd.read_csv("aditional_information.csv")

        # Merge the datasets on the common column "symbol"
        merged_df = processed_df.merge(nasdaq_df[['symbol', 'Name', 'Market Cap', 'Sector', 'Industry', 'Country']],
                                       left_on='symbol', right_on='symbol', how='left')

        # Drop the duplicate "Symbol" column from the merge
        merged_df.drop(columns='symbol', inplace=True)

        # Flatten the 'meta' dictionary into separate columns
        metadata_df = pd.json_normalize(merged_df['meta'])
        merged_df = pd.concat([merged_df.drop('meta', axis=1), metadata_df], axis=1)

        # Drop duplicates based on '_split_overlap'
        merged_df.drop_duplicates(subset=['_split_overlap', ], keep='first', inplace=True)

        # Reset the index of the DataFrame
        merged_df.reset_index(drop=True, inplace=True)

        # Save the DataFrame to a CSV file
        os.makedirs(pdf_folder_path, exist_ok=True)
        merged_df.to_csv(os.path.join(pdf_folder_path, file_name), index=False, escapechar='\\')

        # Return the merged DataFrame (optional)
        return merged_df
    except Exception as e:
            print(f"Error processing and saving dataset: {e}")
    return None

# Example usage:
if __name__ == "__main__":
    pdf_folder_path = Local_directory
    df = process_and_save_dataset(pdf_folder_path, file_name='dataset_mil.csv')
    if df is not None:
        print(df)


