from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings, SentenceTransformerEmbeddings
import pandas as pd


df=pd.read_csv('dataset_v2.csv')

r_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=100 ,
    separators=["\n\n", "\n", "(?<=\. )", " ", ""],
    length_function=len
    )

# Use document column as text and turn it into strings
splits=df.document.tolist()
splits_strings = [str(elemento) for elemento in splits]

# Use the rest of the columns as metadata
metadata=df.drop(columns='document', axis=1)
metadata=metadata.to_dict(orient='records')

assert len(splits_strings)==len(metadata)

embeddings = HuggingFaceEmbeddings(model_name="multi-qa-MiniLM-L6-cos-v1")

db = Chroma.from_documents(
    documents=splits,
    persist_directory="chroma",   
    embedding=embeddings
)