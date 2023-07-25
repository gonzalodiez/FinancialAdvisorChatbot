import chromadb
import pandas as pd


class chroma():
    def __init__(self, path=None, host=None, port=None):
        """
        Initialize Chroma object with either PersistentClient (local database) or HttpClient (Docker backend).

        Args:
            path (str, optional): Path to the local Chroma database. Defaults to None.
            host (str, optional): Host address for the Docker backend. Defaults to None.
            port (int, optional): Port number for the Docker backend. Defaults to None.
        
        Raises:
            ValueError: If neither 'path' nor bothnd 'port' are provided.
        """
        if path is not None:
            # Create PersistentClient for local database
            self.client = chromadb.PersistentClient(path=path)
        elif host is not None and port is not None:
            # Create HttpClient for Docker backend
            self.client = chromadb.HttpClient(host=host, port=port)
        else:
            raise ValueError("Either 'path' or both 'host' and 'port' must be provided.")

    def collection(self, collection, embedding_function):
        """
        Create a new collection in Chroma or get an existing collection.

        Args:
            collection (str): Name of the collection.
            embedding_function (Optional, Callable): Optional function for document embedding. Defaults to None.
        """
        print("If your client is running well then you should see a heartbeat greater than 0: ", self.client.heartbeat())
        self.collection = self.client.get_or_create_collection(
            collection,
            embedding_function=embedding_function
        )


    def add_data(self, csv, rows=None):
        """
        Add data from a CSV file to the Chroma collection.

        Args:
            csv (str): Path to the CSV file containing 'name', 'exchange', 'symbol', 'year', 'document',
                       '_split_id', and '_split_overlap' columns.
        """
        data = pd.read_csv(csv, nrows=rows)
        
        # Rename the first column without a name to 'ids'
        data = data.reset_index()
        ids=data.index.astype(str).tolist()

        # Extract the 'documents' column
        documents = data['document'].tolist()

        # Combine 'name', 'exchange', 'symbol', 'year', '_split_id', and '_split_overlap' columns into a 'meta' dictionary
        meta_data = data.drop(columns=['document','index'], axis=1).to_dict(orient='records')
        meta_data = [dict(row) for row in meta_data]

        
        # Add the data to the Chroma collection
        self.collection.add(
            documents=documents,
            metadatas=meta_data,
            ids=ids
        )

    def change_collection_name(self, collection, new_name):
        """
        Change the name of an existing Chroma collection.

        Args:
            collection (str): Name of the collection to rename.
            new_name (str): New name for the collection.
        """
        collections = self.client.list_collections(collection)
        # Rename the collection
        collections.modify(name=new_name)


    def retriever(
            self,
            query,
            n_results,
            metadata=None,
            context=None
            ):
        #if collection:
        #    self.client.get_collection(collection)
        #else:   "Define a collection"
        return self.collection.query(
            query_texts=query,          # ["doc10", "thus spake zarathustra", ...]
            n_results=n_results,        # 10
            where=metadata,             # {"metadata_field": "is_equal_to_this"}
            where_document=context,     # {"$contains":"search_string"}
            include=["embeddings", "metadatas", "documents", "distances"]         # what to include
            )
        

# Ejemplo de uso 

#from chromadb.utils import embedding_functions
#
##default_ef = embedding_functions.DefaultEmbeddingFunction()
#sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-mpnet-base-v2")
#
#path='/home/gonza/gonza/Financial_Chatbot_Advisor/chroma_new'
#chroma = Chroma(path=path)
#chroma.create_collection(collection='test',embedding_function=sentence_transformer_ef)
#chroma.add_data(csv='/home/gonza/gonza/Financial_Chatbot_Advisor/git/FinancialAdvisorChatbot/data.csv', rows=210)

