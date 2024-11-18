from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore , Qdrant
from qdrant_client import QdrantClient, models
from dotenv import load_dotenv
load_dotenv()
import os

class QdrantInsertRetrievalAll:
    def __init__(self, api_key, url):
        """
        Initialize the QdrantInsertRetrievalAll class with API key and URL.
        
        Parameters:
        - api_key (str): The API key for authenticating with the Qdrant service.
        - url (str): The URL endpoint for the Qdrant instance.
        """
        self.url = url 
        self.api_key = api_key

    # Method to insert documents into Qdrant vector store
    def insertion(self, text, embeddings, collection_name):
        """
        Inserts documents into the Qdrant vector store.
        
        Parameters:
        - text (list): The list of documents to be inserted into the collection.
        - embeddings (object): The embedding model to generate vectors for the text.
        - collection_name (str): The name of the collection to insert the documents into.
        
        Returns:
        - qdrant (QdrantVectorStore): The Qdrant vector store object after insertion.
        """
        qdrant = QdrantVectorStore.from_documents(
            text,
            embeddings,
            url=self.url,
            prefer_grpc=True,
            api_key=self.api_key,
            collection_name=collection_name,
        )
        print("Insertion successful")
        return qdrant

    # Method to retrieve documents from Qdrant vector store
    def retrieval(self, collection_name, embeddings):
        """
        Retrieves documents from the Qdrant vector store.
        
        Parameters:
        - collection_name (str): The name of the collection from which to retrieve documents.
        - embeddings (object): The embedding model to generate vectors for querying.
        
        Returns:
        - qdrant_store (Qdrant): The Qdrant vector store with the specified collection.
        """
        qdrant_client = QdrantClient(
            url=self.url,
            api_key=self.api_key,
        )
        qdrant_store = Qdrant(qdrant_client, collection_name=collection_name, embeddings=embeddings)
        return qdrant_store
    
    # Method to delete a collection from Qdrant
    def delete_collection(self, collection_name):
        """
        Deletes a collection from Qdrant.
        
        Parameters:
        - collection_name (str): The name of the collection to be deleted.
        
        Returns:
        - collection_name (str): The name of the deleted collection.
        """
        qdrant_client = QdrantClient(
            url=self.url,
            api_key=self.api_key,
        )
        qdrant_client.delete_collection(collection_name)
        return collection_name
    
    # Method to create a new collection in Qdrant with cosine similarity
    def create_collection(self, collection_name):
        """
        Creates a new collection in Qdrant with the cosine similarity metric.
        
        Parameters:
        - collection_name (str): The name of the collection to be created.
        
        Returns:
        - collection_name (str): The name of the newly created collection.
        """
        qdrant_client = QdrantClient(
            url=self.url,
            api_key=self.api_key,
        )
        qdrant_client.create_collection(
            collection_name,
            vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE)
        )
        print(f"Your collection {collection_name} created successfully")
        return collection_name
