from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
load_dotenv()
import os
pinecone_key = os.getenv("PINECONE_API_KEY")

class PineconeInsertRetrieval:
    def __init__(self, api_key):
        """
        Initializes the PineconeInsertRetrieval class with the provided API key.
        """
        self.api_key = api_key

    def check_index(self, index):
        """
        Checks if the specified index exists in Pinecone.
        
        Args:
            index (str): The name of the index to check.
        
        Returns:
            str: A message indicating whether the index is found or not.
        """
        pc = Pinecone(api_key=self.api_key)  # Initialize Pinecone client
        indexes = pc.list_indexes().names()  # Get the list of indexes
        if index not in indexes:  # If the index is not found
            return "Not Found index"
        elif index in indexes:  # If the index exists
            return f"Your index name {index} Found"
    
    def create_index(self, index_name, dimensions):
        """
        Creates a new index in Pinecone with the specified name and dimensions.
        
        Args:
            index_name (str): The name of the index to create.
            dimensions (int): The number of dimensions for the index.
        
        Returns:
            str: The name of the created index or an error message.
        """
        try:
            pc = Pinecone(api_key=self.api_key)  # Initialize Pinecone client
            # Create a new index with the specified parameters
            pc.create_index(
                name=index_name,
                dimension=dimensions,
                metric="cosine",  # Similarity metric used (Cosine similarity)
                spec=ServerlessSpec(  # Serverless configuration for the index
                    cloud='aws', 
                    region='us-east-1'
                )
            )
            print(f"Your index {index_name} created successfully")  # Success message
            return index_name
        except Exception as ex:  # Handle exceptions
            return f"Sorry, try again. {ex}"
        
    def delete_index_name(self, index_name):
        """
        Deletes the specified index from Pinecone.
        
        Args:
            index_name (str): The name of the index to delete.
        
        Returns:
            str: A success or error message.
        """
        try:
            pc = Pinecone(api_key=self.api_key)  # Initialize Pinecone client
            indexes = pc.list_indexes().names()  # Get the list of indexes
            if index_name not in indexes:  # If the index doesn't exist
                return f"Index '{index_name}' does not exist."
            pc.delete_index(index_name)  # Delete the specified index
            return f"Index '{index_name}' deleted successfully."
        except Exception as ex:  # Handle exceptions
            return f"Failed to delete index '{index_name}': {ex}"

    def delete_name_spaces(self, index_name, name_space):
        """
        Deletes a specific namespace from a given index in Pinecone.
        
        Args:
            index_name (str): The name of the index containing the namespace.
            name_space (str): The name of the namespace to delete.
        
        Returns:
            str: A success or error message.
        """
        try:
            # Initialize the index
            pc = Pinecone(api_key=self.api_key)
            index = pc.Index(index_name)  # Get the index object
            # Delete the specified namespace
            response = index.delete(namespace=name_space, delete_all=True)
            if response == {}:  # If deletion is successful
                return f"Namespace '{name_space}' deleted successfully from index '{index_name}'."
            else:
                return f"Unexpected response: {response}"  # If the response is unexpected
        except Exception:  # Handle exceptions
            return f"An error occurred: Failed to delete Namespace"
    
    def insert_data_in_namespace(self, documents, embeddings, index_name, name_space):
        """
        Inserts data into a specific namespace within an index.
        
        Args:
            documents (list): A list of documents to be inserted.
            embeddings (object): The embeddings to be used for the documents.
            index_name (str): The name of the index where the namespace exists.
            name_space (str): The name of the namespace where the data will be inserted.
        
        Returns:
            object: The created Pinecone vector store or an error message.
        """
        try:
            doc_search = PineconeVectorStore.from_documents(
                documents,
                embeddings,
                index_name=index_name,
                namespace=name_space
            )
            print(f"Your Namespace {name_space} is created successfully")  # Success message
            return doc_search
        except Exception as ex:  # Handle exceptions
            return f"Failed to create namespace: {ex}"
    
    def insert_data_in_index(self, documents, embeddings, index_name):
        """
        Inserts data directly into a specified index without using namespaces.
        
        Args:
            documents (list): A list of documents to be inserted.
            embeddings (object): The embeddings to be used for the documents.
            index_name (str): The name of the index where the data will be inserted.
        
        Returns:
            str: A success message or error message.
        """
        try:
            PineconeVectorStore.from_documents(
                documents,
                embedding=embeddings,
                index_name=index_name
            )
            print(f"Your data inserted into {index_name} successfully")  # Success message
        except Exception as ex:  # Handle exceptions
            return f"Failed to insert data into index: {ex}"

    def retrieve_from_index_name(self, index_name, embeddings):
        """
        Retrieves data from a specified index.
        
        Args:
            index_name (str): The name of the index to retrieve data from.
            embeddings (object): The embeddings used for retrieving the data.
        
        Returns:
            object: The retrieved Pinecone vector store or an error message.
        """
        try:
            vectorstore = PineconeVectorStore.from_existing_index(
                embedding=embeddings, index_name=index_name
            )
            return vectorstore
        except Exception as ex:  # Handle exceptions
            return f"Failed to load VectorStore: {ex}"
        
    def retrieve_from_namespace(self, index_name, embeddings, name_space):
        """
        Retrieves data from a specific namespace within an index.
        
        Args:
            index_name (str): The name of the index containing the namespace.
            embeddings (object): The embeddings used for retrieving the data.
            name_space (str): The name of the namespace to retrieve data from.
        
        Returns:
            object: The retrieved Pinecone vector store or an error message.
        """
        try:
            vectorstore = PineconeVectorStore.from_existing_index(
                embedding=embeddings, index_name=index_name, namespace=name_space
            )
            return vectorstore
        except Exception as ex:  # Handle exceptions
            return f"Failed to load VectorStore: {ex}"
