from langchain_community.document_loaders import (
    TextLoader, CSVLoader, BSHTMLLoader, JSONLoader, 
    PyPDFLoader, PyMuPDFLoader, UnstructuredPDFLoader, 
    OnlinePDFLoader, PyPDFium2Loader, PDFMinerLoader, 
    PyPDFDirectoryLoader, PDFPlumberLoader, CSVLoader as LangCSVLoader, WebBaseLoader
)
from pathlib import Path
from pprint import pprint

class LangChainLoader:
    """
    A class to handle loading and processing documents from various sources, 
    such as text files, CSV files, PDFs, JSON, HTML, and web URLs.
    """

    def __init__(self):
        """
        Initializes the LangChainLoader class with default attributes.
        Attributes:
            file_path (str): Path to the file to be loaded (default: None).
            url (str): URL of the web resource to be fetched (default: None).
        """
        self.file_path = None  # Attribute to store file path
        self.url = None        # Attribute to store URL for web-based loading

    def load_as_textfile(self, file_path):
        """
        Load a text file and return its content as a list of Document objects.
        
        Args:
            file_path (str): Path to the text file to be loaded.
        
        Returns:
            list: A list of Document objects containing the file content.
        
        Raises:
            FileNotFoundError: If the specified file does not exist.
            Exception: For any other issues during file loading.
        """
        try:
            loader = TextLoader(file_path)  # Instantiate a TextLoader for the given file
            documents = loader.load()      # Load the file and extract documents
            return documents
        except FileNotFoundError as fnf_error:
            print(f"Error: The file '{self.file_path}' was not found.")
            raise fnf_error  # Re-raise the exception for higher-level handling
        except Exception as e:
            print(f"An error occurred while loading the text file: {e}")
            raise

    def load_as_csv(self, file_path):
        """
        Load a CSV file and return its content as a list of Document objects.
        
        Args:
            file_path (str): Path to the CSV file to be loaded.
        
        Returns:
            list: A list of Document objects containing the CSV content.
        
        Raises:
            FileNotFoundError: If the file does not exist.
            Exception: For other issues during loading.
        """
        try:
            loader = CSVLoader(
                file_path=file_path,
                csv_args={'delimiter': ',', 'quotechar': '"'}  # Specify CSV parsing arguments
            )
            documents = loader.load()
            return documents
        except FileNotFoundError as fnf_error:
            print(f"Error: The file '{self.file_path}' was not found.")
            raise fnf_error
        except Exception as e:
            print(f"An error occurred while loading the CSV file: {e}")
            raise

    def load_as_csv_with_source_column(self, file_path, source_column: str) -> list:
        """
        Load a CSV file with a specific source column and return its content as a list of Document objects.
        
        Args:
            file_path (str): Path to the CSV file.
            source_column (str): Column in the CSV file to be used as the source.
        
        Returns:
            list: A list of Document objects filtered by the source column.
        
        Raises:
            FileNotFoundError: If the file is not found.
            Exception: For other issues during loading.
        """
        try:
            loader = LangCSVLoader(file_path=file_path, source_column=source_column)
            documents = loader.load()
            return documents
        except FileNotFoundError as fnf_error:
            print(f"Error: {fnf_error}")
            raise
        except Exception as e:
            print(f"An error occurred while loading the CSV file with source column: {e}")
            raise

    def load_as_html(self, file_path):
        """
        Load an HTML file and return its content as a list of Document objects.
        
        Args:
            file_path (str): Path to the HTML file.
        
        Returns:
            list: A list of Document objects containing the HTML content.
        
        Raises:
            FileNotFoundError: If the file is not found.
            Exception: For other issues during loading.
        """
        try:
            loader = BSHTMLLoader(file_path)  # Use BSHTMLLoader for HTML parsing
            documents = loader.load()
            return documents
        except FileNotFoundError as fnf_error:
            print(f"Error: The file '{self.file_path}' was not found.")
            raise fnf_error
        except Exception as e:
            print(f"An error occurred while loading the HTML file: {e}")
            raise

    def load_as_json(self, file_path):
        """
        Load a JSON file and return its content as a list of Document objects.
        
        Args:
            file_path (str): Path to the JSON file.
        
        Returns:
            list: A list of Document objects parsed from the JSON file.
        
        Raises:
            FileNotFoundError: If the file does not exist.
            Exception: For any other issues during loading.
        """
        try:
            loader = JSONLoader(
                file_path=file_path,
                jq_schema='.messages[].content',  # Schema to filter content from JSON
                text_content=False
            )
            documents = loader.load()
            return documents
        except FileNotFoundError as fnf_error:
            print(f"Error: The file '{self.file_path}' was not found.")
            raise fnf_error
        except Exception as e:
            print(f"An error occurred while loading the JSON file: {e}")
            raise

    def load_as_pdf(self, file_path):
        """
        Load and split a PDF file into smaller chunks using PyPDFLoader.
        
        Args:
            file_path (str): Path to the PDF file.
        
        Returns:
            list: A list of Document objects containing chunks of the PDF content.
        
        Raises:
            FileNotFoundError: If the file does not exist.
            Exception: For other issues during loading.
        """
        try:
            loader = PyPDFLoader(file_path)
            documents = loader.load_and_split()
            return documents
        except FileNotFoundError as fnf_error:
            print(f"Error: The file '{self.file_path}' was not found.")
            raise fnf_error
        except Exception as e:
            print(f"An error occurred while loading the PDF file: {e}")
            raise

    def load_with_pymupdf(self, file_path):
        """
        Load a PDF file using PyMuPDFLoader.
        
        Parameters:
        - file_path (str): Path to the PDF file to load.
        
        Returns:
        - list: A list of Document objects containing the PDF content.
        
        Raises:
        - Exception: If an error occurs during the loading process.
        """
        try:
            # Initialize PyMuPDFLoader and load the PDF file
            loader = PyMuPDFLoader(file_path)
            documents = loader.load()
            return documents
        except Exception as e:
            print(f"An error occurred while loading the PDF with PyMuPDFLoader: {e}")
            raise

    def load_with_unstructuredpdf(self, file_path):
        """
        Load a PDF file using UnstructuredPDFLoader.
        
        Parameters:
        - file_path (str): Path to the PDF file to load.
        
        Returns:
        - list: A list of Document objects containing the PDF content.
        
        Raises:
        - Exception: If an error occurs during the loading process.
        """
        try:
            # Initialize UnstructuredPDFLoader and load the PDF file
            loader = UnstructuredPDFLoader(file_path)
            documents = loader.load()
            return documents
        except Exception as e:
            print(f"An error occurred while loading the PDF with UnstructuredPDFLoader: {e}")
            raise

    def load_with_pypdfium2(self, file_path):
        """
        Load a PDF file using PyPDFium2Loader.
        
        Parameters:
        - file_path (str): Path to the PDF file to load.
        
        Returns:
        - list: A list of Document objects containing the PDF content.
        
        Raises:
        - Exception: If an error occurs during the loading process.
        """
        try:
            # Initialize PyPDFium2Loader and load the PDF file
            loader = PyPDFium2Loader(file_path)
            documents = loader.load()
            return documents
        except Exception as e:
            print(f"An error occurred while loading the PDF with PyPDFium2Loader: {e}")
            raise

    def load_with_pdfminer(self, file_path):
        """
        Load a PDF file using PDFMinerLoader.
        
        Parameters:
        - file_path (str): Path to the PDF file to load.
        
        Returns:
        - list: A list of Document objects containing the PDF content.
        
        Raises:
        - Exception: If an error occurs during the loading process.
        """
        try:
            # Initialize PDFMinerLoader and load the PDF file
            loader = PDFMinerLoader(file_path)
            documents = loader.load()
            return documents
        except Exception as e:
            print(f"An error occurred while loading the PDF with PDFMinerLoader: {e}")
            raise

    def load_directory_pdfs(self, directory_path):
        """
        Load all PDF files in a directory using PyPDFDirectoryLoader.
        
        Parameters:
        - directory_path (str): Path to the directory containing PDF files.
        
        Returns:
        - list: A list of Document objects containing the PDFs' content.
        
        Raises:
        - Exception: If an error occurs during the loading process.
        """
        try:
            # Initialize PyPDFDirectoryLoader and load all PDFs in the directory
            loader = PyPDFDirectoryLoader(directory_path)
            documents = loader.load()
            return documents
        except Exception as e:
            print(f"An error occurred while loading PDFs from directory: {e}")
            raise

    def load_with_pdfplumber(self, file_path):
        """
        Load a PDF file using PDFPlumberLoader.
        
        Parameters:
        - file_path (str): Path to the PDF file to load.
        
        Returns:
        - list: A list of Document objects containing the PDF content.
        
        Raises:
        - Exception: If an error occurs during the loading process.
        """
        try:
            # Initialize PDFPlumberLoader and load the PDF file
            loader = PDFPlumberLoader(file_path)
            documents = loader.load()
            return documents
        except Exception as e:
            print(f"An error occurred while loading the PDF with PDFPlumberLoader: {e}")
            raise

    def load_online_pdf(self, url):
        """
        Load an online PDF from the given URL using OnlinePDFLoader.
        
        Parameters:
        - url (str): The URL of the online PDF to load.
        
        Returns:
        - list: A list of Document objects containing the PDF content.
        
        Raises:
        - Exception: If an error occurs during the loading process.
        """
        try:
            # Initialize OnlinePDFLoader and load the PDF content from the URL
            loader = OnlinePDFLoader(url)
            documents = loader.load()
            return documents
        except Exception as e:
            print(f"An error occurred while loading the online PDF: {e}")
            raise

    def fetch_data_from_url(self, url):
        """
        Fetch data from a web URL using WebBaseLoader.
        
        Parameters:
        - url (str): The web URL to fetch data from.
        
        Returns:
        - list: A list of Document objects containing the web data.
        
        Raises:
        - Exception: If an error occurs during the fetching process.
        """
        try:
            # Initialize WebBaseLoader and fetch data from the URL
            loader = WebBaseLoader(web_paths=[url])
            documents = loader.load()
            return documents
        except Exception as e:
            print(f"An error occurred while fetching data from the URL: {e}")
            raise

