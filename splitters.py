from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.schema import Document
from bs4 import BeautifulSoup
from typing import List, Any

class LangchainSplitters:
    def __init__(self, docs: List[Document]):
        """
        Initialize LangchainSplitters with a list of Document objects.
        
        Parameters:
            docs (List[Document]): A list of Document objects to be processed and split.
        """
        self.docs = docs

    def html_splitter(self, chunk_size: int = 100, chunk_overlap: int = 20, headers_to_split_on: List[str] = ["h1", "h2", "h3", "p"]) -> List[Document]:
        """
        Splits HTML content in the documents based on specified HTML headers and paragraphs.
        
        Parameters:
            chunk_size (int): Maximum size of each chunk.
            chunk_overlap (int): Overlap between consecutive chunks.
            headers_to_split_on (List[str]): HTML tags to use for splitting (e.g., "h1", "h2").
            
        Returns:
            List[Document]: A list of chunked Document objects with added metadata.
        """
        try:
            section_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                is_separator_regex=False
            )

            all_chunked_docs = []
            for doc_index, doc in enumerate(self.docs):
                # Parse HTML content
                soup = BeautifulSoup(doc.page_content, "html.parser")
                sections = [(header_tag, element.get_text()) for header_tag in headers_to_split_on for element in soup.find_all(header_tag)]
                
                chunked_docs = []
                for idx, (header, text) in enumerate(sections):
                    chunks = section_splitter.split_text(text)
                    for chunk_idx, chunk in enumerate(chunks):
                        chunked_docs.append(
                            Document(
                                page_content=chunk,
                                metadata={
                                    **doc.metadata,
                                    "chunkno": f"{idx+1}-{chunk_idx+1}",
                                    "header_type": header
                                }
                            )
                        )
                all_chunked_docs.extend(chunked_docs)

            return all_chunked_docs

        except Exception as e:
            print(f"Error in html_splitter: {str(e)}")
            return []

    def recursive_text_splitter(self, chunk_size: int = 900, chunk_overlap: int = 100) -> List[Document]:
        """
        Splits text in each document recursively based on character length.
        
        Parameters:
            chunk_size (int): Maximum size of each chunk.
            chunk_overlap (int): Overlap between consecutive chunks.
            
        Returns:
            List[Document]: A list of chunked Document objects with added metadata.
        """
        try:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                is_separator_regex=False
            )

            all_chunked_docs = []
            for doc_index, doc in enumerate(self.docs):
                splits = text_splitter.split_text(doc.page_content)
                chunked_docs = [
                    Document(
                        page_content=chunk,
                        metadata={**doc.metadata, "chunkno": idx + 1}
                    )
                    for idx, chunk in enumerate(splits)
                ]
                all_chunked_docs.extend(chunked_docs)

            return all_chunked_docs

        except Exception as e:
            print(f"Error in recursive_text_splitter: {str(e)}")
            return []

    def character_text_splitter(self, chunk_size: int = 1000, chunk_overlap: int = 200, separator: str = "\n") -> List[Document]:
        """
        Splits text in each document based on a character separator.
        
        Parameters:
            chunk_size (int): Maximum size of each chunk.
            chunk_overlap (int): Overlap between consecutive chunks.
            separator (str): Character or string to separate chunks.
            
        Returns:
            List[Document]: A list of chunked Document objects with added metadata.
        """
        try:
            text_splitter = CharacterTextSplitter(
                separator=separator,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                is_separator_regex=False
            )

            all_chunked_docs = []
            for doc_index, doc in enumerate(self.docs):
                splits = text_splitter.split_text(doc.page_content)
                chunked_docs = [
                    Document(
                        page_content=chunk,
                        metadata={**doc.metadata, "chunkno": idx + 1}
                    )
                    for idx, chunk in enumerate(splits)
                ]
                all_chunked_docs.extend(chunked_docs)

            return all_chunked_docs

        except Exception as e:
            print(f"Error in character_text_splitter: {str(e)}")
            return []

