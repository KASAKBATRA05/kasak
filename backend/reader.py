import fitz  # PyMuPDF
import streamlit as st
from typing import Union
import io

class DocumentReader:
    """Handles document reading for PDF and TXT files"""
    
    def __init__(self):
        pass
    
    def extract_from_pdf(self, uploaded_file) -> str:
        """
        Extract text from PDF file using PyMuPDF
        
        Args:
            uploaded_file: Streamlit uploaded file object
            
        Returns:
            str: Extracted text content
        """
        try:
            # Read the uploaded file
            pdf_bytes = uploaded_file.read()
            
            # Open PDF from bytes
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            
            text_content = []
            
            # Extract text from each page
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text = page.get_text()
                
                # Clean up the text
                text = self._clean_text(text)
                
                if text.strip():  # Only add non-empty pages
                    text_content.append(text)
            
            doc.close()
            
            # Join all pages with double newline
            full_text = "\n\n".join(text_content)
            
            if not full_text.strip():
                raise ValueError("No text content found in PDF")
            
            return full_text
            
        except Exception as e:
            raise Exception(f"Error reading PDF: {str(e)}")
    
    def extract_from_txt(self, uploaded_file) -> str:
        """
        Extract text from TXT file
        
        Args:
            uploaded_file: Streamlit uploaded file object
            
        Returns:
            str: Text content
        """
        try:
            # Read text file
            text_content = uploaded_file.read()
            
            # Decode if bytes
            if isinstance(text_content, bytes):
                text_content = text_content.decode('utf-8')
            
            # Clean up the text
            text_content = self._clean_text(text_content)
            
            if not text_content.strip():
                raise ValueError("No text content found in file")
            
            return text_content
            
        except Exception as e:
            raise Exception(f"Error reading TXT file: {str(e)}")
    
    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize text content
        
        Args:
            text: Raw text content
            
        Returns:
            str: Cleaned text
        """
        if not text:
            return ""
        
        # Remove excessive whitespace
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Strip whitespace from each line
            line = line.strip()
            
            # Skip empty lines but preserve paragraph breaks
            if line:
                cleaned_lines.append(line)
            elif cleaned_lines and cleaned_lines[-1] != "":
                cleaned_lines.append("")
        
        # Join lines back
        cleaned_text = '\n'.join(cleaned_lines)
        
        # Remove excessive newlines (more than 2 consecutive)
        import re
        cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text)
        
        return cleaned_text.strip()
    
    def get_document_stats(self, text: str) -> dict:
        """
        Get basic statistics about the document
        
        Args:
            text: Document text
            
        Returns:
            dict: Document statistics
        """
        if not text:
            return {
                'characters': 0,
                'words': 0,
                'sentences': 0,
                'paragraphs': 0
            }
        
        # Count characters
        char_count = len(text)
        
        # Count words
        word_count = len(text.split())
        
        # Count sentences (rough estimate)
        import re
        sentences = re.split(r'[.!?]+', text)
        sentence_count = len([s for s in sentences if s.strip()])
        
        # Count paragraphs
        paragraphs = text.split('\n\n')
        paragraph_count = len([p for p in paragraphs if p.strip()])
        
        return {
            'characters': char_count,
            'words': word_count,
            'sentences': sentence_count,
            'paragraphs': paragraph_count
        }