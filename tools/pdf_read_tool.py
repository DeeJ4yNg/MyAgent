#!/usr/bin/env python3
"""
pdf_read_tool.py - Tool for reading PDF file content

This tool can read the content of PDF files and extract text information.
"""

import os
from typing import Optional
from langchain.tools import BaseTool
from pydantic import BaseModel, Field

try:
    import PyPDF2
except ImportError:
    print("Error: PyPDF2 package is not installed. Please run: pip install PyPDF2")
    exit(1)

try:
    # Try to import pdfplumber as an alternative
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False


class PdfReadToolInput(BaseModel):
    file_path: str = Field(..., description="Absolute path to the PDF file")
    max_length: Optional[int] = Field(None, description="Optional, limit the length of returned text")


class PdfReadTool(BaseTool):
    name: str = "pdf_read"
    description: str = (
        "Read the content of PDF files and extract text information. "
        "Supports text extraction using both PyPDF2 and pdfplumber, with preference for pdfplumber. "
        "Can extract text content from all pages in the PDF and supports content length limitation."
    )
    args_schema: type = PdfReadToolInput

    def _run(self, file_path: str, max_length: Optional[int] = None) -> str:
        """
        Read the content of a PDF file
        
        Args:
            file_path: Path to the PDF file
            max_length: Optional, limit the length of returned text
            
        Returns:
            Text content of the PDF file
        """
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                return f"Error: File '{file_path}' does not exist"
            
            # Check file extension
            if not file_path.lower().endswith('.pdf'):
                return f"Error: File '{file_path}' is not a valid .pdf file"
            
            # First try using pdfplumber (if available) as it usually provides better text extraction
            if HAS_PDFPLUMBER:
                return self._read_pdf_with_pdfplumber(file_path, max_length)
            else:
                # Fallback to PyPDF2
                return self._read_pdf_with_pypdf2(file_path, max_length)
        
        except Exception as e:
            return f"Error reading PDF file: {str(e)}"
    
    def _read_pdf_with_pypdf2(self, file_path: str, max_length: Optional[int] = None) -> str:
        """
        Read PDF file content using PyPDF2
        
        Args:
            file_path: Path to the PDF file
            max_length: Optional, limit the length of returned text
            
        Returns:
            Text content of the PDF file
        """
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                num_pages = len(pdf_reader.pages)
                
                # Extract text from each page
                full_text = []
                for page_num in range(num_pages):
                    page = pdf_reader.pages[page_num]
                    text = page.extract_text()
                    if text.strip():
                        full_text.append(f"--- Page {page_num + 1} ---\n{text}")
                
                content = "\n\n".join(full_text)
                
                # If max length is specified, truncate content
                if max_length and len(content) > max_length:
                    content = content[:max_length] + "\n\n[Content truncated...]"
                
                return content
        
        except Exception as e:
            return f"Error reading PDF with PyPDF2: {str(e)}"
    
    def _read_pdf_with_pdfplumber(self, file_path: str, max_length: Optional[int] = None) -> str:
        """
        Read PDF file content using pdfplumber (alternative method)
        
        Args:
            file_path: Path to the PDF file
            max_length: Optional, limit the length of returned text
            
        Returns:
            Text content of the PDF file
        """
        try:
            full_text = []
            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    text = page.extract_text()
                    if text and text.strip():
                        full_text.append(f"--- Page {page_num + 1} ---\n{text}")
            
            content = "\n\n".join(full_text)
            
            # If max length is specified, truncate content
            if max_length and len(content) > max_length:
                content = content[:max_length] + "\n\n[Content truncated...]"
            
            return content
        
        except Exception as e:
            return f"Error reading PDF with pdfplumber: {str(e)}"


# For backward compatibility, keep the original function
def read_pdf(file_path: str, max_length: Optional[int] = None) -> str:
    """Backward compatible function interface"""
    tool = PdfReadTool()
    return tool._run(file_path, max_length)


if __name__ == "__main__":
    import argparse
    
    # Command line entry point
    parser = argparse.ArgumentParser(description="Read PDF file content")
    parser.add_argument("file_path", help="Path to the PDF file")
    parser.add_argument("--max-length", type=int, help="Limit the length of returned text")
    
    args = parser.parse_args()
    
    content = read_pdf(args.file_path, args.max_length)
    print(content)