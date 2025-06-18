import PyPDF2
import os
from pathlib import Path
import json
from typing import Dict, List, Any
import warnings
import ctypes
from ctypes import wintypes

# Enable Windows long path support
def enable_long_paths():
    try:
        # Constants from Windows API
        MAX_PATH = 260
        PATHCCH_MAX_CCH = 0x8000
        
        # Get the current process
        kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
        
        # Set the process to use long paths
        kernel32.SetProcessDEPPolicy(0)
        
        # Enable long path support
        kernel32.SetFileInformationByHandle.restype = wintypes.BOOL
        kernel32.SetFileInformationByHandle.argtypes = [
            wintypes.HANDLE, wintypes.DWORD, wintypes.LPVOID, wintypes.DWORD
        ]
        
        return True
    except Exception as e:
        print(f"Warning: Could not enable long path support: {str(e)}")
        return False

# Filter out PyPDF2 encoding warnings
warnings.filterwarnings('ignore', category=UserWarning, module='PyPDF2')

class PDFProcessor:
    def __init__(self, literature_folder: str = "Literature_papers"):
        self.literature_folder = Path(literature_folder)
        self.processed_data = {}
        self.processed_count = 0
        # Enable long path support
        enable_long_paths()
        
    def extract_text_from_pdf(self, pdf_path: Path) -> str:
        """Extract text from a PDF file."""
        try:
            # Use absolute path with long path prefix
            abs_path = str(pdf_path.absolute())
            if abs_path.startswith('\\\\?\\'):
                abs_path = '\\\\?\\' + abs_path
                
            with open(abs_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text()
            return text
        except Exception as e:
            print(f"Error reading PDF {pdf_path.name}: {str(e)}")
            print(f"File exists: {pdf_path.exists()}")
            print(f"File size: {pdf_path.stat().st_size if pdf_path.exists() else 'N/A'}")
            print(f"File permissions: {oct(pdf_path.stat().st_mode) if pdf_path.exists() else 'N/A'}")
            raise
    
    def process_all_papers(self) -> Dict[str, Any]:
        """Process all PDFs in the literature folder."""
        self.processed_count = 0
        if not self.literature_folder.exists():
            print(f"Error: Literature folder '{self.literature_folder}' does not exist!")
            return self.processed_data
            
        # Get all PDF files using a different method
        pdf_files = []
        for file in self.literature_folder.iterdir():
            if file.is_file() and file.suffix.lower() == '.pdf':
                pdf_files.append(file)
                
        if not pdf_files:
            print(f"Warning: No PDF files found in '{self.literature_folder}'")
            return self.processed_data
            
        print(f"Found {len(pdf_files)} PDF files to process")
        print("\nAvailable files:")
        for f in pdf_files:
            print(f"  - {f.name}")
        print("\nStarting processing...")
        
        failed_files = []
        for pdf_file in pdf_files:
            try:
                # Try to open the file first to check if it's accessible
                try:
                    abs_path = str(pdf_file.absolute())
                    if abs_path.startswith('\\\\?\\'):
                        abs_path = '\\\\?\\' + abs_path
                    with open(abs_path, 'rb') as test_file:
                        pass
                except Exception as e:
                    print(f"Warning: Cannot access file {pdf_file.name}: {str(e)}")
                    failed_files.append((pdf_file.name, f"Cannot access file: {str(e)}"))
                    continue
                    
                paper_id = pdf_file.stem
                text = self.extract_text_from_pdf(pdf_file)
                if not text.strip():
                    print(f"Warning: No text extracted from {pdf_file.name}")
                    failed_files.append((pdf_file.name, "No text extracted"))
                    continue
                    
                self.processed_data[paper_id] = {
                    "full_text": text,
                    "metadata": {
                        "filename": pdf_file.name
                    }
                }
                # Save after each file is processed
                self.save_processed_data()
                self.processed_count += 1
                print(f"Processed {self.processed_count}/{len(pdf_files)}: {pdf_file.name}")
            except Exception as e:
                print(f"Failed to process {pdf_file.name}: {str(e)}")
                failed_files.append((pdf_file.name, str(e)))
                
        print(f"\nProcessing Summary:")
        print(f"Successfully processed: {self.processed_count} out of {len(pdf_files)} PDF files")
        if failed_files:
            print("\nFailed files:")
            for filename, error in failed_files:
                print(f"  - {filename}: {error}")
        return self.processed_data
    
    def save_processed_data(self, output_file: str = "output/processed_papers.json"):
        """Save processed data to JSON file."""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.processed_data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    # Create output directory if it doesn't exist
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    # Initialize and run PDF processor
    processor = PDFProcessor()
    processor.process_all_papers()
    print("PDF processing complete! Check the 'output' directory for results.")