import PyPDF2
import os
from pathlib import Path
import json
from typing import Dict, List, Any

class PDFProcessor:
    def __init__(self, literature_folder: str = "Literature_papers"):
        self.literature_folder = Path(literature_folder)
        self.processed_data = {}
        self.processed_count = 0
        
    def extract_text_from_pdf(self, pdf_path: Path) -> str:
        """Extract text from a PDF file."""
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
        return text
    
    def process_all_papers(self) -> Dict[str, Any]:
        """Process all PDFs in the literature folder."""
        self.processed_count = 0
        for pdf_file in self.literature_folder.glob("*.pdf"):
            try:
                paper_id = pdf_file.stem
                text = self.extract_text_from_pdf(pdf_file)
                self.processed_data[paper_id] = {
                    "full_text": text,
                    "metadata": {
                        "filename": pdf_file.name
                    }
                }
                # Save after each file is processed
                self.save_processed_data()
                self.processed_count += 1
                print(f"Processed {self.processed_count} ")
            except Exception as e:
                print(f"Failed to process {pdf_file}: {str(e)}")
                
        print(f"Successfully processed {self.processed_count} PDF files")
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