import PyPDF2
import os
from pathlib import Path
import json
from typing import Dict, List, Any
import warnings
import ctypes
from ctypes import wintypes
import re
import logging
import time
from datetime import datetime
import unicodedata

def safe_str(text):
    """Convert text to ASCII-safe string for logging on Windows."""
    if isinstance(text, str):
        # Replace Unicode characters that can't be encoded in cp1252
        try:
            text.encode('cp1252')
            return text
        except UnicodeEncodeError:
            # Replace problematic characters with ASCII equivalents or descriptive text
            # Handle specific Greek characters commonly found in physics papers
            replacements = {
                'γ': 'gamma',
                'π': 'pi',
                'α': 'alpha',
                'β': 'beta',
                'δ': 'delta',
                'ε': 'epsilon',
                'θ': 'theta',
                'λ': 'lambda',
                'μ': 'mu',
                'ν': 'nu',
                'ρ': 'rho',
                'σ': 'sigma',
                'τ': 'tau',
                'φ': 'phi',
                'χ': 'chi',
                'ψ': 'psi',
                'ω': 'omega',
                'Γ': 'Gamma',
                'Π': 'Pi',
                'Α': 'Alpha',
                'Β': 'Beta',
                'Δ': 'Delta',
                'Ε': 'Epsilon',
                'Θ': 'Theta',
                'Λ': 'Lambda',
                'Μ': 'Mu',
                'Ν': 'Nu',
                'Ρ': 'Rho',
                'Σ': 'Sigma',
                'Τ': 'Tau',
                'Φ': 'Phi',
                'Χ': 'Chi',
                'Ψ': 'Psi',
                'Ω': 'Omega'
            }
            
            # Apply character replacements
            for unicode_char, replacement in replacements.items():
                text = text.replace(unicode_char, replacement)
            
            # Try encoding again after replacements
            try:
                text.encode('cp1252')
                return text
            except UnicodeEncodeError:
                # If still failing, normalize and convert to ASCII
                return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')
    return str(text)

class UnicodeStreamHandler(logging.StreamHandler):
    """Custom stream handler that safely handles Unicode characters."""
    def emit(self, record):
        try:
            # Create a safe version of the log message
            if hasattr(record, 'msg') and isinstance(record.msg, str):
                record.msg = safe_str(record.msg)
            if hasattr(record, 'args') and record.args:
                record.args = tuple(safe_str(arg) for arg in record.args)
            super().emit(record)
        except (UnicodeEncodeError, UnicodeDecodeError) as e:
            # Fail fast - don't provide fallback generic messages
            raise RuntimeError(f"Unicode encoding error in log message: {e}") from e

# Configure logging with Unicode handling
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/pdf_processor_{time.strftime("%Y%m%d_%H%M%S")}.log', encoding='utf-8'),
        UnicodeStreamHandler()
    ]
)
logger = logging.getLogger(__name__)

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
        logger.warning(f"Could not enable long path support: {str(e)}")
        return False

# Filter out PyPDF2 encoding warnings
warnings.filterwarnings('ignore', category=UserWarning, module='PyPDF2')

class PDFProcessor:
    def __init__(self, papers_folder: str = None):
        """
        Initialize PDF processor for downloaded ArXiv papers.
        
        Args:
            papers_folder: Path to downloaded papers folder
        """
        self.papers_folder = Path(papers_folder) if papers_folder else None
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
            logger.error(f"Error reading PDF {pdf_path.name}: {str(e)}")
            logger.error(f"File exists: {pdf_path.exists()}")
            logger.error(f"File size: {pdf_path.stat().st_size if pdf_path.exists() else 'N/A'}")
            raise
    
    def remove_references_section(self, text: str) -> str:
        """
        Remove references section from paper text without using LLM.
        Based on the approach from data_extractor.py
        """
        try:
            # Multiple possible reference headers to look for
            reference_headers = [
                "\nReferences",
                "\nREFERENCES", 
                "\nReferences\n",
                "\nREFERENCES\n",
                "REFERENCES\n",
                "references\n",
                "\n19References",  # Sometimes there are formatting issues
                "\nBibliography",
                "\nBIBLIOGRAPHY",
                "\nCitations",
                "\nCITATIONS",
                "References\n",
                

            ]
            
            original_length = len(text)
            
            for header in reference_headers:
                if header in text:
                    text = text.split(header)[0]
                    logger.info(f"Removed references section starting with: '{header}'")
                    break
            
            final_length = len(text)
            if final_length < original_length:
                logger.info(f"Text reduced from {original_length} to {final_length} characters")
            else:
                logger.info("No references section found to remove")
            
            return text
            
        except Exception as e:
            logger.error(f"Error removing references section: {str(e)}")
            return text  # Return original text if removal fails
    
    def process_downloaded_papers(self, papers_folder: str) -> Dict[str, Any]:
        """
        Process all downloaded ArXiv papers in the specified folder.
        
        Args:
            papers_folder: Path to the downloaded papers folder
            
        Returns:
            Dict containing processed paper data
        """
        try:
            papers_dir = Path(papers_folder)
            
            if not papers_dir.exists():
                logger.error(f"Papers folder '{papers_folder}' does not exist!")
                return {}
            
            logger.info(f"Processing papers in: {papers_folder}")
            
            # Load query-paper mapping if available
            mapping_file = papers_dir / "query_paper_mapping.json"
            query_mapping = {}
            if mapping_file.exists():
                with open(mapping_file, 'r') as f:
                    query_mapping = json.load(f)
                logger.info("Loaded query-paper mapping")
            
            processed_papers = {}
            total_processed = 0
            
            # Process papers in each hotspot subdirectory
            for hotspot_dir in papers_dir.iterdir():
                if hotspot_dir.is_dir() and hotspot_dir.name != "__pycache__":
                    logger.info(f"Processing hotspot directory: {hotspot_dir.name}")
                    
                    # Load papers info for this hotspot if available
                    papers_info_file = hotspot_dir / "papers_info.json"
                    papers_info = []
                    if papers_info_file.exists():
                        with open(papers_info_file, 'r') as f:
                            papers_info = json.load(f)
                    
                    # Process PDF files in this hotspot directory
                    pdf_files = list(hotspot_dir.glob("*.pdf"))
                    logger.info(f"Found {len(pdf_files)} PDF files in {hotspot_dir.name}")
                    
                    for pdf_file in pdf_files:
                        try:
                            logger.info(f"Processing: {pdf_file.name}")
                            
                            # Extract text from PDF
                            raw_text = self.extract_text_from_pdf(pdf_file)
                            
                            if not raw_text.strip():
                                logger.warning(f"No text extracted from {pdf_file.name}")
                                continue
                            
                            # Remove references section
                            cleaned_text = self.remove_references_section(raw_text)
                            
                            # Find corresponding paper info
                            paper_info = None
                            for info in papers_info:
                                if info.get('filename') == pdf_file.name:
                                    paper_info = info
                                    break
                            
                            # Create paper data structure
                            paper_id = pdf_file.stem
                            paper_data = {
                                "full_text": cleaned_text,
                                "metadata": {
                                    "filename": pdf_file.name,
                                    "hotspot_name": hotspot_dir.name
                                }
                            }
                            
                            # Add essential metadata if available
                            if paper_info:
                                paper_data["metadata"].update({
                                    "title": paper_info.get("title", ""),
                                    "published_date": paper_info.get("published_date", ""),
                                    "pdf_link": paper_info.get("pdf_link", ""),
                                    "query": paper_info.get("query", "")
                                })
                            
                            processed_papers[paper_id] = paper_data
                            total_processed += 1
                            logger.info(f"Successfully processed: {pdf_file.name}")
                            
                        except Exception as e:
                            logger.error(f"Failed to process {pdf_file.name}: {str(e)}")
                            continue
            
            logger.info(f"Successfully processed {total_processed} papers total")
            
            # Combine with query mapping
            final_result = {
                "processed_papers": processed_papers,
                "query_mapping": query_mapping,
                "processing_summary": {
                    "total_processed": total_processed,
                    "processing_timestamp": str(datetime.now()) if 'datetime' in globals() else str(time.time())
                }
            }
            
            return final_result
            
        except Exception as e:
            logger.error(f"Error processing downloaded papers: {str(e)}")
            raise
    
    def save_processed_papers(self, processed_data: Dict[str, Any], output_file: str):
        """
        Save processed papers data to JSON file.
        
        Args:
            processed_data: Processed papers data
            output_file: Output file path
        """
        try:
            # Ensure output directory exists
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(processed_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Processed papers saved to: {output_file}")
            
        except Exception as e:
            logger.error(f"Error saving processed papers: {str(e)}")
            raise
    
    def process_papers_for_project(self, output_folder: str) -> str:
        """
        Process all downloaded papers for a specific project.
        
        Args:
            output_folder: Project output folder containing downloaded_papers subdirectory
            
        Returns:
            Path to the saved processed papers file
        """
        try:
            papers_folder = Path(output_folder) / "downloaded_papers"
            
            if not papers_folder.exists():
                logger.error(f"No downloaded_papers folder found in {output_folder}")
                return None
            
            # Process papers
            processed_data = self.process_downloaded_papers(str(papers_folder))
            
            # Save processed data
            output_file = Path(output_folder) / "processed_papers.json"
            self.save_processed_papers(processed_data, str(output_file))
            
            return str(output_file)
            
        except Exception as e:
            logger.error(f"Error processing papers for project: {str(e)}")
            raise

# Legacy function for backward compatibility
def process_all_papers_legacy(literature_folder: str = "Literature_papers") -> Dict[str, Any]:
    """Legacy function for processing papers from Literature_papers folder."""
    processor = PDFProcessor()
    processor.papers_folder = Path(literature_folder)
    processed_data = {}
    
    if not processor.papers_folder.exists():
        logger.error(f"Literature folder '{literature_folder}' does not exist!")
        return processed_data
        
    # Get all PDF files
    pdf_files = list(processor.papers_folder.glob("*.pdf"))
    
    if not pdf_files:
        logger.warning(f"No PDF files found in '{literature_folder}'")
        return processed_data
        
    logger.info(f"Found {len(pdf_files)} PDF files to process")
    
    failed_files = []
    for pdf_file in pdf_files:
        try:
            paper_id = pdf_file.stem
            text = processor.extract_text_from_pdf(pdf_file)
            if not text.strip():
                logger.warning(f"No text extracted from {pdf_file.name}")
                failed_files.append((pdf_file.name, "No text extracted"))
                continue
                
            processed_data[paper_id] = {
                "full_text": text,
                "metadata": {
                    "filename": pdf_file.name
                }
            }
            logger.info(f"Processed: {pdf_file.name}")
            
        except Exception as e:
            logger.error(f"Failed to process {pdf_file.name}: {str(e)}")
            failed_files.append((pdf_file.name, str(e)))
            
    if failed_files:
        logger.warning(f"Failed files: {failed_files}")
    
    return processed_data

if __name__ == "__main__":
    # Test the PDF processor with downloaded papers
    processor = PDFProcessor()
    
    # Example: process papers in a project folder
    test_output_folder = "output/automotive_sample_input"
    
    try:
        result_file = processor.process_papers_for_project(test_output_folder)
        if result_file:
            logger.info(f"Processing complete! Results saved to: {result_file}")
        else:
            logger.info("No papers found to process")
    except Exception as e:
        logger.error(f"Processing failed: {str(e)}")