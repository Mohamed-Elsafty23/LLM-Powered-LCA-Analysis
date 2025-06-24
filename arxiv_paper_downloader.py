import urllib.request as libreq
import urllib.parse as urlparse
import xml.etree.ElementTree as ET
import os
import re
import json
from pathlib import Path
from typing import List, Dict, Any
import logging
import time
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
        logging.FileHandler(f'logs/arxiv_downloader_{time.strftime("%Y%m%d_%H%M%S")}.log', encoding='utf-8'),
        UnicodeStreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ArxivPaperDownloader:
    def __init__(self, max_results_per_query: int = 10):
        """
        Initialize the ArXiv paper downloader.
        
        Args:
            max_results_per_query: Maximum number of papers to download per query
        """
        self.max_results_per_query = max_results_per_query
        
    def search_and_download_papers(self, search_queries: Dict[str, str], output_folder: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Search ArXiv and download papers for each query.
        
        Args:
            search_queries: Dictionary mapping hotspot names to search queries
            output_folder: Base output folder for the project
            
        Returns:
            Dict mapping hotspot names to lists of downloaded paper info
        """
        try:
            # Create papers directory in output folder
            papers_dir = Path(output_folder) / "downloaded_papers"
            papers_dir.mkdir(parents=True, exist_ok=True)
            
            downloaded_papers = {}
            all_papers_info = []
            
            for hotspot_name, query in search_queries.items():
                logger.info(f"Searching papers for hotspot: {hotspot_name}")
                logger.info(f"Query: {query}")
                
                # Create subdirectory for this hotspot
                hotspot_dir = papers_dir / hotspot_name.replace(" ", "_").replace("/", "_")
                hotspot_dir.mkdir(exist_ok=True)
                
                # Search and download papers for this query
                papers_info = self._search_and_download_single_query(query, hotspot_dir, hotspot_name)
                downloaded_papers[hotspot_name] = papers_info
                all_papers_info.extend(papers_info)
                
                # Add delay between queries to avoid rate limiting
                time.sleep(2)
            
            # Save query-paper mapping
            query_mapping = {
                "search_queries": search_queries,
                "downloaded_papers": downloaded_papers,
                "total_papers": len(all_papers_info),
                "download_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            with open(papers_dir / "query_paper_mapping.json", 'w') as f:
                json.dump(query_mapping, f, indent=2)
            
            logger.info(f"Downloaded {len(all_papers_info)} papers total for {len(search_queries)} hotspots")
            return downloaded_papers
            
        except Exception as e:
            logger.error(f"Error in search and download: {str(e)}")
            raise
    
    def _search_and_download_single_query(self, query: str, output_dir: Path, hotspot_name: str) -> List[Dict[str, Any]]:
        """
        Search ArXiv and download papers for a single query.
        
        Args:
            query: Search query string
            output_dir: Directory to save papers
            hotspot_name: Name of the hotspot for this query
            
        Returns:
            List of paper information dictionaries
        """
        try:
            # Convert complex query to proper ArXiv API format
            arxiv_query = self._simplify_query_for_arxiv(query)
            
            # Manually encode only spaces and special characters, preserve +, AND, OR operators
            # ArXiv API needs + and AND operators to remain unencoded
            encoded_query = arxiv_query.replace(' ', '%20')
            url = f"http://export.arxiv.org/api/query?search_query={encoded_query}&start=0&max_results={self.max_results_per_query}"
            
            logger.info(f"ArXiv API URL: {url}")
            
            # Download the arXiv query results
            with libreq.urlopen(url) as response:
                xml_data = response.read()
            
            # Parse the XML to extract PDF URLs and titles
            root = ET.fromstring(xml_data)
            
            # Define namespace for arXiv
            ns = {
                'atom': 'http://www.w3.org/2005/Atom',
                'arxiv': 'http://arxiv.org/schemas/atom'
            }
            
            papers_info = []
            failed_downloads = []
            
            # Extract and download PDFs
            for i, entry in enumerate(root.findall('atom:entry', ns)):
                try:
                    # Get the title
                    title = entry.find('atom:title', ns).text.strip()
                    
                    # Get the abstract
                    summary = entry.find('atom:summary', ns)
                    abstract = summary.text.strip() if summary is not None else ""
                    
                    # Get authors
                    authors = []
                    for author in entry.findall('atom:author', ns):
                        name_elem = author.find('atom:name', ns)
                        if name_elem is not None:
                            authors.append(name_elem.text.strip())
                    
                    # Get published date
                    published = entry.find('atom:published', ns)
                    pub_date = published.text.strip() if published is not None else ""
                    
                    # Get ArXiv ID
                    arxiv_id = entry.find('atom:id', ns).text.strip().split('/')[-1]
                    
                    # Clean the title for filename (remove invalid characters and newlines)
                    # Remove newlines and extra whitespace first
                    clean_title = re.sub(r'\s+', ' ', title.replace('\n', ' ').replace('\r', ' ').strip())
                    # Remove invalid filename characters for Windows
                    clean_title = re.sub(r'[<>:"/\\|?*°]', '_', clean_title)
                    # Remove any remaining non-ASCII characters that might cause issues
                    clean_title = re.sub(r'[^\w\s\-_\.\(\)]', '_', clean_title)
                    # Replace multiple underscores with single underscore
                    clean_title = re.sub(r'_+', '_', clean_title)
                    # Limit length and ensure it doesn't end with a dot or space
                    clean_title = clean_title[:80].strip(' ._')
                    
                    # Fallback if title becomes empty after cleaning
                    if not clean_title or clean_title == '_':
                        clean_title = f"paper_{arxiv_id}"
                    
                    # Get the PDF link
                    pdf_link = None
                    for link in entry.findall('atom:link', ns):
                        href = link.get('href')
                        if href and '/pdf/' in href:
                            pdf_link = href
                            break
                    
                    if pdf_link:
                        # Safe logging for paper titles that may contain Unicode characters
                        safe_title = safe_str(title)
                        logger.info(f"Downloading paper {i+1}: {safe_title}")
                        
                        # Download the PDF
                        filename = f"{clean_title}.pdf"
                        filepath = output_dir / filename
                        
                        with libreq.urlopen(pdf_link) as pdf_response:
                            pdf_data = pdf_response.read()
                            
                        with open(filepath, 'wb') as pdf_file:
                            pdf_file.write(pdf_data)
                        
                        paper_info = {
                            "title": title,
                            "hotspot_name": hotspot_name,
                            "pdf_link": pdf_link,
                            "published_date": pub_date,
                            "query": query,
                            "filename": filename,
                            "api_link": url
                        }
                        
                        papers_info.append(paper_info)
                        logger.info(f"Successfully downloaded: {filename}")
                        
                    else:
                        safe_title = safe_str(title)
                        logger.warning(f"No PDF link found for: {safe_title}")
                        failed_downloads.append(title)
                    
                    # Add delay between downloads
                    time.sleep(1)
                    
                except Exception as e:
                    logger.error(f"Error downloading paper {i+1}: {str(e)}")
                    failed_downloads.append(f"Paper {i+1}")
                    continue
            
            if failed_downloads:
                logger.warning(f"Failed to download {len(failed_downloads)} papers for hotspot '{hotspot_name}'")
            
            logger.info(f"Successfully downloaded {len(papers_info)} papers for hotspot '{hotspot_name}'")
            
            # Save paper info for this hotspot
            with open(output_dir / "papers_info.json", 'w') as f:
                json.dump(papers_info, f, indent=2)
            
            return papers_info
            
        except Exception as e:
            logger.error(f"Error in single query search and download: {str(e)}")
            return []

    def _simplify_query_for_arxiv(self, query: str) -> str:
        """
        Convert simple search queries to ArXiv API format.
        
        Example:
        "plastic waste reduction" -> "all:plastic AND all:waste AND all:reduction"
        
        Args:
            query: Simple query string
            
        Returns:
            ArXiv API compatible query string
        """
        try:
            logger.info(f"Converting query: {query}")
            
            # Split by spaces and clean words
            words = [word.strip().lower() for word in query.split()]
            
            # Filter out empty words and common stop words
            words = [word for word in words if word and len(word) > 2 and 
                    word not in ['the', 'and', 'or', 'with', 'for', 'from', 'of', 'in', 'on', 'at', 'to']]
            
            # if not words:
            #     return "all:sustainability"  # Fallback for empty queries
            
            # Add all: prefix to each word and join with AND
            # arxiv_terms = [f"all:{word}" for word in words]
            # final_query = ' AND '.join(arxiv_terms)
            final_query = ' '.join(words)
            
            logger.info(f"Original query: {query}")
            logger.info(f"ArXiv API query: {final_query}")
            
            return final_query
            
        except Exception as e:
            logger.error(f"Error constructing ArXiv query: {str(e)}")
            return "all:sustainability"  # Simple fallback

