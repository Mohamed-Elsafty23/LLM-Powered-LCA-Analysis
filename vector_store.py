import faiss
import numpy as np
import json
from typing import Dict, Any, List, Tuple
import logging
from datetime import datetime
from pathlib import Path
from openai import OpenAI
import pickle
import time
from functools import lru_cache
from config import PRIMARY_API_KEY, BASE_URL

logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self, vector_db_path: str = "vector_db"):
        """Initialize the vector store with FAISS index and metadata."""
        self.vector_db_path = Path(vector_db_path)
        
        # Create vector_db directory if it doesn't exist
        self.vector_db_path.mkdir(exist_ok=True)
        
        # Initialize OpenAI client for embeddings
        self.client = OpenAI(
            api_key=PRIMARY_API_KEY,
            base_url=BASE_URL
        )
        self.model = "e5-mistral-7b-instruct"
        
        # Maximum tokens for the model (conservative estimate)
        self.max_tokens = 4000  # Conservative limit for e5-mistral-7b-instruct
        
        try:
            # Check if index and metadata files exist
            index_path = self.vector_db_path / "index.faiss"
            metadata_path = self.vector_db_path / "metadata.pkl"
            
            if not index_path.exists() or not metadata_path.exists():
                logger.info("Vector database files not found. Creating new index and metadata...")
                # Create new FAISS index
                self.dimension = 4096  # Dimension for e5-mistral-7b-instruct
                self.index = faiss.IndexFlatL2(self.dimension)
                
                # Initialize empty metadata
                self.metadata = []
                
                # Save the initial index and metadata
                self._save()
                logger.info("Created new vector database files")
            else:
                # Load existing index and metadata
                self.index = faiss.read_index(str(index_path))
                with open(metadata_path, 'rb') as f:
                    data = pickle.load(f)
                    self.metadata = data.get("metadata", [])
                
                # Get the dimension of the index
                self.dimension = self.index.d
            
            logger.info(f"Vector store initialized with {len(self.metadata)} papers and dimension {self.dimension}")
            
        except Exception as e:
            logger.error(f"Error initializing vector store: {str(e)}")
            raise
        
        self._embedding_cache = {}
        
    def similarity_search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar papers using the query.
        
        Args:
            query: The search query
            k: Number of results to return
            
        Returns:
            List of paper data dictionaries
        """
        try:
            # Convert query to vector
            query_vector = self._text_to_vector(query)
            
            # Ensure query vector has correct shape and type
            if query_vector.shape[1] != self.dimension:
                raise ValueError(f"Query vector dimension {query_vector.shape[1]} does not match index dimension {self.dimension}")
            
            # Search the index
            distances, indices = self.index.search(query_vector, k)
            
            # Get the results
            results = []
            for idx, distance in zip(indices[0], distances[0]):
                if idx != -1 and idx < len(self.metadata):  # Check for valid index
                    chunk_data = self.metadata[idx].copy()  # Create a copy to avoid modifying original
                    chunk_data['similarity_score'] = float(1 / (1 + distance))  # Convert distance to similarity
                    
                    # Get the full paper data
                    paper_id = chunk_data['paper_id']
                    paper_data = chunk_data['raw_data']
                    
                    # Add paper-level information
                    result = {
                        'paper_id': paper_id,
                        'paper_data': paper_data,
                        'chunk_index': chunk_data['chunk_index'],
                        'chunk_text': chunk_data['chunk_text'],
                        'similarity_score': chunk_data['similarity_score']
                    }
                    results.append(result)
            
            logger.info(f"Found {len(results)} results for query: {query[:100]}...")
            return results
            
        except Exception as e:
            logger.error(f"Error in similarity search: {str(e)}")
            return []
            
    def _text_to_vector(self, text: str) -> np.ndarray:
        """
        Convert text to vector using the same embedding model used to create the index.
        
        Args:
            text: Input text to convert to vector
            
        Returns:
            numpy array of the text vector
        """
        try:
            # Get embedding from OpenAI API
            response = self.client.embeddings.create(
                model=self.model,
                input=text
            )
            
            # Extract the embedding vector
            embedding = response.data[0].embedding
            
            # Convert to numpy array and ensure correct shape and type
            vector = np.array(embedding, dtype=np.float32)
            vector = vector.reshape(1, -1)  # Reshape to 2D array
            
            # Verify dimension
            if vector.shape[1] != self.dimension:
                raise ValueError(f"Generated vector dimension {vector.shape[1]} does not match index dimension {self.dimension}")
            
            return vector
            
        except Exception as e:
            logger.error(f"Error converting text to vector: {str(e)}")
            raise
        
    def get_paper_by_id(self, paper_id: str) -> Dict[str, Any]:
        """
        Get a specific paper by its ID.
        
        Args:
            paper_id: The ID of the paper to retrieve
            
        Returns:
            Paper data dictionary or None if not found
        """
        try:
            # Find the paper in metadata
            for paper in self.metadata:
                if paper.get('paper_metadata', {}).get('doi') == paper_id:
                    return paper.copy()  # Return a copy to avoid modifying original
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving paper by ID: {str(e)}")
            return None
            
    def get_all_papers(self) -> List[Dict[str, Any]]:
        """
        Get all papers in the vector store.
        
        Returns:
            List of all paper data dictionaries
        """
        return [paper.copy() for paper in self.metadata]  # Return copies to avoid modifying originals

    def setup_logging(self):
        """Setup logging configuration."""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # Configure root logger
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        
        # Remove any existing handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # Create formatters
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        
        # File handler with UTF-8 encoding
        file_handler = logging.FileHandler(
            log_dir / f"vector_store_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
            encoding='utf-8'
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # Console handler with UTF-8 encoding
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        # Force UTF-8 encoding for console output
        if hasattr(console_handler.stream, 'reconfigure'):
            console_handler.stream.reconfigure(encoding='utf-8')
        logger.addHandler(console_handler)

    @lru_cache(maxsize=1000)
    def _get_embedding(self, text: str) -> np.ndarray:
        """Cached embedding generation."""
        if text in self._embedding_cache:
            return self._embedding_cache[text]
        
        embedding = self._generate_embedding(text)
        self._embedding_cache[text] = embedding
        return embedding

    def _generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for text using the OpenAI API.
        
        Args:
            text: Input text to generate embedding for
            
        Returns:
            numpy array of the embedding vector
        """
        try:
            # Get embedding from OpenAI API
            response = self.client.embeddings.create(
                model=self.model,
                input=text
            )
            
            # Extract the embedding vector
            embedding = response.data[0].embedding
            
            # Convert to numpy array and ensure correct shape and type
            vector = np.array(embedding, dtype=np.float32)
            vector = vector.reshape(1, -1)  # Reshape to 2D array
            
            # Verify dimension
            if vector.shape[1] != self.dimension:
                raise ValueError(f"Generated vector dimension {vector.shape[1]} does not match index dimension {self.dimension}")
            
            return vector
            
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            raise

    def _chunk_text(self, text: str, max_chunk_size: int = 4000) -> List[str]:
        """
        Split text into chunks that fit within the model's context window.
        
        Args:
            text: The text to split into chunks
            max_chunk_size: Maximum size of each chunk in tokens
            
        Returns:
            List of text chunks
        """
        # Simple character-based chunking (can be improved with better tokenization)
        chunks = []
        current_chunk = []
        current_size = 0
        
        # Split text into sentences for better chunking
        sentences = text.split('. ')
        
        for sentence in sentences:
            # Rough estimate of tokens (4 chars ≈ 1 token)
            sentence_size = len(sentence) // 4
            
            if current_size + sentence_size > max_chunk_size:
                if current_chunk:
                    chunks.append('. '.join(current_chunk) + '.')
                current_chunk = [sentence]
                current_size = sentence_size
            else:
                current_chunk.append(sentence)
                current_size += sentence_size
        
        if current_chunk:
            chunks.append('. '.join(current_chunk) + '.')
        
        return chunks

    def _prepare_document(self, paper_id: str, paper_data: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """
        Prepare document for vector storage with chunking if necessary
        
        Args:
            paper_id (str): Unique identifier for the paper
            paper_data (Dict[str, Any]): Paper data from extracted_data.json
            
        Returns:
            Tuple[str, Dict[str, Any]]: Document text and metadata
        """
        # Combine all available text fields
        text_parts = []
        
        def extract_text(data: Dict[str, Any], prefix: str = ""):
            for key, value in data.items():
                if isinstance(value, dict):
                    extract_text(value, f"{prefix}{key.replace('_', ' ').title()}: ")
                elif isinstance(value, str) and value.strip():
                    text_parts.append(f"{prefix}{key.replace('_', ' ').title()}: {value}")
        
        extract_text(paper_data)
        combined_text = "\n".join(text_parts)
        
        # Check if text needs chunking
        chunks = self._chunk_text(combined_text)
        
        metadata = {
            "paper_id": paper_id,
            "raw_data": paper_data,
            "num_chunks": len(chunks),
            "chunk_sizes": [len(chunk) for chunk in chunks]
        }
        
        return chunks, metadata

    def store_papers(self, extracted_data_file: str, batch_size: int = 3):
        """Store papers with progress tracking and resume capability."""
        try:
            with open(extracted_data_file, 'r', encoding='utf-8') as f:
                papers = json.load(f)
            
            total_papers = len(papers)
            processed_papers = set()
            
            # Load progress if exists
            progress_file = self.vector_db_path / "processing_progress.json"
            if progress_file.exists():
                with open(progress_file, 'r') as f:
                    processed_papers = set(json.load(f))
            
            remaining_papers = {k: v for k, v in papers.items() if k not in processed_papers}
            print(f"Processing {len(remaining_papers)} remaining papers...")
            
            try:
                for i in range(0, len(remaining_papers), batch_size):
                    batch = dict(list(remaining_papers.items())[i:i + batch_size])
                    self._process_batch(batch)
                    
                    # Update progress more frequently
                    processed_papers.update(batch.keys())
                    with open(progress_file, 'w') as f:
                        json.dump(list(processed_papers), f)
                    
                    print(f"Progress: {len(processed_papers)}/{total_papers} papers processed")
            except KeyboardInterrupt:
                print("\nProcess interrupted by user. Saving progress...")
                # Save progress before exiting
                with open(progress_file, 'w') as f:
                    json.dump(list(processed_papers), f)
                print(f"Progress saved. {len(processed_papers)}/{total_papers} papers processed.")
                raise
            
        except Exception as e:
            logging.error(f"Error in store_papers: {str(e)}")
            print(f"Error in store_papers: {str(e)}")

    def _save(self):
        """Save the FAISS index and metadata to disk"""
        try:
            # Save FAISS index
            faiss.write_index(self.index, str(self.vector_db_path / "index.faiss"))
            
            # Save metadata and documents
            with open(self.vector_db_path / "metadata.pkl", "wb") as f:
                pickle.dump({
                    "metadata": self.metadata,
                }, f)
                
        except Exception as e:
            logging.error(f"Error saving to disk: {str(e)}")
            print(f"Error saving to disk: {str(e)}")

    def _load(self):
        """Load the FAISS index and metadata from disk"""
        try:
            # Load FAISS index
            self.index = faiss.read_index(str(self.vector_db_path / "index.faiss"))
            
            # Load metadata and documents
            with open(self.vector_db_path / "metadata.pkl", "rb") as f:
                data = pickle.load(f)
                self.metadata = data["metadata"]
                
        except Exception as e:
            logging.error(f"Error loading from disk: {str(e)}")
            print(f"Error loading from disk: {str(e)}")

    def search_papers(self, query: str, n_results: int = 5, 
                     filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Enhanced search with filtering capabilities."""
        try:
            query_vector = self._get_embedding(query)
            distances, indices = self.index.search(query_vector, n_results * 2)
            
            results = []
            for idx, distance in zip(indices[0], distances[0]):
                if idx == -1 or idx >= len(self.metadata):
                    continue
                
                paper_data = self.metadata[idx]
                
                # Apply filters
                if filters:
                    if not self._apply_filters(paper_data, filters):
                        continue
                
                results.append({
                    "paper": paper_data,
                    "similarity_score": float(1 / (1 + distance)),
                    "distance": float(distance)
                })
            
            return sorted(results, key=lambda x: x["similarity_score"], reverse=True)[:n_results]
        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            return []

    def _calculate_relevance_score(self, query: str, paper_data: Dict[str, Any], 
                                 distance: float, section_weights: Dict[str, float] = None) -> float:
        """
        Calculate relevance score based on multiple factors
        """
        # Base score from FAISS distance
        base_score = 1.0 / (1.0 + distance)
        
        # Section weight score
        section_score = 0.0
        if section_weights:
            for section, weight in section_weights.items():
                if section in paper_data:
                    section_score += weight
        
        # Combine scores
        final_score = base_score * (1.0 + section_score)
        return final_score

    def format_search_results(self, results: List[Dict[str, Any]], 
                             include_sections: List[str] = None) -> List[Dict[str, Any]]:
        """
        Format search results with optional section filtering
        """
        formatted_results = []
        
        for result in results:
            paper_data = result["metadata"]["raw_data"]
            formatted_result = {
                "title": paper_data.get("paper_metadata", {}).get("title", "N/A"),
                "doi": paper_data.get("paper_metadata", {}).get("doi", "N/A"),
                "relevance_score": result["relevance_score"],
                "distance": result["distance"]
            }
            
            # Add selected sections
            if include_sections:
                for section in include_sections:
                    if section in paper_data:
                        formatted_result[section] = paper_data[section]
            
            formatted_results.append(formatted_result)
        
        return formatted_results

    def _safe_load_index(self) -> bool:
        """Safely load the FAISS index with recovery options."""
        try:
            self.index = faiss.read_index(str(self.vector_db_path / "index.faiss"))
            return True
        except Exception as e:
            logger.error(f"Failed to load index: {str(e)}")
            # Create backup of corrupted index if it exists
            if (self.vector_db_path / "index.faiss").exists():
                backup_path = self.vector_db_path / f"index.faiss.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                (self.vector_db_path / "index.faiss").rename(backup_path)
            return False

    def _update_metadata(self, paper_id: str, updates: Dict[str, Any]):
        """Update metadata for a specific paper."""
        try:
            for idx, paper in enumerate(self.metadata):
                if paper.get("paper_id") == paper_id:
                    self.metadata[idx].update(updates)
                    self._save()
                    return True
            return False
        except Exception as e:
            logger.error(f"Failed to update metadata: {str(e)}")
            return False

    def _optimize_index(self):
        """Optimize FAISS index for better search performance."""
        try:
            if isinstance(self.index, faiss.IndexFlatL2):
                # Convert to IVF index for better performance
                nlist = min(100, len(self.metadata) // 10)  # Number of clusters
                quantizer = faiss.IndexFlatL2(self.dimension)
                self.index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)
                
                # Train the index
                if len(self.metadata) > 0:
                    training_vectors = np.array([self._get_embedding(paper["content"]) 
                                              for paper in self.metadata[:1000]])
                    self.index.train(training_vectors)
                
                # Add all vectors
                for paper in self.metadata:
                    vector = self._get_embedding(paper["content"])
                    self.index.add(vector)
                
                self._save()
                logger.info("Index optimized successfully")
        except Exception as e:
            logger.error(f"Index optimization failed: {str(e)}")

    def check_health(self) -> Dict[str, Any]:
        """Check the health of the vector store."""
        health_status = {
            "index_size": self.index.ntotal,
            "metadata_count": len(self.metadata),
            "dimension": self.dimension,
            "status": "healthy"
        }
        
        try:
            # Verify index and metadata consistency
            if self.index.ntotal != len(self.metadata):
                health_status["status"] = "inconsistent"
                health_status["issue"] = "Index and metadata count mismatch"
            
            # Check for corrupted entries
            corrupted = 0
            for paper in self.metadata:
                if not self._validate_paper_data(paper):
                    corrupted += 1
            
            if corrupted > 0:
                health_status["status"] = "degraded"
                health_status["corrupted_entries"] = corrupted
            
            return health_status
        except Exception as e:
            health_status["status"] = "error"
            health_status["error"] = str(e)
            return health_status

    def _process_batch(self, batch: Dict[str, Dict[str, Any]]):
        """
        Process a batch of papers and add them to the vector store.
        
        Args:
            batch: Dictionary of paper_id to paper_data
        """
        try:
            for paper_id, paper_data in batch.items():
                # Prepare document text and metadata
                chunks, metadata = self._prepare_document(paper_id, paper_data)
                
                # Store each chunk
                for i, chunk in enumerate(chunks):
                    # Get embedding for the chunk
                    vector = self._get_embedding(chunk)
                    
                    # Add to FAISS index
                    self.index.add(vector.reshape(1, -1))
                    
                    # Store metadata with chunk information
                    chunk_metadata = metadata.copy()
                    chunk_metadata.update({
                        "chunk_index": i,
                        "chunk_text": chunk
                    })
                    self.metadata.append(chunk_metadata)
                    
                    # Use print for progress instead of logger to avoid encoding issues
                    print(f"Processed chunk {i+1}/{len(chunks)} of paper: {paper_id}")
                
                # Save after each paper
                self._save()
                
        except Exception as e:
            logger.error(f"Error processing batch: {str(e)}")
            raise

if __name__ == "__main__":
    # Create output directory if it doesn't exist
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    # Initialize and run vector store
    vector_store = VectorStore()
    vector_store.store_papers("output/extracted_data.json")
    print("Vector store creation complete! Check the 'vector_db' directory for results.")