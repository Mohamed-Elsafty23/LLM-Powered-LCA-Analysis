import json
import faiss
import pickle
import numpy as np
from openai import OpenAI
from typing import List, Dict, Any
import logging
import concurrent.futures
from itertools import chain
import os
import time
from tenacity import retry, stop_after_attempt, wait_exponential
from config import PRIMARY_API_KEY, SECONDARY_API_KEY, BASE_URL
from pathlib import Path
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/sustainable_solutions.log'),
        logging.StreamHandler()
        ]
)
logger = logging.getLogger(__name__)

class APIClient:
    def __init__(self, api_key: str, base_url: str, model: str):
        """
        Initialize an API client.
        
        Args:
            api_key: API key for the LLM service
            base_url: Base URL for the LLM service
            model: Model name to use
        """
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=180.0  # Increase timeout to 180 seconds
        )
        self.model = model
        self.api_key = api_key

class VectorSearch:
    def __init__(self, vector_db_path: str, api_clients: List[APIClient]):
        """
        Initialize the VectorSearch.
        
        Args:
            vector_db_path: Path to the vector database directory
            api_clients: List of API clients to use
        """
        self.vector_db_path = vector_db_path
        self.api_clients = api_clients
        self.current_client_index = 0
        
        try:
            # Load vector database
            self.index = faiss.read_index(f"{vector_db_path}/index.faiss")
            with open(f"{vector_db_path}/metadata.pkl", 'rb') as f:
                data = pickle.load(f)
                self.metadata = data.get("metadata", [])
            
            logger.info(f"Vector index size: {self.index.ntotal}")
            logger.info(f"Available metadata entries: {len(self.metadata)}")
            logger.info("Initialized VectorSearch")
            
        except Exception as e:
            logger.error(f"Error initializing VectorSearch: {str(e)}")
            raise

    def _get_next_client(self):
        """Get the next available client in rotation."""
        client = self.api_clients[self.current_client_index]
        self.current_client_index = (self.current_client_index + 1) % len(self.api_clients)
        return client

    def _make_api_request(self, input_text: str, model: str = "e5-mistral-7b-instruct"):
        """Make API request with multiple client fallback logic."""
        last_exception = None
        
        # Try each client once
        for attempt in range(len(self.api_clients)):
            client = self._get_next_client()
            client_index = (self.current_client_index - 1) % len(self.api_clients)
            
            try:
                logging.debug(f"Attempting embedding request with client {client_index + 1}/{len(self.api_clients)}")
                response = client.client.embeddings.create(
                    input=input_text,
                    model=model
                )
                logging.debug(f"Embedding request successful with client {client_index + 1}")
                return response
                
            except Exception as e:
                last_exception = e
                logging.warning(f"Embedding request failed with client {client_index + 1}: {str(e)}")
                if attempt < len(self.api_clients) - 1:
                    logging.info(f"Trying next client...")
                    time.sleep(2)  # Brief delay before trying next client
                continue
        
        # If all clients failed, use retry logic with exponential backoff
        logging.warning("All clients failed on first attempt. Retrying with exponential backoff...")
        return self._make_api_request_with_retry(input_text, model)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=4, max=20),
        reraise=True
    )
    def _make_api_request_with_retry(self, input_text: str, model: str = "e5-mistral-7b-instruct"):
        """Make API request with retry logic and client rotation."""
        client = self._get_next_client()
        client_index = (self.current_client_index - 1) % len(self.api_clients)
        
        try:
            logging.debug(f"Retry embedding attempt with client {client_index + 1}/{len(self.api_clients)}")
            response = client.client.embeddings.create(
                input=input_text,
                model=model
            )
            return response
        except Exception as e:
            logging.warning(f"Retry embedding failed with client {client_index + 1}: {str(e)}")
            raise

    def get_embedding(self, text: str) -> np.ndarray:
        """
        Get embedding for a text using the LLM.
        
        Args:
            text: Input text
            
        Returns:
            np.ndarray: Text embedding
        """
        try:
            response = self._make_api_request(text, "e5-mistral-7b-instruct")
            
            # Convert to numpy array and reshape for FAISS
            embedding = np.array(response.data[0].embedding, dtype=np.float32)
            return embedding.reshape(1, -1)
        except Exception as e:
            logging.error(f"Error getting embedding: {str(e)}")
            raise

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for relevant papers in the vector database.
        
        Args:
            query: The search query
            top_k: Number of top results to return
            
        Returns:
            List[Dict]: List of relevant papers with their metadata
        """
        try:
            print("\n=== Starting Search Process ===")
            print(f"Query: {query}")
            print(f"Top k: {top_k}")
            print(f"Total metadata entries: {len(self.metadata)}")
            
            # Get query embedding
            query_vector = self.get_embedding(query)
            print("\n=== After Getting Embedding ===")
            print(f"Query vector shape: {query_vector.shape}")
            
            # Search the index - get more results to account for potential invalid indices
            search_k = min(top_k * 2, self.index.ntotal)
            distances, indices = self.index.search(query_vector, search_k)
            print("\n=== After FAISS Search ===")
            print(f"Found indices: {indices[0]}")
            print(f"Distances: {distances[0]}")
            
            # Get results
            results = []
            for idx, distance in zip(indices[0], distances[0]):
                print(f"\n=== Processing Index {idx} ===")
                if idx != -1 and idx < len(self.metadata):
                    try:
                        print(f"Metadata at index {idx}:")
                        print(f"Type: {type(self.metadata[idx])}")
                        print(f"Content: {self.metadata[idx]}")
                        
                        # Get the paper data from raw_data
                        paper_data = self.metadata[idx].get('raw_data', {})
                        if paper_data:
                            # Add similarity score to the paper data
                            paper_data['similarity_score'] = float(1 / (1 + distance))
                            results.append(paper_data)
                            print(f"Successfully added paper to results. Current results count: {len(results)}")
                        else:
                            print(f"Warning: No raw_data found in paper metadata")
                        
                        # Break if we have enough results
                        if len(results) >= top_k:
                            print("Reached desired number of results")
                            break
                            
                    except Exception as e:
                        print(f"Error processing index {idx}: {str(e)}")
                        logging.warning(f"Error processing index {idx}: {str(e)}")
                        continue
            
            if not results:
                print("\n=== No Valid Results Found ===")
                logger.warning("No valid results found in the search")
                # If no results found, return any available papers
                if self.metadata:
                    print("Attempting fallback to available papers")
                    logger.info("Returning available papers as fallback")
                    for i, paper_data in enumerate(self.metadata[:top_k]):
                        try:
                            print(f"\nProcessing fallback paper {i}:")
                            print(f"Type: {type(paper_data)}")
                            print(f"Content: {paper_data}")
                            
                            raw_data = paper_data.get('raw_data', {})
                            if raw_data:
                                raw_data['similarity_score'] = 0.1  # Low similarity for fallback
                                results.append(raw_data)
                                print(f"Added fallback paper. Current results count: {len(results)}")
                            else:
                                print(f"Warning: No raw_data found in fallback paper")
                                
                        except Exception as e:
                            print(f"Error processing fallback paper {i}: {str(e)}")
                            continue
            
            print(f"\n=== Search Complete ===")
            print(f"Total results found: {len(results)}")
            logger.info(f"Found {len(results)} relevant papers")
            return results
            
        except Exception as e:
            print(f"\n=== Error in Search Process ===")
            print(f"Error: {str(e)}")
            logger.error(f"Error during vector search: {str(e)}")
            return []

    def _flatten_paper_data(self, paper_data: Dict[str, Any]) -> str:
        """
        Flatten paper data into a single string containing all information.
        
        Args:
            paper_data: The paper data dictionary
            
        Returns:
            str: Flattened paper content
        """
        def flatten_dict(d: Dict[str, Any], prefix: str = "") -> str:
            content = []
            for key, value in d.items():
                current_key = f"{prefix}{key}"
                if isinstance(value, dict):
                    content.append(flatten_dict(value, f"{current_key}."))
                elif isinstance(value, list):
                    if value:  # Only process non-empty lists
                        list_content = []
                        for item in value:
                            if isinstance(item, dict):
                                list_content.append(flatten_dict(item, ""))
                            else:
                                list_content.append(str(item))
                        content.append(f"{current_key}: {' '.join(list_content)}")
                elif value is not None:  # Skip None values
                    content.append(f"{current_key}: {str(value)}")
            return " ".join(content)
        
        return flatten_dict(paper_data)

class SustainableSolutionsGenerator:
    def __init__(self, vector_db_path: str, api_configs: List[Dict[str, str]]):
        """
        Initialize the SustainableSolutionsGenerator.
        
        Args:
            vector_db_path: Path to the vector database directory
            api_configs: List of API configurations, each containing api_key, base_url, and model
        """
        self.vector_db_path = vector_db_path
        
        # Initialize API clients
        self.api_clients = [
            APIClient(
                api_key=config['api_key'],
                base_url=config['base_url'],
                model=config['model']
            ) for config in api_configs
        ]
        
        self.current_client_index = 0
        
        # Initialize vector search
        self.vector_search = VectorSearch(
            vector_db_path=vector_db_path,
            api_clients=self.api_clients
        )
            
        logger.info(f"Initialized SustainableSolutionsGenerator with {len(self.api_clients)} API clients")

    def _get_next_client(self):
        """Get the next available client in rotation."""
        client = self.api_clients[self.current_client_index]
        self.current_client_index = (self.current_client_index + 1) % len(self.api_clients)
        return client

    def _make_api_request(self, messages: List[Dict[str, str]], model: str = None):
        """Make API request with multiple client fallback logic."""
        last_exception = None
        
        # Try each client once
        for attempt in range(len(self.api_clients)):
            client = self._get_next_client()
            client_index = (self.current_client_index - 1) % len(self.api_clients)
            
            try:
                logging.debug(f"Attempting chat request with client {client_index + 1}/{len(self.api_clients)}")
                response = client.client.chat.completions.create(
                    messages=messages,
                    model=model or client.model
                )
                logging.debug(f"Chat request successful with client {client_index + 1}")
                return response
                
            except Exception as e:
                last_exception = e
                logging.warning(f"Chat request failed with client {client_index + 1}: {str(e)}")
                if attempt < len(self.api_clients) - 1:
                    logging.info(f"Trying next client...")
                    time.sleep(2)  # Brief delay before trying next client
                continue
        
        # If all clients failed, use retry logic with exponential backoff
        logging.warning("All clients failed on first attempt. Retrying with exponential backoff...")
        return self._make_api_request_with_retry(messages, model)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=4, max=20),
        reraise=True
    )
    def _make_api_request_with_retry(self, messages: List[Dict[str, str]], model: str = None):
        """Make API request with retry logic and client rotation."""
        client = self._get_next_client()
        client_index = (self.current_client_index - 1) % len(self.api_clients)
        
        try:
            logging.debug(f"Retry chat attempt with client {client_index + 1}/{len(self.api_clients)}")
            response = client.client.chat.completions.create(
                messages=messages,
                model=model or client.model
            )
            return response
        except Exception as e:
            logging.warning(f"Retry chat failed with client {client_index + 1}: {str(e)}")
            raise

    def generate_query(self, lca_report: Dict[str, Any]) -> str:
        """
        Generate a comprehensive query based on the LCA report.
        
        Args:
            lca_report: The LCA report data
            
        Returns:
            str: Only Generated query
        """
        prompt = f"""Based on the complete LCA report below, generate a specific search query to find relevant research papers 
        that could provide sustainable solutions. The query should focus on the exact environmental impacts and areas for improvement 
        identified in this specific LCA report.

        Complete LCA Report:
        {json.dumps(lca_report, indent=2)}

        Instructions:
        1. Extract the specific product/system name, materials, and processes from the LCA report
        2. Identify the exact environmental impacts and their magnitudes from the report
        3. Note the specific life cycle phases that show significant environmental burdens
        4. Create a query that combines:
           - The exact product/system name and materials from the report
           - The specific environmental impacts identified
           - The problematic life cycle phases
           - The exact processes that need improvement

        Return ONLY the search query string, with no additional text, explanations, or formatting.
        The query should be specific to this exact LCA case and its findings.
        """

        # Use the new API request method
        response = self._make_api_request([
                {"role": "system", "content": "You are an expert in generating precise research queries for finding sustainable solutions. Return only the query string, no other text."},
                {"role": "user", "content": prompt}
        ])
        
        query = response.choices[0].message.content.strip()
        logger.info(f"Generated query: {query}")
        return query

    def search_papers(self, query: str, top_k) -> List[Dict[str, Any]]:
        """
        Search for relevant papers in the vector database.
        
        Args:
            query: The search query
            top_k: Number of top results to return
            
        Returns:
            List[Dict]: List of relevant papers with their metadata
        """
        return self.vector_search.search(query, top_k)

    def analyze_paper(self, paper: Dict[str, Any], lca_report: Dict[str, Any], api_client: APIClient) -> Dict[str, Any]:
        # save paper_content to a json file
        # with open('paper_content.json', 'w', encoding='utf-8') as f:
        #     json.dump(paper, f, indent=2)
        
        """
        Analyze a single paper to extract sustainable solutions.
        
        Args:
            paper: The paper data
            lca_report: The LCA report data
            api_client: The API client to use
            
        Returns:
            Dict: Analysis results with sustainable solutions
        """
        # Extract paper ID and content
        # paper_id = list(paper.keys())[0]
        paper_content = paper
        
        # Get paper citation from metadata
        citation = paper_content.get('paper_metadata', {}).get('citation', '') 
        title = paper_content.get('paper_metadata', {}).get('title', '')
        doi = paper_content.get('paper_metadata', {}).get('doi', '')
        
        print(f"Citation: {citation} and DOI url: https://doi.org/{doi}")

        if not citation or citation == 'Not available':
            return None
        
        # Create a flattened version of the paper content for analysis
        flattened_paper = {
            'paper_metadata': paper_content.get('paper_metadata', {}),
            'Paper_content': paper_content
        }
        
        prompt = f"""Analyze the following research paper to identify sustainable solutions that could improve the environmental performance described in the LCA report.

        Research Paper Content:
        {json.dumps(flattened_paper, indent=2)}

        LCA Report Content (BASELINE DATA - these are current environmental impacts, NOT improvements):
        {json.dumps(lca_report, indent=2)}

        CRITICAL INSTRUCTIONS:
        1. The LCA report contains BASELINE/CURRENT environmental impacts, NOT improvements
        2. Your task is to find solutions in the research paper that could REDUCE these baseline impacts
        3. DO NOT confuse baseline LCA values with potential reductions
        4. DO NOT attribute LCA baseline data as achievements from the research paper
        5. ONLY propose solutions that are actually described in the research paper
        6. Clearly distinguish between:
           - BASELINE (from LCA): Current environmental impacts that need to be reduced
           - SOLUTIONS (from paper): Methods to achieve reductions from the baseline
        
        For each solution you identify:
        1. Extract the specific solution from the research paper
        2. Explain how it could theoretically reduce the BASELINE impacts from the LCA
        3. Provide implementation details ONLY from the research paper
        4. Include quantitative improvements ONLY if the paper provides them for similar applications
        5. Always cite: title: {title} with DOI: https://doi.org/{doi}
        6. Be explicit about feasibility and applicability limitations
        
        Example of CORRECT analysis:
        - BASELINE (from LCA): "Production phase consumes 0.02996 kWh"
        - SOLUTION (from paper): "Paper proposes material X that could reduce energy consumption by Y% in similar manufacturing processes"
        - RESULT: "This could potentially reduce the baseline 0.02996 kWh by Y%, resulting in approximately Z kWh"
        
        Example of INCORRECT analysis:
        - "The paper achieves 0.02996 kWh reduction" (This confuses baseline data with improvements)
        
        Focus only on solutions that are:
        - Actually described in the research paper
        - Technically feasible for ECU applications
        - Supported by evidence in the paper
        """

        response = api_client.client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are an expert in analyzing research papers for sustainable solutions. You must clearly distinguish between baseline environmental impacts (from LCA) and potential improvements (from research). Never confuse current impacts with potential reductions."},
                {"role": "user", "content": prompt}
            ],
            model=api_client.model
        )
        
        analysis = response.choices[0].message.content
        logger.info(f"Analyzed paper with similarity score: {paper.get('similarity_score', 0.0)}")
        return {
            "analysis": analysis,
            "similarity_score": paper.get('similarity_score', 0.0)
        }

    def analyze_papers_parallel(self, papers: List[Dict[str, Any]], lca_report: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Analyze multiple papers in parallel using available API clients.
        
        Args:
            papers: List of papers to analyze
            lca_report: The LCA report data
            
        Returns:
            List[Dict]: List of analysis results
        """
        analyses = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.api_clients)) as executor:
            # Create a list of (paper, api_client) pairs
            paper_api_pairs = [
                (paper, self.api_clients[i % len(self.api_clients)])
                for i, paper in enumerate(papers)
            ]
            
            # Submit all tasks
            future_to_paper = {
                executor.submit(self.analyze_paper, paper, lca_report, api_client): paper
                for paper, api_client in paper_api_pairs
            }
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_paper):
                paper = future_to_paper[future]
                try:
                    analysis = future.result()
                    analyses.append(analysis)
                except Exception as e:
                    logger.error(f"Error analyzing paper: {str(e)}")
        
        return analyses

    def review_and_cleanup(self, analyses: List[Dict[str, Any]], lca_report: Dict[str, Any]) -> Dict[str, Any]:
        """
        Review and clean up all analyses to create a final comprehensive report.
        
        Args:
            analyses: List of paper analyses
            lca_report: The LCA report data
            
        Returns:
            Dict: Final cleaned and organized report
        """
        # Filter out None analyses (papers without valid citations)
        valid_analyses = [a for a in analyses if a is not None and a.get('analysis')]
        
        if not valid_analyses:
            return {
                "final_report": "No valid solutions found with proper citations.",
                "source_analyses": []
            }
        
        # Format analyses for the prompt
        analyses_text = ""
        for i, analysis in enumerate(valid_analyses, 1):
            analyses_text += f"\n--- Analysis {i} ---\n"
            analyses_text += f"Similarity Score: {analysis.get('similarity_score', 0.0)}\n"
            analyses_text += f"Content: {analysis['analysis']}\n"
        
        prompt = f"""Review and organize the following analyses of sustainable solutions from research papers.
        Create a comprehensive report that addresses how to improve the environmental impacts identified in the LCA report.

        Complete LCA Report (BASELINE ENVIRONMENTAL IMPACTS):
        {json.dumps(lca_report, indent=2)}

        Paper Analyses from Retrieved Research:
        {analyses_text}

        CRITICAL INSTRUCTIONS FOR ACCURATE REPORTING:
        1. The LCA report shows BASELINE/CURRENT environmental impacts - these are NOT achievements or improvements
        2. ONLY use solutions and data from the "Paper Analyses" section above
        3. DO NOT create or invent any citations not found in the analyses
        4. Clearly distinguish between BASELINE impacts (from LCA) and POTENTIAL IMPROVEMENTS (from papers)
        5. Never attribute LCA baseline values as achievements from research papers
        
        For each environmental impact identified in the LCA report:
        1. State the BASELINE impact clearly (e.g., "Current production phase consumes 0.02996 kWh")
        2. List relevant solutions from the analyzed papers that could reduce this baseline
        3. For each solution:
           - Describe the solution from the research paper
           - Explain how it could theoretically reduce the baseline impact
           - Include quantitative improvements ONLY if the paper provides them for similar applications
           - Specify implementation requirements from the paper
           - Include ONLY citations from papers that were actually analyzed
           - Note applicability limitations and feasibility concerns
        
        Prioritization criteria:
        - Solutions actually described in the analyzed papers
        - Technical feasibility for ECU applications
        - Evidence quality from the research
        - Potential for meaningful impact reduction
        
        Report structure:
        1. Introduction - explain that this report proposes improvements to baseline LCA impacts
        2. Environmental Impacts and Solutions (by life cycle phase):
           - Clearly state baseline impacts from LCA
           - Propose solutions from analyzed papers
           - Distinguish between current state and potential improvements
        3. Prioritization of Solutions
        4. Conclusion
        5. References (ONLY papers that were actually analyzed)
        
        CRITICAL: 
        - Never say a paper "achieves" a baseline LCA value
        - Always clarify when values come from LCA baseline vs. potential improvements
        - Only reference information that appears in the provided paper analyses
        - If no relevant solutions exist in the papers for a specific impact, state this clearly
        """

        response = self._make_api_request([
            {"role": "system", "content": "You are an expert in synthesizing research. You MUST only use information from the provided paper analyses. Never invent or hallucinate citations. If no relevant solutions exist in the provided papers, state that clearly."},
                {"role": "user", "content": prompt}
        ])
        
        final_report = response.choices[0].message.content
        logger.info("Generated final report")
        return {
            "final_report": final_report,
            "source_analyses": valid_analyses
        }

    def validate_citations(self, report_text: str, retrieved_papers: List[Dict]) -> str:
        """Validate that all citations in the report exist in retrieved papers and check for baseline confusion."""
        # Extract DOIs from retrieved papers
        valid_dois = set()
        for paper in retrieved_papers:
            doi = paper.get('paper_metadata', {}).get('doi', '')
            if doi:
                valid_dois.add(doi)
        
        # Check for fake DOIs in report
        doi_pattern = r'https://doi\.org/([^\s\)]+)'
        found_dois = re.findall(doi_pattern, report_text)
        
        for doi in found_dois:
            if doi not in valid_dois:
                logger.warning(f"Invalid DOI found in report: {doi}")
                # Replace with warning
                report_text = report_text.replace(
                    f"https://doi.org/{doi}", 
                    "[CITATION ERROR: Paper not in retrieved database]"
                )
        
        # Check for baseline data confusion patterns
        baseline_confusion_patterns = [
            r'achieves?\s+(\d+\.?\d*)\s*kWh',
            r'provides?\s+(\d+\.?\d*)\s*kWh\s+reduction',
            r'demonstrates?\s+(\d+\.?\d*)\s*kWh\s+improvement',
            r'shows?\s+(\d+\.?\d*)\s*kWh\s+energy\s+consumption'
        ]
        
        for pattern in baseline_confusion_patterns:
            matches = re.findall(pattern, report_text, re.IGNORECASE)
            if matches:
                logger.warning(f"Potential baseline confusion detected with pattern: {pattern}")
                # Add warning comment
                report_text = f"[WARNING: Report may contain confusion between baseline LCA data and improvements]\n\n{report_text}"
                break
        
        return report_text

    def get_output_folder_from_lca_file(self, lca_file: str) -> str:
        """
        Get the output folder based on the LCA analysis file path.
        
        Args:
            lca_file: Path to the LCA analysis file
            
        Returns:
            str: Output folder path
        """
        if not lca_file:
            return "output/automotive_sample"
        
        # Extract folder from LCA file path
        lca_path = Path(lca_file)
        if lca_path.parent.name != "output":
            # If LCA file is in output/project_name/llm_based_lca_analysis.json
            return str(lca_path.parent)
        else:
            # Fallback: extract from filename if it follows pattern
            folder_name = lca_path.stem.replace("_llm_based_lca_analysis", "")
            return f"output/{folder_name}"

    def generate_sustainable_solutions(self, lca_report_path: str, output_path: str = None):
        try:
            # Determine output folder
            self.output_folder = self.get_output_folder_from_lca_file(lca_report_path)
            
            # Set default output path if not provided
            if output_path is None:
                output_path = f"{self.output_folder}/sustainable_solutions_report.txt"
            elif not str(output_path).startswith(self.output_folder):
                # If output_path doesn't include the folder, prepend it
                filename = Path(output_path).name
                output_path = f"{self.output_folder}/{filename}"
            
            # Create output folder if it doesn't exist
            Path(self.output_folder).mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Using LCA report: {lca_report_path}")
            logger.info(f"Using output folder: {self.output_folder}")
            logger.info(f"Solutions report will be saved to: {output_path}")
            
            # Load LCA report
            with open(lca_report_path, 'r') as f:
                lca_report = json.load(f)
            
            # Generate query
            query = self.generate_query(lca_report)
            logger.info(f"Generated search query: {query}")
            
            # Search for relevant papers with max 5 results
            papers = self.search_papers(query, top_k=15)
            logger.info(f"Retrieved {len(papers)} papers from vector database")
            
            # Log retrieved papers for debugging
            for i, paper in enumerate(papers):
                title = paper.get('paper_metadata', {}).get('title', 'Unknown')
                doi = paper.get('paper_metadata', {}).get('doi', 'No DOI')
                logger.info(f"Paper {i+1}: {title} (DOI: {doi})")
            
            # Save retrieved papers to the same folder
            retrieved_papers_path = f"{self.output_folder}/retrieved_papers.json"
            with open(retrieved_papers_path, 'w') as f:
                json.dump(papers, f, indent=2)
            logger.info(f"Retrieved papers saved to {retrieved_papers_path}")
            
            # Analyze papers in parallel
            analyses = self.analyze_papers_parallel(papers, lca_report)
            logger.info(f"Generated {len([a for a in analyses if a])} valid analyses")
            
            # Review and clean up
            final_report = self.review_and_cleanup(analyses, lca_report)
            
            # Validate citations
            validated_report = self.validate_citations(final_report['final_report'], papers)
            
            # Save output directly as text
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("SUSTAINABLE SOLUTIONS REPORT\n")
                f.write("=" * 50 + "\n\n")
                f.write(validated_report)
            
            logger.info(f"Successfully generated sustainable solutions report at {output_path}")
            return papers
        except Exception as e:
            logger.error(f"Error generating sustainable solutions: {str(e)}")
            raise

def main():
    # Configuration for both APIs
    api_configs = [
        {
            "api_key": PRIMARY_API_KEY,
            "base_url": BASE_URL,
            "model": "llama-3.3-70b-instruct"
        },
        {
            "api_key": SECONDARY_API_KEY,
            "base_url": BASE_URL,
            "model": "llama-3.3-70b-instruct"
        }
    ]
    
    # Initialize generator
    generator = SustainableSolutionsGenerator(
        vector_db_path="vector_db",
        api_configs=api_configs
    )
    
    # Default paths - will use automotive_sample folder
    lca_report_path = "output/automotive_sample/llm_based_lca_analysis.json"
    
    # Generate solutions (output path will be determined automatically)
    generator.generate_sustainable_solutions(lca_report_path=lca_report_path)

if __name__ == "__main__":
    main() 