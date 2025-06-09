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
from config import PRIMARY_API_KEY, SECONDARY_API_KEY, BASE_URL

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/sustainable_solutions.log'),
        logging.StreamHandler()
    ]
)

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
            base_url=base_url
        )
        self.model = model

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
        
        try:
            # Load vector database
            self.index = faiss.read_index(f"{vector_db_path}/index.faiss")
            with open(f"{vector_db_path}/metadata.pkl", 'rb') as f:
                data = pickle.load(f)
                self.metadata = data.get("metadata", [])
            
            logging.info(f"Vector index size: {self.index.ntotal}")
            logging.info(f"Available metadata entries: {len(self.metadata)}")
            logging.info("Initialized VectorSearch")
            
        except Exception as e:
            logging.error(f"Error initializing VectorSearch: {str(e)}")
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
            # Use the first API client for embeddings
            response = self.api_clients[0].client.embeddings.create(
                input=text,
                model="e5-mistral-7b-instruct"
            )
            
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
                logging.warning("No valid results found in the search")
                # If no results found, return any available papers
                if self.metadata:
                    print("Attempting fallback to available papers")
                    logging.info("Returning available papers as fallback")
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
            logging.info(f"Found {len(results)} relevant papers")
            return results
            
        except Exception as e:
            print(f"\n=== Error in Search Process ===")
            print(f"Error: {str(e)}")
            logging.error(f"Error during vector search: {str(e)}")
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
        
        # Initialize vector search
        self.vector_search = VectorSearch(
            vector_db_path=vector_db_path,
            api_clients=self.api_clients
        )
            
        logging.info("Initialized SustainableSolutionsGenerator")

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

        # Use the first API client for query generation
        response = self.api_clients[0].client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are an expert in generating precise research queries for finding sustainable solutions. Return only the query string, no other text."},
                {"role": "user", "content": prompt}
            ],
            model=self.api_clients[0].model
        )
        
        query = response.choices[0].message.content.strip()
        logging.info(f"Generated query: {query}")
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
        
        prompt = f"""Analyze the following research paper and LCA report to identify sustainable solutions.
        Focus on practical recommendations that could be implemented to improve the environmental performance.

        Research Paper Content:
        {json.dumps(flattened_paper, indent=2)}

        LCA Report Content:
        {json.dumps(lca_report, indent=2)}

        Instructions:
        1. Extract specific sustainable solutions from the paper that directly address the environmental impacts identified in the LCA report
        2. For each solution:
           - Explain how it specifically addresses the LCA findings
           - Provide implementation feasibility based on the paper's technical details
           - Include any specific parameters, requirements, or conditions mentioned
           - Include any numerical improvements, calculations, or metrics that show the potential impact
        3. Always include the paper citation: title: {title} and with DOI link url: "https://doi.org/{doi}" if available.
        4. Focus only on solutions that are:
           - Technically feasible based on the paper's evidence
           - Directly relevant to the specific environmental impacts in the LCA report
           - Supported by concrete data or results from the paper
           
        Format the response in a clear, readable way that highlights:
        - Solution name and description
        - Specific environmental impact addressed
        - Technical implementation details
        - Numerical improvements and calculations
        - Feasibility assessment
        - Paper citation title with DOI link 	
        """

        response = api_client.client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are an expert in analyzing research papers for sustainable solutions. Focus on specific, evidence-based solutions that directly address the LCA findings. Include numerical improvements and paper citations."},
                {"role": "user", "content": prompt}
            ],
            model=api_client.model
        )
        
        analysis = response.choices[0].message.content
        logging.info(f"Analyzed paper with similarity score: {paper.get('similarity_score', 0.0)}")
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
                    logging.error(f"Error analyzing paper: {str(e)}")
        
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
        # # Filter out None analyses (papers without valid citations)
        # valid_analyses = [a for a in analyses if a is not None]
        
        # if not valid_analyses:
        #     return {
        #         "final_report": "No valid solutions found with proper citations.",
        #         "source_analyses": []
        #     }
        
        prompt = f"""Review and organize the following analyses of sustainable solutions.
        Create a comprehensive report that directly addresses the environmental impacts identified in the LCA report.

        Complete LCA Report:
        {json.dumps(lca_report, indent=2)}

        

        Instructions:
        1. For each environmental impact identified in the LCA report:
           - List all relevant solutions from the papers
           - Prioritize solutions based on:
             * Direct relevance to the specific impact
             * Technical feasibility
             * Implementation complexity
             * Expected environmental benefit
        2. For each solution:
           - Provide specific implementation steps
           - Include exact technical parameters from the papers
           - Include all numerical improvements and calculations
           - Note any limitations or requirements
           - Include the paper citation
        3. Focus on solutions that:
           - Are directly supported by the papers
           - Address specific impacts from the LCA report
           - Have clear implementation guidelines
           - Include numerical evidence of improvements
        4. Format the report with clear sections:
           - Introduction
           - Environmental Impacts and Solutions (organized by life cycle phase)
           - Prioritization of Solutions
           - Conclusion
           - References (only include papers that were actually used) : use citation title and DOI link if available.

        Format the response in a clear, readable way that emphasizes:
        - Environmental impacts and their solutions
        - Technical details and implementation steps 
        - Numerical improvements and calculations
        - Paper citations with DOI link if available
        - Priority and feasibility of each solution
        
        Feel free to include any other relevant information that would help understand and implement the solutions, such as:
        - Cost considerations and economic feasibility
        - Social impacts and stakeholder considerations
        - Regulatory requirements and compliance needs
        - Case studies and real-world examples
        - Risk factors and mitigation strategies
        - Long-term sustainability implications
        """

        response = self.api_clients[0].client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are an expert in synthesizing and organizing sustainable solutions research. Focus on specific, evidence-based solutions that directly address the LCA findings. Include numerical improvements and paper citations."},
                {"role": "user", "content": prompt}
            ],
            model=self.api_clients[0].model
        )
        
        final_report = response.choices[0].message.content
        logging.info("Generated final report")
        return {
            "final_report": final_report,
            # "source_analyses": valid_analyses
        }

    def generate_sustainable_solutions(self, lca_report_path: str, output_path: str):
        """
        Main method to generate sustainable solutions.
        
        Args:
            lca_report_path: Path to the LCA report JSON file
            output_path: Path to save the output
        """
        try:
            # Load LCA report
            with open(lca_report_path, 'r') as f:
                lca_report = json.load(f)
            
            # Generate query
            query = self.generate_query(lca_report)
            
            # Search for relevant papers with max 5 results
            papers = self.search_papers(query, top_k=5)
            
            # Analyze papers in parallel
            analyses = self.analyze_papers_parallel(papers, lca_report)
            
            # Review and clean up
            final_report = self.review_and_cleanup(analyses, lca_report)
            
            # Save output directly as text
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("SUSTAINABLE SOLUTIONS REPORT\n")
                f.write("=" * 50 + "\n\n")
                f.write(final_report['final_report'])
            
            logging.info(f"Successfully generated sustainable solutions report at {output_path}")
            
        except Exception as e:
            logging.error(f"Error generating sustainable solutions: {str(e)}")
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
    
    # Generate solutions
    generator.generate_sustainable_solutions(
        lca_report_path="output/llm_based_lca_analysis.json",
        output_path="output/sustainable_solutions_report.txt"
    )

if __name__ == "__main__":
    main() 