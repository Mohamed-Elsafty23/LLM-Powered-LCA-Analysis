import json
from openai import OpenAI
from typing import Dict, List, Any
import logging
import time
from pathlib import Path
from datetime import datetime
import re
from tenacity import retry, stop_after_attempt, wait_exponential
from config import PRIMARY_API_KEY, SECONDARY_API_KEY, BASE_URL

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/hotspot_lca_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class HotspotLCAAnalyzer:
    def __init__(self, api_keys: List[str], base_url: str, model: str = "llama-3.3-70b-instruct"):
        """Initialize the Hotspot LCA analyzer with multiple API keys."""
        self.api_keys = [key for key in api_keys if key]  # Filter out None/empty keys
        self.base_url = base_url
        self.model = model
        self.current_client_index = 0
        self.output_folder = None  # Will be set when processing files
        
        # Create clients for each API key
        self.clients = []
        for api_key in self.api_keys:
            client = OpenAI(
                api_key=api_key,
                base_url=base_url,
                timeout=180.0  # Increase timeout to 180 seconds
            )
            self.clients.append(client)
            
        if not self.clients:
            raise ValueError("No valid API keys provided")
            
        logger.info(f"Initialized HotspotLCAAnalyzer with {len(self.clients)} API clients")
        
        # System prompt for hotspot LCA analysis
        self.system_prompt = """You are an expert Life Cycle Assessment (LCA) analyst specializing in hotspot identification for Electronic Control Units (ECUs) and automotive electronics.

Your expertise includes:
- ISO 14040/44 LCA standards and methodology
- Hotspot identification across all life cycle phases
- Environmental impact assessment and quantification
- ECU production, use, distribution, and end-of-life phases
- Mathematical modeling of environmental impacts
- Quantitative LCA calculations and carbon footprint analysis

CRITICAL REQUIREMENTS FOR HOTSPOT ANALYSIS:
1. You are identifying environmental HOTSPOTS - the most significant environmental impacts in the system
2. Work EXCLUSIVELY with data that is explicitly provided in the raw input data
3. Do NOT make assumptions about missing data or fill in gaps with estimates
4. Do NOT hallucinate or invent quantitative values
5. If specific quantitative data (weights, energy consumption, materials) is provided, use it for hotspot calculations
6. If quantitative data is NOT provided, provide qualitative hotspot assessment only
7. Structure responses as valid JSON with only the data that can be supported by the input
8. Base hotspot identification ONLY on explicitly provided raw input data
9. Follow automotive LCA research methodology for hotspot identification
10. When quantitative data is available, provide specific hotspot values and units (kg CO2 eq, kWh, etc.)
11. When quantitative data is NOT available, provide qualitative hotspot assessments only
12. Always indicate data limitations and what additional information would be needed for complete hotspot analysis
13. Identify hotspots across ALL life cycle phases: production, distribution, use, end-of-life
14. For each hotspot, provide specific details about the environmental impact source
15. Prioritize hotspots by environmental significance and impact magnitude"""

    def get_output_folder_from_input_file(self, input_file: str) -> str:
        """
        Get the output folder based on the input file path.
        
        Args:
            input_file: Path to the input file
            
        Returns:
            str: Output folder path
        """
        if not input_file:
            return "output/automotive_sample"
        
        # Extract folder from input file path
        input_path = Path(input_file)
        folder_name = input_path.stem  # Gets filename without extension
        return f"output/{folder_name}"

    def _get_next_client(self):
        """Get the next available client in rotation."""
        client = self.clients[self.current_client_index]
        self.current_client_index = (self.current_client_index + 1) % len(self.clients)
        return client

    def _make_api_request(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """Make API request with multiple client fallback logic."""
        last_exception = None
        
        # Try each client once
        for attempt in range(len(self.clients)):
            client = self._get_next_client()
            client_index = (self.current_client_index - 1) % len(self.clients)
            
            try:
                logger.debug(f"Attempting request with client {client_index + 1}/{len(self.clients)}")
                response = client.chat.completions.create(
                    messages=messages,
                    model=self.model,
                    response_format={"type": "json_object"},
                    temperature=0.1
                )
                logger.debug(f"Request successful with client {client_index + 1}")
                return response
                
            except Exception as e:
                last_exception = e
                logger.warning(f"Request failed with client {client_index + 1}: {str(e)}")
                if attempt < len(self.clients) - 1:
                    logger.info(f"Trying next client...")
                    time.sleep(2)  # Brief delay before trying next client
                continue
        
        # If all clients failed, use retry logic with exponential backoff
        logger.warning("All clients failed on first attempt. Retrying with exponential backoff...")
        return self._make_api_request_with_retry(messages)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=4, max=20),
        reraise=True
    )
    def _make_api_request_with_retry(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """Make API request with retry logic and client rotation."""
        client = self._get_next_client()
        client_index = (self.current_client_index - 1) % len(self.clients)
        
        try:
            logger.debug(f"Retry attempt with client {client_index + 1}/{len(self.clients)}")
            response = client.chat.completions.create(
                messages=messages,
                model=self.model,
                response_format={"type": "json_object"},
                temperature=0.1
            )
            return response
        except Exception as e:
            logger.warning(f"Retry failed with client {client_index + 1}: {str(e)}")
            raise

    def perform_hotspot_analysis(self, raw_input_data: str) -> Dict[str, Any]:
        """Perform comprehensive hotspot analysis on raw input data."""
        try:
            prompt = f"""Perform comprehensive environmental hotspot analysis on the following raw input data:

Raw Input Data: {raw_input_data}

TASK: Identify and analyze environmental HOTSPOTS across all life cycle phases using ONLY the data explicitly provided above.

HOTSPOT ANALYSIS REQUIREMENTS:
1. Production Phase Hotspots - identify the most significant environmental impacts in production using only explicitly mentioned data
2. Distribution Phase Hotspots - identify the most significant environmental impacts in distribution using only explicitly mentioned data  
3. Use Phase Hotspots - identify the most significant environmental impacts during use using only explicitly mentioned data
4. End-of-Life Phase Hotspots - identify the most significant environmental impacts at end-of-life using only explicitly mentioned data
5. Overall System Hotspots - rank all hotspots by environmental significance

CRITICAL INSTRUCTIONS - PREVENTING HALLUCINATION:
- Work EXCLUSIVELY with data that is explicitly present in the raw input data above
- Do NOT make assumptions about missing data
- Do NOT add information not found in the provided raw input data
- Do NOT use industry standards or typical values unless they are explicitly provided in the input data
- Do NOT include fields like "not specified", "not available", or "not mentioned" - simply omit those fields entirely
- If weight/size data is explicitly provided in the input, use it for hotspot calculations
- If energy consumption data is explicitly provided in the input, identify energy-related hotspots
- If materials are explicitly named in the input, analyze their hotspot potential
- If processes are explicitly described in the input, analyze their hotspot potential
- If insufficient data is provided for quantitative hotspot analysis, provide qualitative assessment only
- Always indicate what data limitations prevent more detailed hotspot analysis

OUTPUT FORMAT:
Return JSON with the following structure:
{{
  "production_hotspots": [
    {{
      "hotspot_name": "string",
      "impact_category": "string", 
      "environmental_significance": "high/medium/low",
      "impact_source": "string",
      "quantitative_impact": "string (if available)",
      "description": "string"
    }}
  ],
  "distribution_hotspots": [...],
  "use_hotspots": [...],
  "end_of_life_hotspots": [...],
  "overall_hotspot_ranking": [
    {{
      "rank": number,
      "hotspot_name": "string",
      "life_cycle_phase": "string",
      "environmental_significance": "high/medium/low",
      "priority_justification": "string"
    }}
  ],
  "data_limitations": ["string"]
}}

Only include hotspots that can be identified from the explicitly provided raw input data - omit empty or unknown fields completely.
Include a "data_limitations" field listing what additional information would be needed for more complete hotspot analysis."""

            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ]
            
            response = self._make_api_request(messages)
            content = response.choices[0].message.content
            
            try:
                result = json.loads(content)
                logger.info("Hotspot analysis completed successfully")
                return result
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response: {e}")
                raise
                
        except Exception as e:
            logger.error(f"Hotspot analysis failed: {str(e)}")
            raise

    def generate_search_queries_for_hotspots(self, hotspot_analysis: Dict[str, Any], raw_input_data: str) -> Dict[str, str]:
        """Generate highly specific search queries for quantitative sustainability solutions."""
        try:
            prompt = f"""Generate highly specific search queries to find research papers with quantitative sustainability solutions for each hotspot.

Raw Input Data: {raw_input_data}

Hotspot Analysis: {json.dumps(hotspot_analysis, indent=2)}

REQUIREMENTS:
1. Focus on the exact materials, processes, and parameters from the raw input data
2. Target papers with:
   - Specific energy/resource reduction percentages 
   - Process optimization techniques
   - Quantitative efficiency improvements
   - Material substitution studies
   - Parameter optimization research

For each hotspot, create queries that combine:
- Component name (e.g. "Housing", "Radiator")
- Material (e.g. "PBT", "Aluminium") 
- Manufacturing process (e.g. "Injection Molding", "Die casting")
- Sustainability terms (e.g. "energy efficiency", "optimization")
- Quantitative metrics (e.g. "kWh reduction", "cycle time")

Example query structure:
"injection molding PBT housing energy efficiency optimization cycle time reduction"

  OUTPUT FORMAT:
  {{
    "hotspot_queries": {{
      "hotspot_name": "specific technical query using exact terms from input data",
      ...
    }},
    "query_strategy": "description of query construction approach"
  }}

Focus on finding papers with concrete, measurable improvements. Avoid generic sustainability terms."""

            messages = [
                {"role": "system", "content": "You are an expert in generating precise research queries for sustainability solutions. Create targeted queries that will find engineering papers with quantitative process improvements and measurable environmental benefits."},
                {"role": "user", "content": prompt}
            ]
            
            response = self._make_api_request(messages)
            content = response.choices[0].message.content
            
            try:
                result = json.loads(content)
                logger.info(f"Generated {len(result.get('hotspot_queries', {}))} improved search queries for hotspots")
                return result
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response: {e}")
                raise
        except Exception as e:
            logger.error(f"Search query generation failed: {str(e)}")
            raise

    def analyze_from_raw_input(self, input_file: str) -> Dict[str, Any]:
        """Perform complete hotspot LCA analysis directly from raw input file."""
        try:
            # Set output folder
            self.output_folder = self.get_output_folder_from_input_file(input_file)
            Path(self.output_folder).mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Using input file: {input_file}")
            logger.info(f"Using output folder: {self.output_folder}")
            
            # Read raw input data
            with open(input_file, 'r', encoding='utf-8') as f:
                raw_input_data = f.read()
            
            logger.info("Raw input data loaded successfully")
            
            # Perform hotspot analysis
            logger.info("Starting hotspot analysis...")
            hotspot_analysis = self.perform_hotspot_analysis(raw_input_data)
            
            # Generate search queries for each hotspot
            logger.info("Generating search queries for hotspots...")
            search_queries = self.generate_search_queries_for_hotspots(hotspot_analysis, raw_input_data)
            
            # Combine results
            complete_analysis = {
                "hotspot_analysis": hotspot_analysis,
                "search_queries": search_queries
            }
            
            # Save results
            output_file = f"{self.output_folder}/hotspot_lca_analysis.json"
            with open(output_file, 'w') as f:
                json.dump(complete_analysis, f, indent=2)
            
            logger.info(f"Hotspot LCA analysis completed successfully. Results saved to {output_file}")
            return complete_analysis
                
        except Exception as e:
            logger.error(f"Error in hotspot LCA analysis: {str(e)}")
            raise

def main():
    """Main execution function for hotspot LCA analysis."""
    # Configuration
    api_keys = [PRIMARY_API_KEY, SECONDARY_API_KEY]
    base_url = BASE_URL
    input_file = "automotive_sample_input.txt"  # Default input file
    
    try:
        # Initialize analyzer
        analyzer = HotspotLCAAnalyzer(api_keys, base_url)
        
        # Perform hotspot analysis directly from raw input
        analysis_results = analyzer.analyze_from_raw_input(input_file)
        
        logger.info("Hotspot LCA analysis completed successfully")
        return analysis_results
        
    except Exception as e:
        logger.error(f"Critical error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main() 