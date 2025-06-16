import json
from openai import OpenAI
from typing import Dict, List, Any
import logging
import re
from pathlib import Path
from datetime import datetime
import time
from tenacity import retry, stop_after_attempt, wait_exponential
from config import PRIMARY_API_KEY, SECONDARY_API_KEY, BASE_URL

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/component_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ComponentAnalyzer:
    def __init__(self, api_keys: List[str] = None, base_url: str = None, model: str = "llama-3.3-70b-instruct"):
        """Initialize the component analyzer with LLM configuration and multiple API keys."""
        # Handle backward compatibility with single api_key parameter
        if api_keys is None:
            api_keys = [PRIMARY_API_KEY, SECONDARY_API_KEY]
        elif isinstance(api_keys, str):
            # If a single string is passed, treat it as a single API key
            api_keys = [api_keys]
        
        if base_url is None:
            base_url = BASE_URL
            
        self.api_keys = [key for key in api_keys if key]  # Filter out None/empty keys
        self.base_url = base_url
        self.model = model
        self.current_client_index = 0
        
        # Create clients for each API key
        self.clients = []
        for api_key in self.api_keys:
            client = OpenAI(
                api_key=api_key,
                base_url=base_url,
                timeout=60.0  # Increase timeout to 60 seconds
            )
            self.clients.append(client)
            
        if not self.clients:
            raise ValueError("No valid API keys provided")
            
        logger.info(f"Initialized ComponentAnalyzer with {len(self.clients)} API clients")
        
        self.system_prompt = """You are an expert in automotive Electronic Control Units (ECUs) and their Life Cycle Assessment (LCA). 
        You have deep knowledge of:
        - ECU architecture and components
        - Automotive electronics manufacturing processes
        - Automotive-grade materials and their properties
        - Vehicle electronics environmental requirements
        - Automotive industry standards and regulations
        
        Your task is to extract and structure ONLY the information that is explicitly mentioned in the provided ECU component data.
        This extracted data will be used for BASELINE LCA calculations to determine current environmental impacts.
        
        CRITICAL REQUIREMENTS - PREVENTING HALLUCINATION:
        - Extract ONLY information that is explicitly stated in the input text
        - Do NOT add any information based on your general knowledge of ECUs
        - Do NOT make assumptions about typical automotive ECU characteristics
        - Do NOT fill in missing information with industry standards
        - If certain information is not explicitly mentioned, omit those fields completely
        - Focus exclusively on what is actually written in the provided text
        - Remember: this data will be used to calculate CURRENT environmental impacts, not improvements
        
        Pay attention to explicitly mentioned aspects such as:
        - Materials that are specifically named in the text
        - Manufacturing processes that are specifically described in the text  
        - Energy consumption values that are specifically provided in the text
        - Physical characteristics that are specifically stated in the text
        - Environmental aspects that are specifically mentioned in the text
        - Operating conditions that are specifically described in the text
        - Any other information that is explicitly stated in the text
        
        Provide responses in a structured JSON format, organizing only the explicitly mentioned information into logical categories.
        Do not include fields for information that is not explicitly provided in the input text."""

    def get_output_folder(self, input_file: str) -> str:
        """
        Get the output folder name based on the input file name.
        
        Args:
            input_file: Path to the input file
            
        Returns:
            str: Output folder path
        """
        if not input_file:
            # Default folder if no input file provided
            return "output/automotive_sample"
        
        # Extract filename without extension
        input_path = Path(input_file)
        folder_name = input_path.stem  # Gets filename without extension
        output_folder = f"output/{folder_name}"
        
        # Create the folder if it doesn't exist
        Path(output_folder).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Using output folder: {output_folder}")
        return output_folder

    def _get_next_client(self):
        """Get the next available client in rotation."""
        client = self.clients[self.current_client_index]
        self.current_client_index = (self.current_client_index + 1) % len(self.clients)
        return client

    def clean_response(self, response: str) -> str:
        """Clean the LLM response for JSON parsing."""
        # Remove control characters and non-printable characters
        cleaned = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', response)
        # Normalize whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned)
        # Remove any leading/trailing whitespace
        cleaned = cleaned.strip()
        return cleaned

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
                    response_format={"type": "json_object"}
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
                response_format={"type": "json_object"}
            )
            return response
        except Exception as e:
            logger.warning(f"Retry failed with client {client_index + 1}: {str(e)}")
            raise

    def analyze_component(self, component_data: str) -> Dict[str, Any]:
        """Analyze a single component using LLM."""
        try:
            prompt = f"""Extract and structure information from the following ECU component data for BASELINE LCA analysis:
            {component_data}
            
            PURPOSE: This data will be used to calculate CURRENT environmental impacts of the ECU system.
            
            CRITICAL INSTRUCTIONS - PREVENTING HALLUCINATION:
            - Extract ONLY information that is explicitly mentioned in the text above
            - Do NOT add any information based on general ECU knowledge
            - Do NOT make assumptions about typical automotive components
            - Do NOT use industry standards to fill in missing information
            - If information is not explicitly stated in the text, do not include it in the response
            - This extracted data represents the CURRENT state of the system for impact assessment
            
            Extract and organize ONLY the explicitly mentioned information that could be relevant for:
            - Materials analysis (only if materials are specifically named in the text)
            - Manufacturing process analysis (only if processes are specifically described in the text)
            - Energy and resource consumption (only if consumption data is specifically provided in the text)
            - Physical and technical specifications (only if specifications are specifically stated in the text)
            - Environmental impacts (only if environmental aspects are specifically mentioned in the text)
            - Operational parameters (only if parameters are specifically described in the text)
            - Any other information that is explicitly stated in the provided text
            
            FORMAT REQUIREMENTS:
            - Return a JSON object organizing only the explicitly mentioned information into logical categories
            - Only include information that is explicitly stated in the provided text
            - Do not add explanatory text or assumptions
            - Omit any fields where information is not explicitly provided in the input text
            - Remember: this data will be used to calculate BASELINE environmental impacts"""

            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ]
            
            # Make API request with retry logic
            response = self._make_api_request(messages)
            
            # Clean and parse the response
            content = response.choices[0].message.content
            cleaned_content = self.clean_response(content)
            logger.debug(f"Raw LLM response: {content}")
            
            try:
                analysis = json.loads(cleaned_content)
                return analysis
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response: {content}")
                raise
            
        except Exception as e:
            logger.error(f"Error analyzing component: {str(e)}")
            raise

    def analyze_ecu_components(self, input_file: str) -> Dict[str, Any]:
        """Analyze all ECU components from the input file."""
        try:
            # Get output folder based on input file
            self.output_folder = self.get_output_folder(input_file)
            
            # Read input file
            with open(input_file, 'r') as f:
                input_data = f.read()

            # Split input into component sections
            components = self._split_components(input_data)
            
            # Analyze each component
            analysis_results = {}
            for component_name, component_data in components.items():
                logger.info(f"Analyzing ECU component: {component_name}")
                try:
                    analysis_results[component_name] = self.analyze_component(component_data)
                except Exception as e:
                    logger.error(f"Failed to analyze component {component_name}: {str(e)}")
                    analysis_results[component_name] = {"error": str(e)}
                # Add delay between requests to avoid rate limiting
                time.sleep(2)
            
            # Add overall ECU analysis
            try:
                analysis_results['ecu_overall'] = self._analyze_overall_ecu(input_data)
            except Exception as e:
                logger.error(f"Failed to analyze overall ECU: {str(e)}")
                analysis_results['ecu_overall'] = {"error": str(e)}
            
            return analysis_results
            
        except Exception as e:
            logger.error(f"Error in ECU component analysis: {str(e)}")
            raise

    def analyze_ecu_components_from_content(self, content: str, project_name: str) -> Dict[str, Any]:
        """
        Analyze all ECU components directly from content string.
        
        Args:
            content: The input data as a string
            project_name: Name of the project for output folder
            
        Returns:
            Dict: Analysis results
        """
        try:
            # Get output folder based on project name
            self.output_folder = f"output/{project_name}"
            Path(self.output_folder).mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Analyzing ECU components for project: {project_name}")
            logger.info(f"Using output folder: {self.output_folder}")

            # Split input into component sections
            components = self._split_components(content)
            
            # Analyze each component
            analysis_results = {}
            for component_name, component_data in components.items():
                logger.info(f"Analyzing ECU component: {component_name}")
                try:
                    analysis_results[component_name] = self.analyze_component(component_data)
                except Exception as e:
                    logger.error(f"Failed to analyze component {component_name}: {str(e)}")
                    analysis_results[component_name] = {"error": str(e)}
                # Add delay between requests to avoid rate limiting
                time.sleep(2)
            
            # Add overall ECU analysis
            try:
                analysis_results['ecu_overall'] = self._analyze_overall_ecu(content)
            except Exception as e:
                logger.error(f"Failed to analyze overall ECU: {str(e)}")
                analysis_results['ecu_overall'] = {"error": str(e)}
            
            return analysis_results
            
        except Exception as e:
            logger.error(f"Error in ECU component analysis: {str(e)}")
            raise

    def _split_components(self, input_data: str) -> Dict[str, str]:
        """Use LLM to intelligently identify and extract ECU components from input data."""
        try:
            # First request: Identify components and their boundaries
            identification_prompt = """You are a JSON-focused data extraction expert. Your task is to identify ECU components from the input text.

            CRITICAL: You MUST return a JSON object with EXACTLY this structure, nothing else:
            {
                "components": [
                    {
                        "name": "component name",
                        "start_line": "first line of component description",
                        "end_line": "last line of component description"
                    }
                ]
            }

            Rules:
            1. The response MUST be a valid JSON object
            2. The response MUST have a top-level "components" array
            3. Each component MUST have "name", "start_line", and "end_line" fields
            4. Do not include any additional fields or text
            5. Do not include any explanations or notes
            6. The response must be parseable by json.loads()

            Component Identification Guidelines:
            - Look for sections describing ECU parts or elements
            - Consider both explicitly named components and implicitly described ones
            - Include main components and sub-components
            - Look for technical specifications and manufacturing details
            - Consider operational parameters and material descriptions
            - Include any component-related information

            Input data:
            {input_data}"""

            max_retries = 3
            for attempt in range(max_retries):
                try:
                    messages = [
                        {"role": "system", "content": "You are a JSON-focused data extraction expert. You MUST return valid JSON objects with the exact structure specified. Do not include any additional text or explanations."},
                        {"role": "user", "content": identification_prompt}
                    ]
                    
                    response = self._make_api_request(messages)
                    cleaned_response = self.clean_response(response.choices[0].message.content)
                    
                    # Try to parse the response
                    component_boundaries = json.loads(cleaned_response)
                    
                    # Validate structure
                    if not isinstance(component_boundaries, dict):
                        raise ValueError("Response is not a dictionary")
                    if "components" not in component_boundaries:
                        raise ValueError("Missing 'components' field")
                    if not isinstance(component_boundaries["components"], list):
                        raise ValueError("'components' is not an array")
                    if not component_boundaries["components"]:
                        raise ValueError("'components' array is empty")
                    
                    # Validate each component
                    for comp in component_boundaries["components"]:
                        if not isinstance(comp, dict):
                            raise ValueError("Component is not a dictionary")
                        if not all(k in comp for k in ["name", "start_line", "end_line"]):
                            raise ValueError("Component missing required fields")
                    
                    # If we get here, the response is valid
                    break
                    
                except (json.JSONDecodeError, ValueError) as e:
                    if attempt == max_retries - 1:
                        logger.error(f"Failed to get valid component structure after {max_retries} attempts. Last error: {str(e)}")
                        logger.error(f"Last response: {cleaned_response}")
                        raise ValueError(f"Failed to get valid component structure: {str(e)}")
                    logger.warning(f"Attempt {attempt + 1} failed: {str(e)}. Retrying...")
                    time.sleep(2 ** attempt)  # Exponential backoff
            
            # Second request: Extract detailed information for each component
            components = {}
            for comp in component_boundaries["components"]:
                extraction_prompt = f"""You are a JSON-focused data extraction expert. Extract information about this ECU component.

                Component details:
                - Name: {comp['name']}
                - Start: {comp['start_line']}
                - End: {comp['end_line']}

                CRITICAL: You MUST return a JSON object with EXACTLY this structure, nothing else:
                {{
                    "component_name": "{comp['name']}",
                    "component_data": "complete extracted information about the component"
                }}

                Rules:
                1. The response MUST be a valid JSON object
                2. The response MUST have exactly these two fields
                3. Do not include any additional fields or text
                4. Do not include any explanations or notes
                5. The response must be parseable by json.loads()

                Extract ALL information about this component:
                - Technical specifications
                - Materials and properties
                - Manufacturing processes
                - Performance parameters
                - Environmental aspects
                - Any related sub-components
                - All numerical values and units
                - Any contextual information

                Input data:
                {input_data}"""

                try:
                    messages = [
                        {"role": "system", "content": "You are a JSON-focused data extraction expert. You MUST return valid JSON objects with the exact structure specified. Do not include any additional text or explanations."},
                        {"role": "user", "content": extraction_prompt}
                    ]
                    
                    response = self._make_api_request(messages)
                    component_info = json.loads(self.clean_response(response.choices[0].message.content))
                    
                    if not isinstance(component_info, dict):
                        raise ValueError("Response is not a dictionary")
                    if not all(k in component_info for k in ["component_name", "component_data"]):
                        raise ValueError("Missing required fields")
                    
                    components[component_info["component_name"]] = component_info["component_data"]
                    
                except (json.JSONDecodeError, ValueError) as e:
                    logger.error(f"Error processing component {comp['name']}: {str(e)}")
                    continue
                
                # Add delay between requests
                time.sleep(2)
            
            if not components:
                raise ValueError("No valid components were extracted from the input data")
                
            return components
            
        except Exception as e:
            logger.error(f"Error in component splitting: {str(e)}")
            raise

    def _analyze_overall_ecu(self, input_data: str) -> Dict[str, Any]:
        """Analyze overall ECU characteristics."""
        try:
            prompt = f"""Extract overall ECU system-level information from the following data for BASELINE LCA analysis:
            {input_data}
            
            PURPOSE: This data will be used to calculate CURRENT system-level environmental impacts of the ECU.
            
            CRITICAL INSTRUCTIONS - PREVENTING HALLUCINATION:
            - Extract ONLY information that is explicitly mentioned in the text above
            - Do NOT add any information based on general ECU knowledge
            - Do NOT make assumptions about typical automotive ECU systems
            - Do NOT use industry standards to fill in missing information
            - If information is not explicitly stated in the text, do not include it in the response
            - This extracted data represents the CURRENT operational state for impact assessment
            
            Extract ONLY the explicitly mentioned information about:
            - Operating conditions and parameters (only if specifically described in the text)
            - Energy consumption and efficiency metrics (only if specifically provided in the text)
            - Distribution and logistics details (only if specifically mentioned in the text)
            - End-of-life considerations (only if specifically described in the text)
            - Any other system-level information that is explicitly stated in the text
            
            FORMAT REQUIREMENTS:
            - Return a JSON object organizing only the explicitly mentioned information into logical categories
            - Only include information that is explicitly stated in the provided text
            - Do not add explanatory text or assumptions
            - Omit any fields where information is not explicitly provided in the input text
            - Remember: this data will be used to calculate BASELINE system environmental impacts"""

            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ]
            
            # Make API request with retry logic
            response = self._make_api_request(messages)
            
            # Clean and parse the response
            content = response.choices[0].message.content
            cleaned_content = self.clean_response(content)
            logger.debug(f"Raw LLM response: {content}")
            
            try:
                return json.loads(cleaned_content)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response: {content}")
                raise
            
        except Exception as e:
            logger.error(f"Error in overall ECU analysis: {str(e)}")
            raise

    def _clean_component_data(self, component_name: str, component_data: Dict[str, Any]) -> Dict[str, Any]:
        """Clean a single component's data."""
        try:
            prompt = f"""You are a JSON cleaning expert. Clean up the following component data by:
            1. Removing any fields that contain "Not available", "Not specified", or similar empty indicators
            2. Removing any empty objects or arrays
            3. Removing any redundant or duplicate information
            4. Keeping only meaningful data that could be useful for LCA analysis
            5. Maintaining the JSON structure but with only valuable information

            CRITICAL: You MUST return a valid JSON object with the exact same structure, but cleaned.
            Do not add any explanations or notes.
            The response must be parseable by json.loads()

            Component Name: {component_name}
            Component Data:
            {json.dumps(component_data, indent=2)}"""

            messages = [
                {"role": "system", "content": "You are a JSON cleaning expert. You MUST return valid JSON objects with the exact structure specified. Do not include any additional text or explanations."},
                {"role": "user", "content": prompt}
            ]
            
            response = self._make_api_request(messages)
            cleaned_content = self.clean_response(response.choices[0].message.content)
            
            try:
                cleaned_data = json.loads(cleaned_content)
                return cleaned_data
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse cleaned JSON response for component {component_name}: {cleaned_content}")
                return component_data  # Return original data if cleaning fails
            
        except Exception as e:
            logger.error(f"Error cleaning component {component_name}: {str(e)}")
            return component_data  # Return original data if cleaning fails

    def _clean_analysis_results(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Clean up analysis results by processing each component individually."""
        try:
            cleaned_results = {}
            
            # Process each component separately
            for component_name, component_data in analysis_results.items():
                logger.info(f"Cleaning data for component: {component_name}")
                cleaned_data = self._clean_component_data(component_name, component_data)
                
                # Only add non-empty components
                if cleaned_data and not self._is_empty_data(cleaned_data):
                    cleaned_results[component_name] = cleaned_data
                
                # Add delay between requests
                time.sleep(2)
            
            return cleaned_results
            
        except Exception as e:
            logger.error(f"Error in analysis results cleaning: {str(e)}")
            return analysis_results  # Return original results if cleaning fails

    def _is_empty_data(self, data: Dict[str, Any]) -> bool:
        """Check if the data is effectively empty."""
        if not data:
            return True
            
        # Check if all values are empty strings, None, or empty collections
        for value in data.values():
            if isinstance(value, dict):
                if not self._is_empty_data(value):
                    return False
            elif isinstance(value, list):
                if value and not all(not x for x in value):
                    return False
            elif value not in (None, "", "Not available", "Not specified"):
                return False
                
        return True

    def save_analysis(self, analysis_results: Dict[str, Any], output_file: str = None):
        """Save analysis results to a JSON file in the project-specific folder."""
        try:
            # Clean up the analysis results before saving
            cleaned_results = self._clean_analysis_results(analysis_results)
            
            # Use default filename if not provided
            if output_file is None:
                output_file = f"{self.output_folder}/component_analysis.json"
            elif not output_file.startswith(self.output_folder):
                # If output_file doesn't include the folder, prepend it
                filename = Path(output_file).name
                output_file = f"{self.output_folder}/{filename}"
            
            # Only save if we have results
            if cleaned_results:
                with open(output_file, 'w') as f:
                    json.dump(cleaned_results, f, indent=2)
                logger.info(f"Analysis results saved to {output_file}")
            else:
                logger.warning("No valid results to save after cleaning")
                
        except Exception as e:
            logger.error(f"Error saving analysis results: {str(e)}")
            raise

def main():
    # Configuration
    api_keys = [PRIMARY_API_KEY, SECONDARY_API_KEY]
    base_url = BASE_URL
    input_file = "automotive_sample.txt"  # Default input file
    
    try:
        # Initialize analyzer
        analyzer = ComponentAnalyzer(api_keys, base_url)
        
        # Perform analysis
        analysis_results = analyzer.analyze_ecu_components(input_file)
        
        # Save results (will use folder structure automatically)
        analyzer.save_analysis(analysis_results)
        
        logger.info("ECU component analysis completed successfully")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main() 