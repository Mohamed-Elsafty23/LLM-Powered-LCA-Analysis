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
        logging.FileHandler(f'logs/lca_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class LLMBasedLCAAnalyzer:
    def __init__(self, api_keys: List[str], base_url: str, model: str = "llama-3.3-70b-instruct"):
        """Initialize the LLM-based LCA analyzer with multiple API keys."""
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
            
        logger.info(f"Initialized LLMBasedLCAAnalyzer with {len(self.clients)} API clients")
        
        # System prompt for LCA analysis following the paper's methodology
        self.system_prompt = """You are an expert Life Cycle Assessment (LCA) analyst specializing in automotive Electronic Control Units (ECUs).

Your expertise includes:
- ISO 14040/44 LCA standards and methodology
- Automotive electronics manufacturing and materials
- Environmental impact assessment and quantification
- ECU production, use, distribution, and end-of-life phases
- Mathematical modeling of environmental impacts
- Quantitative LCA calculations and carbon footprint analysis

CRITICAL REQUIREMENTS:
1. You are calculating BASELINE/CURRENT environmental impacts - NOT improvements or reductions
2. Your analysis represents the environmental burden of the system as it currently exists
3. ONLY work with data that is explicitly provided in the component data
4. Do NOT make assumptions about missing data or fill in gaps with estimates
5. Do NOT hallucinate or invent quantitative values
6. If specific quantitative data (weights, energy consumption, materials) is provided, use it for calculations
7. If quantitative data is NOT provided, acknowledge the limitation and work qualitatively
8. Structure responses as valid JSON with only the data that can be supported by the input
9. Base calculations ONLY on explicitly provided component data and specifications
10. Do NOT use industry standard estimates unless they are directly supported by provided data
11. Follow the same analytical approach as automotive LCA research papers
12. When quantitative data is available, provide specific values and units (kg CO2 eq, kWh, etc.)
13. When quantitative data is NOT available, provide qualitative assessments only
14. Always indicate data limitations and what additional information would be needed for complete analysis
15. All calculated values represent CURRENT environmental burdens that need to be reduced through improvements"""

    def get_output_folder_from_component_file(self, component_file: str) -> str:
        """
        Get the output folder based on the component analysis file path.
        
        Args:
            component_file: Path to the component analysis file
            
        Returns:
            str: Output folder path
        """
        if not component_file:
            return "output/automotive_sample"
        
        # Extract folder from component file path
        component_path = Path(component_file)
        if component_path.parent.name == "output":
            # If component file is in output/project_name/component_analysis.json
            if len(component_path.parts) >= 2:
                project_folder = component_path.parent
                return str(project_folder)
        
        # Fallback: extract from filename if it follows pattern
        folder_name = component_path.parent.name if component_path.parent.name != "output" else component_path.stem.replace("_component_analysis", "")
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

    def _make_robust_api_request(self, messages: List[Dict[str, str]], max_retries: int = 5) -> Dict[str, Any]:
        """Legacy method for backward compatibility - now uses the new approach."""
        return self._make_api_request(messages)

    def analyze_production_phase(self, component_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze production phase with flexible data handling."""
        try:
            prompt = f"""Analyze production phase environmental impacts using the available component data:

Component Data: {json.dumps(component_data, indent=2)}

TASK: Perform production phase LCA analysis STRICTLY based on the data provided above.

ANALYSIS REQUIREMENTS:
1. Material impacts - analyze ONLY materials explicitly mentioned in the provided data
2. Manufacturing process impacts - analyze ONLY processes explicitly mentioned in the provided data
3. Total production impact estimate - provide ONLY if sufficient data is available in the input

CRITICAL INSTRUCTIONS - PREVENTING HALLUCINATION:
- Work EXCLUSIVELY with data that is explicitly present in the component data above
- Do NOT make assumptions about missing data
- Do NOT add information not found in the provided component data
- Do NOT use industry standards or typical values unless they are explicitly provided in the input data
- Do NOT include fields like "not specified", "not available", or "not mentioned" - simply omit those fields entirely
- If weight/size data is explicitly provided in the input, use it for calculations
- If energy consumption data is explicitly provided in the input, convert to environmental impacts
- If materials are explicitly named in the input, analyze their impacts
- If manufacturing processes are explicitly described in the input, analyze their impacts
- If insufficient data is provided for quantitative analysis, provide qualitative assessment only
- Always indicate what data limitations prevent more detailed analysis

OUTPUT FORMAT:
Return JSON with material_impacts, manufacturing_processes, and total_production_impact objects.
Only include fields that have actual data from the input - omit empty or unknown fields completely.
Include a "data_limitations" field listing what additional information would be needed for more complete analysis."""

            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ]
            
            response = self._make_api_request(messages)
            content = response.choices[0].message.content
            
            try:
                result = json.loads(content)
                logger.info("Production phase analysis completed successfully")
                return result
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response: {e}")
                raise
                
        except Exception as e:
            logger.error(f"Production phase analysis failed: {str(e)}")
            raise

    def analyze_use_phase(self, component_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze use phase with flexible data handling."""
        try:
            prompt = f"""Analyze use phase environmental impacts using available component data:

            Component Data: {json.dumps(component_data, indent=2)}
            
TASK: Perform use phase LCA analysis STRICTLY based on the data provided above.

ANALYSIS REQUIREMENTS:
1. Energy consumption analysis - analyze ONLY power/energy data explicitly mentioned in the provided data
2. Usage scenarios - provide scenarios ONLY if usage data is explicitly provided in the input
3. Total use impact estimate - provide ONLY if sufficient data is available in the input

CRITICAL INSTRUCTIONS - PREVENTING HALLUCINATION:
- Work EXCLUSIVELY with data that is explicitly present in the component data above
- Do NOT make assumptions about missing data
- Do NOT add information not found in the provided component data
- Do NOT use industry standards or typical values unless they are explicitly provided in the input data
- Do NOT include fields like "not specified", "not available", or "not mentioned" - simply omit those fields entirely
- If power consumption data is explicitly provided in the input, use it for calculations
- If usage scenarios are explicitly described in the input, analyze them
- If weight data is explicitly provided in the input, consider weight-related energy impacts only if correlation is explicitly mentioned
- If operating conditions are explicitly described in the input, analyze their impacts
- If insufficient data is provided for quantitative analysis, provide qualitative assessment only
- Always indicate what data limitations prevent more detailed analysis

OUTPUT FORMAT:
Return JSON with energy_consumption_analysis, usage_scenarios, and total_use_impact.
Only include fields that have actual data from the input - omit empty or unknown fields completely.
Include a "data_limitations" field listing what additional information would be needed for more complete analysis."""

            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ]
            
            response = self._make_api_request(messages)
            content = response.choices[0].message.content
            
            try:
                result = json.loads(content)
                logger.info("Use phase analysis completed successfully")
                return result
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response: {e}")
                raise
            
        except Exception as e:
            logger.error(f"Use phase analysis failed: {str(e)}")
            raise

    def analyze_distribution_phase(self, component_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze distribution phase with flexible data handling."""
        try:
            prompt = f"""Analyze distribution phase environmental impacts using available component data:

            Component Data: {json.dumps(component_data, indent=2)}
            
TASK: Perform distribution phase LCA analysis STRICTLY based on the data provided above.

ANALYSIS REQUIREMENTS:
1. Transportation analysis - analyze ONLY transportation/logistics data explicitly mentioned in the provided data
2. Packaging impacts - analyze ONLY packaging data explicitly mentioned in the provided data
3. Total distribution impact estimate - provide ONLY if sufficient data is available in the input

CRITICAL INSTRUCTIONS - PREVENTING HALLUCINATION:
- Work EXCLUSIVELY with data that is explicitly present in the component data above
- Do NOT make assumptions about missing data
- Do NOT add information not found in the provided component data
- Do NOT use industry standards or typical values unless they are explicitly provided in the input data
- Do NOT include fields like "not specified", "not available", or "not mentioned" - simply omit those fields entirely
- If transportation distance/method data is explicitly provided in the input, use it for calculations
- If packaging materials are explicitly mentioned in the input, analyze their impacts
- If weight data is explicitly provided in the input, use it for transportation impact calculations only if transportation method is also provided
- If distribution logistics are explicitly described in the input, analyze their impacts
- If insufficient data is provided for quantitative analysis, provide qualitative assessment only
- Always indicate what data limitations prevent more detailed analysis

OUTPUT FORMAT:
Return JSON with transportation_analysis, packaging_impacts, and total_distribution_impact.
Only include fields that have actual data from the input - omit empty or unknown fields completely.
Include a "data_limitations" field listing what additional information would be needed for more complete analysis."""

            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ]
            
            response = self._make_api_request(messages)
            content = response.choices[0].message.content
            
            try:
                result = json.loads(content)
                logger.info("Distribution phase analysis completed successfully")
                return result
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response: {e}")
                raise
            
        except Exception as e:
            logger.error(f"Distribution phase analysis failed: {str(e)}")
            raise

    def analyze_end_of_life_phase(self, component_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze end-of-life phase with flexible data handling."""
        try:
            prompt = f"""Analyze end-of-life phase environmental impacts using available component data:

            Component Data: {json.dumps(component_data, indent=2)}
            
TASK: Perform end-of-life phase LCA analysis STRICTLY based on the data provided above.

ANALYSIS REQUIREMENTS:
1. Recycling analysis - analyze ONLY recycling/end-of-life data explicitly mentioned in the provided data
2. Disposal impacts - analyze ONLY disposal methods explicitly mentioned in the provided data
3. Total end-of-life impact estimate - provide ONLY if sufficient data is available in the input

CRITICAL INSTRUCTIONS - PREVENTING HALLUCINATION:
- Work EXCLUSIVELY with data that is explicitly present in the component data above
- Do NOT make assumptions about missing data
- Do NOT add information not found in the provided component data
- Do NOT use industry standards or typical values unless they are explicitly provided in the input data
- Do NOT include fields like "not specified", "not available", or "not mentioned" - simply omit those fields entirely
- If recycling rates/methods are explicitly mentioned in the input, use them for calculations
- If materials are explicitly mentioned in the input, analyze their recyclability only if recycling information is also provided
- If disposal methods are explicitly described in the input, analyze their impacts
- If insufficient data is provided for quantitative analysis, provide qualitative assessment only
- Always indicate what data limitations prevent more detailed analysis

OUTPUT FORMAT:
Return JSON with recycling_analysis, disposal_impacts, and total_eol_impact.
Only include fields that have actual data from the input - omit empty or unknown fields completely.
Include a "data_limitations" field listing what additional information would be needed for more complete analysis."""

            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ]
            
            response = self._make_api_request(messages)
            content = response.choices[0].message.content
            
            try:
                result = json.loads(content)
                logger.info("End-of-life phase analysis completed successfully")
                return result
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response: {e}")
                raise
            
        except Exception as e:
            logger.error(f"End-of-life phase analysis failed: {str(e)}")
            raise

    def generate_comprehensive_report(self, analysis_results: Dict[str, Any], component_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive LCA report with flexible data synthesis."""
        try:
            # Define report sections with their specific focuses
            report_sections = {
                "executive_summary": {
                    "instruction": "Generate a concise executive summary of the LCA findings",
                    "focus": "high-level overview"
                },
                "methodology": {
                    "instruction": "Describe the methodology and scope of the analysis",
                    "focus": "approach and standards"
                },
                "impact_assessment": {
                    "instruction": "Provide comprehensive impact assessment with detailed technical data",
                    "focus": "detailed technical analysis"
                },
                "hotspot_analysis": {
                    "instruction": "Identify key environmental hotspots and trends",
                    "focus": "critical findings"
                },
                "conclusions": {
                    "instruction": "Present conclusions and future work suggestions",
                    "focus": "summary and next steps"
                }
            }
            
            final_report = {}
            for section, config in report_sections.items():
                logger.info(f"Generating {section} section...")
                
                # Special handling for impact assessment to include detailed technical data
                if section == "impact_assessment":
                    section_prompt = f"""Generate the {section} section of the LCA report with detailed technical information.

Phase Analysis Results: {json.dumps(analysis_results, indent=2)}
Component Data: {json.dumps(component_data, indent=2)}

TASK: {config['instruction']}

REQUIRED ELEMENTS:
1. For each life cycle phase (production, distribution, use, end-of-life):
   - Total environmental impact with units
   - Detailed technical specifications
   - Process parameters and conditions
   - Material compositions and properties
   - Energy consumption details
   - Emission factors and calculations
   - Unit specifications and conversions
   - Percentage contributions

2. Include all technical details:
   - Material weights and compositions
   - Process temperatures and times
   - Energy consumption rates
   - Emission factors per unit
   - Transportation parameters
   - Recycling rates and methods
   - All relevant units and conversions

3. Structure the data hierarchically:
   - Phase level impacts
   - Component level details
   - Process level specifications
   - Material level properties

IMPORTANT INSTRUCTIONS:
- Include ALL technical specifications and parameters
- Provide complete unit information
- Include detailed emission factors
- Maintain all process parameters
- Keep all material compositions
- Preserve all technical calculations
- Return a JSON object with the section content"""
                else:
                    section_prompt = f"""Generate the {section} section of the LCA report.

Phase Analysis Results: {json.dumps(analysis_results, indent=2)}
Component Data: {json.dumps(component_data, indent=2)}

TASK: {config['instruction']}

FOCUS: {config['focus']}

IMPORTANT INSTRUCTIONS:
- Work ONLY with analysis results and component data that is actually present
- Do NOT make assumptions about missing data
- Focus on insights that can be derived from actual data present
- Provide professional interpretation of available results only
- Return a JSON object with the section content"""

                messages = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": section_prompt}
                ]
                
                try:
                    response = self._make_api_request(messages)
                    content = response.choices[0].message.content
                    section_data = json.loads(content)
                    final_report.update(section_data)
                    logger.info(f"{section} section generated successfully")
                    time.sleep(2)  # Prevent rate limiting
                except Exception as e:
                    logger.error(f"Failed to generate {section} section: {str(e)}")
                    raise

            logger.info("Comprehensive report generated successfully")
            return final_report
                
        except Exception as e:
            logger.error(f"Report generation failed: {str(e)}")
            raise

def main():
    """Main execution function for LLM-based LCA analysis."""
    # Configuration
    api_keys = [PRIMARY_API_KEY, SECONDARY_API_KEY]
    base_url = BASE_URL
    input_file = "output/automotive_sample/component_analysis.json"  # Default path
    
    try:
        # Initialize analyzer
        analyzer = LLMBasedLCAAnalyzer(api_keys, base_url)
        
        # Determine output folder from input file
        analyzer.output_folder = analyzer.get_output_folder_from_component_file(input_file)
        output_file = f"{analyzer.output_folder}/llm_based_lca_analysis.json"
        
        logger.info(f"Using input file: {input_file}")
        logger.info(f"Using output folder: {analyzer.output_folder}")
        logger.info(f"Output file will be: {output_file}")
        
        # Create output folder if it doesn't exist
        Path(analyzer.output_folder).mkdir(parents=True, exist_ok=True)
        
        # Load component data
        with open(input_file, 'r') as f:
            component_data = json.load(f)
        logger.info("Component data loaded successfully")
        
        # Analyze phases
        analysis_results = {}
        phases = ['production', 'distribution', 'use', 'end_of_life']
        
        for phase in phases:
            logger.info(f"Analyzing {phase} phase...")
            try:
                if phase == 'production':
                    analysis_results[phase] = analyzer.analyze_production_phase(component_data)
                elif phase == 'distribution':
                    analysis_results[phase] = analyzer.analyze_distribution_phase(component_data)
                elif phase == 'use':
                    analysis_results[phase] = analyzer.analyze_use_phase(component_data)
                elif phase == 'end_of_life':
                    analysis_results[phase] = analyzer.analyze_end_of_life_phase(component_data)
                
                logger.info(f"{phase.title()} phase analysis completed")
                time.sleep(2)  # Prevent rate limiting
                
            except Exception as e:
                logger.error(f"{phase.title()} phase analysis failed: {e}")
                raise
        
        # Generate comprehensive report
        logger.info("Generating comprehensive LCA report...")
        final_report = analyzer.generate_comprehensive_report(analysis_results, component_data)
        
        # Save results
        output_data = {
            "lca_report": final_report
            # "detailed_phase_analysis": analysis_results,
        }
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        logger.info(f"LLM-based LCA analysis completed successfully. Results saved to {output_file}")
        
    except Exception as e:
        logger.error(f"Critical error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main() 