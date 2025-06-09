import json
from openai import OpenAI
from typing import Dict, List, Any
import logging
import time
from pathlib import Path
from datetime import datetime
import re
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
        """Initialize the LLM-based LCA analyzer."""
        self.api_keys = api_keys
        self.current_key_index = 0
        self.base_url = base_url
        self.model = model
        
        # Increase timeout to 180 seconds
        self.request_timeout = 180.0
        
        # Initialize rate limiting tracking
        self.rate_limits = {
            key: {
                "requests": 0,
                "last_reset": datetime.now(),
                "errors": 0,
                "consecutive_failures": 0
            } for key in api_keys
        }
        
        # Initialize clients for each API key
        self.clients = {
            key: OpenAI(
                api_key=key,
                base_url=base_url,
                timeout=self.request_timeout
            ) for key in api_keys
        }
        
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
1. Provide QUANTITATIVE results with specific values and units (kg CO2 eq, kWh, etc.)
2. Follow cradle-to-grave LCA methodology (Production → Distribution → Use → End-of-Life)
3. Calculate percentage breakdowns for each phase contribution
4. Use mathematical formulas for energy consumption and environmental impacts
5. Provide detailed component-level analysis with specific impact values
6. Structure responses as valid JSON with numerical data
7. Base calculations on provided component data and engineering principles
8. Do NOT use placeholder values - calculate realistic estimates based on component specifications
9. Follow the same analytical approach as automotive LCA research papers
10. Provide multiple usage scenarios (low, default, high) where applicable"""

        # Initialize OpenAI client
        self.client = OpenAI(
            api_key=PRIMARY_API_KEY,
            base_url=BASE_URL
        )

    def _get_best_api_key(self) -> str:
        """Get the best available API key."""
        current_time = datetime.now()
        
        # Reset old counters
        for key in self.api_keys:
            rate_limit = self.rate_limits[key]
            if (current_time - rate_limit["last_reset"]).total_seconds() > 1800:
                rate_limit["requests"] = 0
                rate_limit["errors"] = 0
                rate_limit["consecutive_failures"] = 0
                rate_limit["last_reset"] = current_time
        
        # Find best key
        best_key = None
        best_score = float('-inf')
        
        for key in self.api_keys:
            rate_limit = self.rate_limits[key]
            if rate_limit["consecutive_failures"] >= 3:
                continue
            
            score = 100 - (rate_limit["errors"] * 10) - (rate_limit["consecutive_failures"] * 20)
            if score > best_score:
                best_score = score
                best_key = key
        
        if best_key is None:
            best_key = min(self.api_keys, key=lambda k: self.rate_limits[k]["consecutive_failures"])
            self.rate_limits[best_key]["consecutive_failures"] = 0
        
        return best_key

    def _make_robust_api_request(self, messages: List[Dict[str, str]], max_retries: int = 5) -> Dict[str, Any]:
        """Make API request with enhanced reliability."""
        
        total_attempts = 0
        max_total_attempts = len(self.api_keys) * max_retries
        
        while total_attempts < max_total_attempts:
            # Cycle through API keys
            api_key = self.api_keys[total_attempts % len(self.api_keys)]
            attempt = (total_attempts // len(self.api_keys)) + 1
            
            try:
                client = self.clients[api_key]
                self.rate_limits[api_key]["requests"] += 1
                
                logger.info(f"Making API request with key ending in ...{api_key[-4:]} (attempt {attempt}/{max_retries})")
                
                # Add exponential backoff between retries with longer initial wait
                if total_attempts > 0:
                    backoff_time = min(5 * (2 ** (total_attempts // len(self.api_keys))), 60)  # Start with 5s, cap at 60s
                    logger.info(f"Waiting {backoff_time} seconds before retry...")
                    time.sleep(backoff_time)
                
                response = client.chat.completions.create(
                    messages=messages,
                    model=self.model,
                    response_format={"type": "json_object"},
                    temperature=0.1,
                    timeout=self.request_timeout
                )
                
                self.rate_limits[api_key]["consecutive_failures"] = 0
                logger.info("API request successful")
                return response
                
            except Exception as e:
                error_msg = str(e)
                self.rate_limits[api_key]["errors"] += 1
                self.rate_limits[api_key]["consecutive_failures"] += 1
                
                logger.warning(f"API request failed with key ...{api_key[-4:]} (attempt {attempt}): {error_msg}")
                
                # If it's a timeout error, try next key immediately
                if "timeout" in error_msg.lower():
                    logger.info("Timeout detected, switching API key...")
                    time.sleep(2)  # Increased delay before switching keys
                
                total_attempts += 1
        
        logger.error(f"All API request attempts failed across all keys after {max_total_attempts} total attempts")
        raise Exception("All API requests failed - check network and API status")

    def analyze_production_phase(self, component_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze production phase with flexible data handling."""
        try:
            prompt = f"""Analyze production phase environmental impacts using the available component data:

Component Data: {json.dumps(component_data, indent=2)}

TASK: Perform production phase LCA analysis based on whatever data is available.

ANALYSIS REQUIREMENTS:
1. Material impacts - analyze any materials mentioned in the data
2. Manufacturing process impacts - analyze any processes mentioned
3. Total production impact estimate

IMPORTANT INSTRUCTIONS:
- Work ONLY with data that is actually present in the component data
- Do NOT make assumptions about missing data
- Do NOT include fields like "not specified", "not available", or "not mentioned" - simply omit those fields
- Only include information that is explicitly present in the input data
- Focus on quantifiable data where available in the input
- If weight/size data is available in input, use it for calculations
- If energy consumption data is available in input, convert to environmental impacts
- Include any other important LCA-relevant information found in the data that wasn't specifically mentioned in these instructions
- Provide realistic estimates based on automotive industry standards only when actual data supports calculations

Return JSON with material_impacts, manufacturing_processes, and total_production_impact objects.
Only include fields that have actual data - omit empty or unknown fields completely."""

            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ]
            
            response = self._make_robust_api_request(messages)
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
            
TASK: Perform use phase LCA analysis based on whatever data is available.

ANALYSIS REQUIREMENTS:
1. Energy consumption analysis - analyze any power/energy data mentioned
2. Usage scenarios - provide multiple scenarios if possible
3. Total use impact estimate

IMPORTANT INSTRUCTIONS:
- Work ONLY with data that is actually present in the component data
- Do NOT make assumptions about missing data
- Do NOT include fields like "not specified", "not available", or "not mentioned" - simply omit those fields
- Only include information that is explicitly present in the input data
- If power consumption data is available in input, use it for calculations
- Consider multiple usage scenarios if possible based on available data
- If weight data is available in input, consider weight-related energy impacts
- Focus on operational impacts and energy consumption found in the input
- Include any other important LCA-relevant information found in the data that wasn't specifically mentioned in these instructions

Return JSON with energy_consumption_analysis, usage_scenarios, and total_use_impact.
Only include fields that have actual data - omit empty or unknown fields completely."""

            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ]
            
            response = self._make_robust_api_request(messages)
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
            
TASK: Perform distribution phase LCA analysis based on whatever data is available.

ANALYSIS REQUIREMENTS:
1. Transportation analysis - analyze any transportation/logistics data mentioned
2. Packaging impacts - analyze any packaging data mentioned
3. Total distribution impact estimate

IMPORTANT INSTRUCTIONS:
- Work ONLY with data that is actually present in the component data
- Do NOT make assumptions about missing data
- Do NOT include fields like "not specified", "not available", or "not mentioned" - simply omit those fields
- Only include information that is explicitly present in the input data
- If transportation distance/method data is available in input, use it for calculations
- If packaging materials are mentioned in input, analyze their impacts
- If weight data is available in input, use it for transportation impact calculations
- Focus on transportation and packaging impacts found in the input
- Include any other important LCA-relevant information found in the data that wasn't specifically mentioned in these instructions

Return JSON with transportation_analysis, packaging_impacts, and total_distribution_impact.
Only include fields that have actual data - omit empty or unknown fields completely."""

            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ]
            
            response = self._make_robust_api_request(messages)
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
            
TASK: Perform end-of-life phase LCA analysis based on whatever data is available.

ANALYSIS REQUIREMENTS:
1. Recycling analysis - analyze any recycling/end-of-life data mentioned
2. Disposal impacts - analyze disposal methods mentioned
3. Total end-of-life impact estimate

IMPORTANT INSTRUCTIONS:
- Work ONLY with data that is actually present in the component data
- Do NOT make assumptions about missing data
- Do NOT include fields like "not specified", "not available", or "not mentioned" - simply omit those fields
- Only include information that is explicitly present in the input data
- If recycling rates/methods are mentioned in input, use them for calculations
- If materials are mentioned in input, consider their recyclability
- Focus on recycling potential and disposal impacts found in the input
- Consider both benefits (avoided impacts) and costs (disposal impacts) based on available data
- Include any other important LCA-relevant information found in the data that wasn't specifically mentioned in these instructions

Return JSON with recycling_analysis, disposal_impacts, and total_eol_impact.
Only include fields that have actual data - omit empty or unknown fields completely."""

            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ]
            
            response = self._make_robust_api_request(messages)
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
                    response = self._make_robust_api_request(messages)
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
    api_keys = [
        "d9960fad1d2aaa16167902b0d26e369f",
        "d1c9ed1ca70b9721dee1087d93f9662a"
    ]
    base_url = "https://chat-ai.academiccloud.de/v1"
    input_file = "output/component_analysis.json"
    output_file = "output/llm_based_lca_analysis.json"
    
    try:
        # Initialize analyzer
        analyzer = LLMBasedLCAAnalyzer(api_keys, base_url)
        logger.info("LLM-based LCA Analyzer initialized successfully")
        
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
        
        # Performance summary
        total_errors = sum(analyzer.rate_limits[key]["errors"] for key in analyzer.api_keys)
        logger.info(f"Analysis completed with {total_errors} errors across all API keys")
        
    except Exception as e:
        logger.error(f"Critical error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main() 