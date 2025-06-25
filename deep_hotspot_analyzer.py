"""
Deep Hotspot LCA Analysis System
Performs focused, quantitative analysis for each ECU hotspot component following IFIP methodology.
Generates professional LCA reports with primary data focus and quantitative sustainability metrics.
"""

import json
import logging
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional
import time
import concurrent.futures
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
import sys
import unicodedata
from tavily import TavilyClient

# PDF generation imports
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY

# Configure logging
def safe_str(text):
    """Convert text to ASCII-safe string for logging on Windows."""
    if isinstance(text, str):
        try:
            text.encode('cp1252')
            return text
        except UnicodeEncodeError:
            replacements = {
                'γ': 'gamma', 'π': 'pi', 'α': 'alpha', 'β': 'beta', 'δ': 'delta',
                'ε': 'epsilon', 'θ': 'theta', 'λ': 'lambda', 'μ': 'mu', 'ν': 'nu',
                'ρ': 'rho', 'σ': 'sigma', 'τ': 'tau', 'φ': 'phi', 'χ': 'chi',
                'ψ': 'psi', 'ω': 'omega'
            }
            for unicode_char, replacement in replacements.items():
                text = text.replace(unicode_char, replacement)
            try:
                text.encode('cp1252')
                return text
            except UnicodeEncodeError:
                return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')
    return str(text)

class UnicodeStreamHandler(logging.StreamHandler):
    def emit(self, record):
        try:
            # Skip HTTP request logs that cause formatting issues
            if hasattr(record, 'msg') and 'HTTP Request:' in str(record.msg):
                return
            
            if hasattr(record, 'msg') and isinstance(record.msg, str):
                record.msg = safe_str(record.msg)
            if hasattr(record, 'args') and record.args:
                record.args = tuple(safe_str(arg) for arg in record.args)
            super().emit(record)
        except (UnicodeEncodeError, UnicodeDecodeError) as e:
            raise RuntimeError(f"Unicode encoding error in log message: {e}") from e
        except Exception as e:
            # Handle any other logging errors gracefully
            print(f"Logging error: {e}")
            return

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/deep_hotspot_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log', encoding='utf-8'),
        UnicodeStreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class APIClient:
    def __init__(self, api_key: str, base_url: str, model: str):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

class DeepHotspotAnalyzer:
    def __init__(self, api_configs: List[Dict[str, str]], tavily_api_key: str):
        """Initialize with multiple API configurations and Tavily for web search."""
        self.api_clients = [APIClient(config["api_key"], config["base_url"], config["model"]) for config in api_configs]
        self.current_client_index = 0
        self.tavily_client = TavilyClient(api_key=tavily_api_key)
        self.output_folder = None
        
        logger.info(f"Initialized DeepHotspotAnalyzer with {len(self.api_clients)} API clients")

    def _get_next_client(self):
        """Get the next API client for load balancing."""
        client = self.api_clients[self.current_client_index]
        self.current_client_index = (self.current_client_index + 1) % len(self.api_clients)
        return client

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=4, max=20), reraise=True)
    def _make_api_request(self, messages: List[Dict[str, str]], model: str = None):
        """Make API request with retry logic."""
        api_client = self._get_next_client()
        try:
            response = api_client.client.chat.completions.create(
                messages=messages,
                model=model or api_client.model,
                temperature=0.3,  # Lower temperature for more precise analysis
                max_tokens=4000
            )
            return response
        except Exception as e:
            logger.error(f"API request failed: {str(e)}")
            raise

    def _web_search_for_quantitative_data(self, search_query: str) -> Dict[str, Any]:
        """Perform web search for additional quantitative data and return structured results."""
        search_metadata = {
            "search_query": search_query,
            "search_results": [],
            "formatted_content": "",
            "success": False,
            "error_message": None
        }
        
        max_retries = 2  # Reduced retries since we're now sequential
        for attempt in range(max_retries):
            try:
                # Add delay for rate limiting - longer for first attempt
                delay = 3 if attempt == 0 else 5
                time.sleep(delay)
                
                logger.info(f"Performing web search (attempt {attempt + 1}/{max_retries}): {search_query}")
                
                # Create a fresh client instance for each search to avoid connection issues
                fresh_client = TavilyClient(api_key="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")
                response = fresh_client.search(search_query)
                
                # Debug: Log the raw response
                logger.info(f"Raw response type: {type(response)}, has results: {bool(response and response.get('results'))}")
                
                # More detailed response checking
                if not response:
                    logger.warning(f"Web search returned empty response for: {search_query} (attempt {attempt + 1})")
                    if attempt == max_retries - 1:
                        search_metadata["error_message"] = "Web search returned empty response after multiple attempts"
                        search_metadata["formatted_content"] = search_metadata["error_message"]
                        return search_metadata
                    continue
                
                if not response.get('results'):
                    logger.warning(f"Web search returned no results for: {search_query} (attempt {attempt + 1})")
                    if attempt == max_retries - 1:
                        search_metadata["error_message"] = "No web search results found after multiple attempts"
                        search_metadata["formatted_content"] = search_metadata["error_message"]
                        return search_metadata
                    continue
                
                results = response.get('results', [])
                if len(results) == 0:
                    logger.warning(f"Web search results array empty for: {search_query} (attempt {attempt + 1})")
                    if attempt == max_retries - 1:
                        search_metadata["error_message"] = "No web search results found after multiple attempts"
                        search_metadata["formatted_content"] = search_metadata["error_message"]
                        return search_metadata
                    continue
                
                # If we get here, we have valid results
                search_results_formatted = []
                for i, result in enumerate(results[:3]):  # Top 3 results
                    title = result.get('title', 'Unknown Title')
                    url = result.get('url', 'Unknown URL')
                    content = result.get('content', 'No content available')
                    
                    # Store structured result metadata
                    search_metadata["search_results"].append({
                        "rank": i + 1,
                        "title": title,
                        "url": url,
                        "content": content
                    })
                    
                    # Format each result clearly for LLM processing
                    search_results_formatted.append(f"""
WEB SEARCH RESULT {i+1}:
Title: {title}
URL: {url}
Content: {content}
Citation Format: [{title}] (URL: {url})
""")
                
                # Combine all results with clear header
                formatted_results = f"""
=== WEB SEARCH RESULTS FOR QUERY: "{search_query}" ===

{chr(10).join(search_results_formatted)}

=== END OF WEB SEARCH RESULTS ===

IMPORTANT: Use these web search results ONLY if they contain relevant quantitative data (numbers, percentages, formulas, metrics) that can help with sustainability analysis. Each piece of quantitative data used from web search MUST be cited with the format: [Title] (URL: full_url)
"""
                
                search_metadata["formatted_content"] = formatted_results
                search_metadata["success"] = True
                
                logger.info(f"✅ Web search SUCCESS: Found {len(results)} results for '{search_query}' (total length: {len(formatted_results)} chars)")
                return search_metadata
                
            except Exception as e:
                logger.error(f"Web search failed for query '{search_query}' (attempt {attempt + 1}): {str(e)}")
                if attempt == max_retries - 1:
                    search_metadata["error_message"] = f"Web search failed for query: {search_query} - Error: {str(e)}"
                    search_metadata["formatted_content"] = search_metadata["error_message"]
                    return search_metadata
                continue
        
        # This shouldn't be reached, but just in case
        search_metadata["error_message"] = f"Web search failed for query: {search_query} - Maximum retries exceeded"
        search_metadata["formatted_content"] = search_metadata["error_message"]
        return search_metadata

    def parse_existing_report(self, report_file: str) -> Dict[str, Dict[str, Any]]:
        """Parse the existing sustainability report to extract hotspot data and papers."""
        try:
            with open(report_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            hotspots_data = {}
            
            # Split content by ### markers  
            sections = content.split('###')[1:]
            
            for section in sections:
                if not section.strip():
                    continue
                    
                # Get the first line as the hotspot name
                lines = section.strip().split('\n')
                if not lines:
                    continue
                    
                raw_hotspot_name = lines[0].strip()
                hotspot_name = raw_hotspot_name.replace('_', ' ')
                hotspot_content = section
                
                # Debug logging for Housing_Production specifically
                if 'Housing' in raw_hotspot_name:
                    logger.info(f"DEBUG: Found Housing section - Raw: '{raw_hotspot_name}', Processed: '{hotspot_name}'")
                
                # Extract ALL papers from this hotspot - let LLM decide which ones are useful
                papers_with_data = []
                
                # Find all papers in this section regardless of category
                # Updated regex to handle multi-line paper titles
                paper_pattern = r'\*\*Paper:\*\*\s*(.*?)\s*\(PDF:\s*([^)]+)\)'
                papers = re.findall(paper_pattern, hotspot_content, re.DOTALL)
                
                # Split content by paper sections
                paper_sections = re.split(r'\*\*Paper:\*\*[^\n]+', hotspot_content)
                
                for j, (paper_title, pdf_url) in enumerate(papers):
                    if j + 1 < len(paper_sections):
                        paper_content = paper_sections[j + 1]
                        
                        # Add ALL papers - let LLM decide what's useful
                        paper_data = {
                            'title': paper_title.strip(),
                            'pdf_url': pdf_url.strip(),
                            'content': paper_content.strip(),
                            'has_quantitative_data': True  # Will be filtered by LLM later
                        }
                        
                        
                        papers_with_data.append(paper_data)
                        
                        # Debug logging for Housing_Production specifically
                        if 'Housing' in raw_hotspot_name:
                            logger.info(f"DEBUG: Housing paper found - '{paper_title.strip()[:50]}...'")
                
                hotspots_data[hotspot_name] = {
                    'papers_with_data': papers_with_data,
                    'full_content': hotspot_content
                }
                
                # Debug logging
                logger.info(f"Hotspot '{hotspot_name}': Found {len(papers_with_data)} papers with quantitative data")
            
            logger.info(f"Parsed {len(hotspots_data)} hotspots from existing report")
            return hotspots_data
            
        except Exception as e:
            logger.error(f"Error parsing existing report: {str(e)}")
            raise

    def perform_deep_hotspot_analysis(self, hotspot_name: str, hotspot_data: Dict[str, Any], 
                                    original_ecu_data: str, api_client: APIClient) -> Dict[str, Any]:
        """Perform deep, focused LCA-style analysis for a single hotspot component."""
        try:
            # Get ALL papers - let LLM decide what's useful
            all_papers = hotspot_data['papers_with_data']
            
            if not all_papers:
                logger.warning(f"No papers found for hotspot: {hotspot_name}")
                return None
            
            # First, let LLM decide if web search is needed
            search_decision_prompt = f"""You are analyzing the "{hotspot_name}" component for sustainability improvements.

AVAILABLE RESEARCH PAPERS ({len(all_papers)} papers):
{chr(10).join([f"- {paper['title']}: {paper['content']}..." for paper in all_papers])}

DECISION: Do you need additional specific CALCULATION METHODS, METRICS, or FORMULAS to quantitatively measure sustainability improvements?

CRITICAL ASSESSMENT - Search is SPECIFICALLY needed if papers lack:
1. **CALCULATION FORMULAS**: Missing specific mathematical equations to calculate:
   - GWP formulas (kgCO2eq = emission_factor × activity_data)
   - Energy efficiency calculations (η = useful_output/total_input)
   - Material utilization ratios (yield = output_mass/input_mass)
   - Improvement percentage formulas (% = (new-old)/old × 100)

2. **MEASUREMENT METRICS**: Missing precise units and measurement methods for:
   - Baseline values (current performance numbers)
   - Target values (improvement goals with specific numbers)
   - Conversion factors between units
   - Industry standard benchmarks with numerical thresholds

3. **QUANTIFICATION METHODS**: Missing procedures to calculate:
   - Before/after comparison methodologies
   - Performance improvement quantification
   - Cost-benefit calculation methods
   - ROI formulas for sustainability investments

SEARCH QUERY FOCUS - Target searches for:
- Exact calculation methods and mathematical formulas
- Measurement protocols and quantification standards
- Industry-specific metrics and conversion factors
- Baseline data and benchmark values for comparison

SEARCH QUERY REQUIREMENTS:
- Include terms like: "calculation", "formula", "measurement", "quantification"
- Target specific calculation methods, not general information
- Focus on mathematical approaches to measure improvements
- Max 6-8 words focusing on quantitative methods

CITATION REQUIREMENT: ALL findings must include proper citations with paper titles and PDF links.

Respond with:
SEARCH_NEEDED: [YES/NO]
SEARCH_QUERY: [specific terms targeting calculation methods/formulas, max 8 words]
REASONING: [what specific calculation methods/formulas are missing and needed to quantify improvements]

EXAMPLES OF TARGETED SEARCH QUERIES:
- "aluminum die casting energy efficiency calculation formula"
- "GWP calculation method electronic components manufacturing"
- "material utilization measurement formula injection molding"\""""

            search_response = api_client.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a sustainability measurement specialist who identifies when specific CALCULATION METHODS, FORMULAS, and QUANTIFICATION METRICS are needed for rigorous sustainability analysis. Focus on mathematical approaches to measure improvements, not general environmental information. Search decisions should target: 1) Specific calculation formulas (e.g., GWP equations, efficiency ratios) 2) Measurement protocols and standards 3) Baseline data for comparison 4) Quantitative benchmarks. Only recommend searches for precise mathematical tools needed to calculate and measure sustainability improvements."},
                    {"role": "user", "content": search_decision_prompt}
                ],
                model=api_client.model,
                temperature=0.1,
                max_tokens=300
            )
            
            search_decision = search_response.choices[0].message.content
            additional_data = ""
            
            # Debug: Log the LLM's search decision
            logger.info(f"LLM search decision for {hotspot_name}: {search_decision[:200]}...")
            
            # Initialize search metadata
            search_metadata = None
            additional_data = ""
            
            # Perform web search if needed
            if "SEARCH_NEEDED: YES" in search_decision:
                search_query_match = re.search(r'SEARCH_QUERY:\s*([^\n]+)', search_decision)
                if search_query_match:
                    search_query = search_query_match.group(1).strip().strip('"')  # Remove quotes if present
                    logger.info(f"Performing web search for {hotspot_name}: {search_query}")
                    search_metadata = self._web_search_for_quantitative_data(search_query)
                    additional_data = search_metadata["formatted_content"]
                    
                    # If the primary search fails, try a simpler fallback query
                    if not search_metadata["success"] and search_metadata.get("error_message"):
                        # Create a simpler fallback query based on hotspot name
                        fallback_query = f"{hotspot_name.lower().replace(' ', ' ')} environmental impact"
                        logger.info(f"Primary search failed, trying fallback query: {fallback_query}")
                        fallback_metadata = self._web_search_for_quantitative_data(fallback_query)
                        if fallback_metadata["success"]:
                            search_metadata = fallback_metadata
                            additional_data = fallback_metadata["formatted_content"]
                    
                    # Debug logging
                    if search_metadata["success"]:
                        logger.info(f"✅ Web search SUCCESS for {hotspot_name}: Found {len(search_metadata['search_results'])} results")
                    else:
                        logger.warning(f"❌ Web search failed for {hotspot_name}: {search_metadata.get('error_message', 'Unknown error')}")
                else:
                    logger.warning(f"SEARCH_NEEDED: YES but no search query found for {hotspot_name}")
            else:
                logger.info(f"No web search performed for {hotspot_name} - LLM decided sufficient data available")
            
            # Now perform the deep analysis
            analysis_prompt = f"""You are a senior sustainability researcher conducting a focused, quantitative analysis of the "{hotspot_name}" component.

COMPONENT SPECIFICATION:
{original_ecu_data}

ALL AVAILABLE RESEARCH PAPERS ({len(all_papers)} papers):
{chr(10).join([f"PAPER {i+1}: {paper['title']}\\nPDF: {paper['pdf_url']}\\nCITATION FORMAT: [{paper['title']}] (PDF: {paper['pdf_url']})\\nCONTENT: {paper['content']}\\n{'='*50}" for i, paper in enumerate(all_papers)])}

ADDITIONAL WEB SEARCH DATA:
{additional_data if additional_data else "No web search performed - sufficient information available in research papers"}

CRITICAL INSTRUCTIONS FOR WEB SEARCH DATA:
- Extract ONLY quantitative data with specific numbers from web search results
- Every web search finding MUST be cited as: [Title] (URL: full_url)
- If web search lacks specific numbers, percentages, or metrics, ignore it completely
- Do not use web search for general statements or background information
- Only integrate web data that provides concrete, measurable sustainability metrics

ANALYSIS OBJECTIVE:
Extract and synthesize quantitative sustainability improvements for this specific component. Focus on concrete, measurable impacts that can be implemented. 

IMPORTANT: Review ALL papers provided and identify which ones contain useful quantitative data. Ignore papers that state "No direct relevance", "no material overlap", "not directly applicable", or "No specific quantitative sustainability improvements".

AVAILABLE METRICS (use only what's relevant and supported by data):
- Environmental: GWP (kgCO2eq), energy consumption (kWh, MJ), water usage, waste generation
- Efficiency: Material utilization rates, process efficiency improvements, recycling rates
- Performance: Temperature optimization, pressure reduction, cycle time improvements
- Economic: Cost reductions, material savings, energy cost savings

ANALYSIS APPROACH:
1. Extract ONLY quantitative improvements explicitly stated in research with full citations
2. Focus on implementable solutions with clear numerical impact metrics
3. Skip any section if no concrete data with citations exists
4. Use specific numbers and methods only - no generic statements
5. Organize findings by measurable impact

OUTPUT STRUCTURE (only include sections with concrete citable data):

## {hotspot_name.upper()} ANALYSIS

### Component Overview
[Only if specific environmental data available - state actual component function and quantified concerns]

### Quantitative Findings from Research
[ONLY numerical improvements with full citations - skip if no specific data available]

### Implementation-Ready Solutions
[ONLY solutions with quantitative benefits and full citations - skip if no concrete data]

### Key Metrics and Targets
[ONLY specific, citable targets from research - skip if no quantified data available]

### Critical Gaps
[Only state gaps in quantitative data - skip generic limitations]

MANDATORY CITATION RULES:
- EVERY quantitative finding MUST include citation: [Paper Title] (PDF: [PDF_URL]) OR [Web Source Title] (URL: [FULL_URL])
- ABSOLUTELY FORBIDDEN: "No direct citation available", "X%", "Target: Reduce by X%", placeholder text
- DELETE any statement you cannot cite with a specific source
- If no citable data exists for a section heading, omit that entire section
- Use format: "15% improvement [Source Title] (PDF: [PDF_URL])" with actual numbers only
- ZERO tolerance for uncited statements or placeholder metrics

CRITICAL REQUIREMENTS:
- DELETE statements without citations - do not keep them with disclaimers
- SKIP entire sections if no quantitative citable data exists
- NO generic targets like "Reduce energy consumption by X%" 
- If you cannot provide exact numbers with sources, write nothing for that point
- Shorter analysis with only facts is better than longer analysis with placeholders
- ABSOLUTE RULE: Every number, percentage, improvement claim needs a source citation"""

            # Debug logging to confirm web search data is being passed
            if additional_data and len(additional_data) > 100:  # Check for substantial data
                logger.info(f"Passing {len(additional_data)} characters of web search data to LLM for {hotspot_name} analysis")
                # Check if it contains actual search results
                if "WEB SEARCH RESULT" in additional_data:
                    logger.info(f"Web search data for {hotspot_name} contains {additional_data.count('WEB SEARCH RESULT')} search results")
                else:
                    logger.warning(f"Web search data for {hotspot_name} does not contain expected result format")
            else:
                logger.warning(f"No substantial web search data being passed to LLM for {hotspot_name} (data length: {len(additional_data) if additional_data else 0})")
                if additional_data:
                    logger.info(f"Sample of short web search data: '{additional_data[:200]}...'")
                else:
                    logger.info(f"Web search data is empty or None")

            analysis_response = api_client.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a sustainability researcher who ONLY reports quantitative data with direct citations. ZERO TOLERANCE RULES: 1) NEVER write 'No direct citation available' or 'X%' placeholders 2) DELETE any statement without a citation 3) SKIP entire sections if no quantitative data exists 4) FORBIDDEN: generic targets, template language, or uncited claims 5) Every number/percentage MUST have [Source] (PDF: URL) or [Source] (URL: URL) 6) If you cannot cite a specific source for a claim, do not include that claim at all 7) Shorter analysis with only facts is preferred over longer analysis with placeholders"},
                    {"role": "user", "content": analysis_prompt}
                ],
                model=api_client.model,
                temperature=0.2,
                max_tokens=4000
            )
            
            analysis_content = analysis_response.choices[0].message.content
            
            safe_hotspot_name = safe_str(hotspot_name)
            logger.info(f"Completed deep analysis for hotspot: {safe_hotspot_name}")
            
            return {
                "hotspot_name": hotspot_name,
                "analysis_content": analysis_content,
                "relevant_papers_count": len(all_papers),
                "papers_analyzed": [paper['title'] for paper in all_papers],
                "web_search_performed": "SEARCH_NEEDED: YES" in search_decision,
                "search_metadata": search_metadata if search_metadata else None,
                "search_query": search_metadata["search_query"] if search_metadata else None,
                "search_results": search_metadata["search_results"] if search_metadata else []
            }
            
        except Exception as e:
            logger.error(f"Error in deep hotspot analysis for {hotspot_name}: {str(e)}")
            return None

    def generate_comprehensive_lca_report(self, hotspot_analyses: List[Dict[str, Any]], 
                                        original_report_path: str) -> str:
        """Generate a focused sustainability report based on actual analysis results."""
        try:
            # Read original component data
            with open('ECU_sample.txt', 'r', encoding='utf-8') as f:
                component_data = f.read()
            
            # Build focused report
            report_sections = []
            
            # Title
            report_sections.append("# SUSTAINABILITY ANALYSIS REPORT")
            report_sections.append("")
            
            # Component specification
            report_sections.append("## COMPONENT SPECIFICATION")
            report_sections.append("")
            report_sections.append("```")
            report_sections.append(component_data)
            report_sections.append("```")
            report_sections.append("")
            

            
            # Analysis summary
            total_analyzed = len([analysis for analysis in hotspot_analyses if analysis])
            total_papers = sum(analysis.get('relevant_papers_count', 0) for analysis in hotspot_analyses if analysis)
            
            report_sections.append("## ANALYSIS OVERVIEW")
            report_sections.append("")
            report_sections.append(f"**Components Analyzed:** {total_analyzed}")
            report_sections.append(f"**Research Papers:** {total_papers}")
            report_sections.append(f"**Date:** {datetime.now().strftime('%B %d, %Y')}")
            report_sections.append("")
            report_sections.append("**CITATION NOTE:** All quantitative findings, technologies, and process improvements in this report include proper citations with paper titles and PDF links as required.")
            report_sections.append("")
            
            # Component analyses
            report_sections.append("## COMPONENT ANALYSIS RESULTS")
            report_sections.append("")
            
            for analysis in hotspot_analyses:
                if analysis:
                    report_sections.append(analysis['analysis_content'])
                    report_sections.append("")
                    report_sections.append("---")
                    report_sections.append("")
            
            # Metadata
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            report_sections.append(f"**Report Generated:** {timestamp}")
            report_sections.append("**Analysis Method:** Evidence-based quantitative assessment with mandatory citations")
            
            return "\n".join(report_sections)
            
        except Exception as e:
            logger.error(f"Error generating comprehensive LCA report: {str(e)}")
            raise

    def run_deep_analysis(self, current_report_path: str):
        """Main function to run deep analysis on existing sustainability report."""
        try:
            # Set up output paths
            self.output_folder = str(Path(current_report_path).parent)
            
            # Parse existing report directly (no renaming)
            hotspots_data = self.parse_existing_report(current_report_path)
            logger.info(f"Analyzing existing report: {current_report_path}")
            
            # Read original ECU data
            with open('ECU_sample.txt', 'r', encoding='utf-8') as f:
                original_ecu_data = f.read()
            
            # Perform deep analysis for each hotspot using SEQUENTIAL processing to avoid web search rate limiting
            hotspot_analyses = []
            
            logger.info("Processing hotspots sequentially to ensure reliable web search results...")
            
            for i, (hotspot_name, hotspot_data) in enumerate(hotspots_data.items()):
                try:
                    # Use API clients in round-robin fashion
                    api_client = self.api_clients[i % len(self.api_clients)]
                    
                    logger.info(f"Starting analysis for hotspot {i+1}/{len(hotspots_data)}: {hotspot_name}")
                    
                    result = self.perform_deep_hotspot_analysis(
                        hotspot_name,
                        hotspot_data,
                        original_ecu_data,
                        api_client
                    )
                    
                    if result:
                        hotspot_analyses.append(result)
                        logger.info(f"✅ Completed analysis for: {hotspot_name} ({i+1}/{len(hotspots_data)})")
                    else:
                        logger.warning(f"❌ Analysis failed for: {hotspot_name}")
                    
                    # Add delay between hotspot analyses to prevent API overload
                    if i < len(hotspots_data) - 1:  # Don't delay after the last one
                        time.sleep(3)
                        
                except Exception as e:
                    logger.error(f"Error analyzing hotspot {hotspot_name}: {str(e)}")
            
            # Generate comprehensive LCA report
            comprehensive_report = self.generate_comprehensive_lca_report(hotspot_analyses, current_report_path)
            
            # Save new comprehensive report with unique name
            new_report_path = f"{self.output_folder}/sustainable_solutions_report.txt"
            with open(new_report_path, 'w', encoding='utf-8') as f:
                f.write(comprehensive_report)
            
            # Save search metadata to JSON file
            search_metadata_path = f"{self.output_folder}/web_search_metadata.json"
            self._save_search_metadata(hotspot_analyses, search_metadata_path)
            
            # Generate PDF version
            try:
                pdf_path = f"{self.output_folder}/sustainable_solutions_report.pdf"
                self._generate_professional_pdf(comprehensive_report, pdf_path)
                logger.info(f"Generated PDF report: {pdf_path}")
            except Exception as e:
                logger.warning(f"PDF generation failed: {str(e)}")
            
            logger.info(f"Deep analysis completed successfully!")
            logger.info(f"New sustainability solutions report: {new_report_path}")
            logger.info(f"Search metadata saved to: {search_metadata_path}")
            logger.info(f"Original report preserved at: {current_report_path}")
            
            return new_report_path
            
        except Exception as e:
            logger.error(f"Error in deep analysis: {str(e)}")
            raise

    def _generate_professional_pdf(self, report_content: str, pdf_path: str):
        """Generate a professional PDF report similar to academic papers."""
        try:
            doc = SimpleDocTemplate(
                pdf_path,
                pagesize=A4,
                rightMargin=72,
                leftMargin=72,
                topMargin=72,
                bottomMargin=72
            )
            
            styles = getSampleStyleSheet()
            
            # Custom styles for academic paper format
            styles.add(ParagraphStyle(
                name='AcademicTitle',
                parent=styles['Title'],
                fontSize=16,
                spaceAfter=20,
                alignment=TA_CENTER,
                textColor=HexColor('#000000'),
                fontName='Helvetica-Bold'
            ))
            
            styles.add(ParagraphStyle(
                name='SectionHeader',
                parent=styles['Heading1'],
                fontSize=14,
                spaceAfter=12,
                spaceBefore=18,
                textColor=HexColor('#000000'),
                fontName='Helvetica-Bold'
            ))
            
            styles.add(ParagraphStyle(
                name='SubsectionHeader',
                parent=styles['Heading2'],
                fontSize=12,
                spaceAfter=8,
                spaceBefore=12,
                textColor=HexColor('#000000'),
                fontName='Helvetica-Bold'
            ))
            
            styles.add(ParagraphStyle(
                name='BodyText',
                parent=styles['Normal'],
                fontSize=10,
                spaceAfter=6,
                alignment=TA_JUSTIFY,
                textColor=HexColor('#000000')
            ))
            
            # Build PDF content
            story = []
            
            lines = report_content.split('\n')
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                if line.startswith('# '):
                    story.append(Paragraph(line[2:], styles['AcademicTitle']))
                elif line.startswith('## '):
                    story.append(Paragraph(line[3:], styles['SectionHeader']))
                elif line.startswith('### '):
                    story.append(Paragraph(line[4:], styles['SubsectionHeader']))
                elif line.startswith('```'):
                    continue  # Skip code blocks markers
                else:
                    story.append(Paragraph(line, styles['BodyText']))
            
            doc.build(story)
            
        except Exception as e:
            logger.error(f"PDF generation error: {str(e)}")
            raise

    def _save_search_metadata(self, hotspot_analyses: List[Dict[str, Any]], search_metadata_path: str):
        """Save search metadata to a JSON file."""
        try:
            metadata = []
            for analysis in hotspot_analyses:
                if analysis['web_search_performed']:
                    metadata.append({
                        "hotspot_name": analysis['hotspot_name'],
                        "search_query": analysis['search_query'],
                        "search_results": analysis['search_results'],
                        "relevant_papers_count": analysis['relevant_papers_count'],
                        "papers_analyzed": analysis['papers_analyzed'],
                        "web_search_performed": analysis['web_search_performed'],
                        "search_metadata": analysis['search_metadata']
                    })
            
            with open(search_metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f)
            
        except Exception as e:
            logger.error(f"Error saving search metadata: {str(e)}")
            raise

def main():
    """Run the deep hotspot analysis."""
    from config import PRIMARY_API_KEY, SECONDARY_API_KEY, BASE_URL
    
    # API configurations
    api_configs = [
        {"api_key": PRIMARY_API_KEY, "base_url": BASE_URL, "model": "llama-3.3-70b-instruct"},
        {"api_key": SECONDARY_API_KEY, "base_url": BASE_URL, "model": "llama-3.3-70b-instruct"}
    ]
    
    # Tavily API key for web search
    TAVILY_API_KEY = "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"
    
    analyzer = DeepHotspotAnalyzer(api_configs, TAVILY_API_KEY)
    
    # Run analysis on existing report
    current_report = "output/automotive_sample_input/sustainable_solutions_report.txt"
    if Path(current_report).exists():
        result = analyzer.run_deep_analysis(current_report)
        print(f"Deep analysis completed. New report: {result}")
    else:
        print(f"Report file not found: {current_report}")

if __name__ == "__main__":
    main()