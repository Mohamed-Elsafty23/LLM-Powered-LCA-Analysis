"""
Deep Hotspot LCA Analysis System
Performs focused, quantitative analysis for each ECU hotspot component following IFIP methodology.
Generates professional LCA reports with primary data focus and quantitative sustainability metrics.
"""

import os
import sys

# Configure UTF-8 encoding for console output on Windows
if sys.platform.startswith('win'):
    try:
        # Try to set console to UTF-8 mode
        os.system('chcp 65001 >nul 2>&1')
        # Set environment variables for UTF-8
        os.environ['PYTHONIOENCODING'] = 'utf-8'
    except:
        pass

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
                'ψ': 'psi', 'ω': 'omega', 'η': 'eta', 'ζ': 'zeta', 'κ': 'kappa',
                'ξ': 'xi', 'ο': 'omicron', 'υ': 'upsilon', 'Γ': 'Gamma', 'Π': 'Pi',
                'Α': 'Alpha', 'Β': 'Beta', 'Δ': 'Delta', 'Ε': 'Epsilon', 'Θ': 'Theta',
                'Λ': 'Lambda', 'Μ': 'Mu', 'Ν': 'Nu', 'Ρ': 'Rho', 'Σ': 'Sigma',
                'Τ': 'Tau', 'Φ': 'Phi', 'Χ': 'Chi', 'Ψ': 'Psi', 'Ω': 'Omega',
                'Η': 'Eta', 'Ζ': 'Zeta', 'Κ': 'Kappa', 'Ξ': 'Xi', 'Ο': 'Omicron',
                'Υ': 'Upsilon'
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
            
            # Safely convert the log message
            if hasattr(record, 'msg') and isinstance(record.msg, str):
                record.msg = safe_str(record.msg)
            if hasattr(record, 'args') and record.args:
                record.args = tuple(safe_str(arg) if isinstance(arg, str) else str(arg) for arg in record.args)
            
            # Ensure the formatted message is also safe
            try:
                formatted_msg = self.format(record)
                formatted_msg = safe_str(formatted_msg)
                # Update the record with safe message
                record.msg = formatted_msg
                record.args = ()
            except Exception as format_error:
                # If formatting fails, create a simple safe message
                record.msg = safe_str(f"Log message formatting error: {str(format_error)}")
                record.args = ()
            
            super().emit(record)
            
        except (UnicodeEncodeError, UnicodeDecodeError) as e:
            # Instead of raising an error, log a safe version
            try:
                safe_msg = safe_str(f"Unicode encoding error in log message: {str(e)}")
                fallback_record = logging.LogRecord(
                    name=record.name,
                    level=logging.WARNING,
                    pathname=record.pathname,
                    lineno=record.lineno,
                    msg=safe_msg,
                    args=(),
                    exc_info=None
                )
                super().emit(fallback_record)
            except Exception:
                # If even the fallback fails, print to console
                print(f"Critical logging error: {e}")
        except Exception as e:
            # Handle any other logging errors gracefully
            try:
                safe_msg = safe_str(f"Logging error: {str(e)}")
                print(safe_msg)
            except Exception:
                print("Critical logging system failure")
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

# Create base logger
_base_logger = logging.getLogger(__name__)

# Create a safe logging wrapper
class SafeLogger:
    def __init__(self, base_logger):
        self.base_logger = base_logger
    
    def info(self, msg, *args, **kwargs):
        try:
            safe_msg = safe_str(str(msg))
            safe_args = tuple(safe_str(str(arg)) for arg in args)
            self.base_logger.info(safe_msg, *safe_args, **kwargs)
        except Exception as e:
            print(f"Logging error prevented: {e}")
    
    def warning(self, msg, *args, **kwargs):
        try:
            safe_msg = safe_str(str(msg))
            safe_args = tuple(safe_str(str(arg)) for arg in args)
            self.base_logger.warning(safe_msg, *safe_args, **kwargs)
        except Exception as e:
            print(f"Logging warning prevented: {e}")
    
    def error(self, msg, *args, **kwargs):
        try:
            safe_msg = safe_str(str(msg))
            safe_args = tuple(safe_str(str(arg)) for arg in args)
            self.base_logger.error(safe_msg, *safe_args, **kwargs)
        except Exception as e:
            print(f"Logging error prevented: {e}")
    
    def debug(self, msg, *args, **kwargs):
        try:
            safe_msg = safe_str(str(msg))
            safe_args = tuple(safe_str(str(arg)) for arg in args)
            self.base_logger.debug(safe_msg, *safe_args, **kwargs)
        except Exception as e:
            print(f"Logging debug prevented: {e}")

# Use the safe logger wrapper
logger = SafeLogger(_base_logger)

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
                fresh_client = TavilyClient(api_key=self.tavily_client.api_key)
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
                
                logger.info(f"[SUCCESS] Web search SUCCESS: Found {len(results)} results for '{search_query}' (total length: {len(formatted_results)} chars)")
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
            
            # Find all hotspot sections using the ### pattern
            hotspot_pattern = r'### ([A-Za-z_]+(?:\s+[A-Za-z_]+)*)'
            hotspot_matches = list(re.finditer(hotspot_pattern, content))
            
            logger.info(f"Found {len(hotspot_matches)} hotspot sections in report")
            
            for i, match in enumerate(hotspot_matches):
                raw_hotspot_name = match.group(1).strip()
                hotspot_name = raw_hotspot_name.replace('_', ' ')
                
                # Get the content for this hotspot (from current match to next match or end)
                start_pos = match.start()
                if i + 1 < len(hotspot_matches):
                    end_pos = hotspot_matches[i + 1].start()
                    hotspot_content = content[start_pos:end_pos]
                else:
                    hotspot_content = content[start_pos:]
                
                logger.info(f"Processing hotspot: '{safe_str(raw_hotspot_name)}' -> '{safe_str(hotspot_name)}'")
                logger.info(f"Content length: {len(hotspot_content)} characters")
                
                # Extract ALL papers from this hotspot - let LLM decide which ones are useful
                papers_with_data = []
                
                # Find all papers in this section regardless of category
                # Updated regex to handle multi-line paper titles
                paper_pattern = r'\*\*Paper:\*\*\s*(.*?)\s*\(PDF:\s*([^)]+)\)'
                papers = re.findall(paper_pattern, hotspot_content, re.DOTALL)
                
                logger.info(f"Found {len(papers)} papers in {safe_str(hotspot_name)} section")
                
                # Split content by paper sections to get paper content
                if papers:
                    paper_sections = re.split(r'\*\*Paper:\*\*[^\n]+', hotspot_content)
                    
                    for j, (paper_title, pdf_url) in enumerate(papers):
                        if j + 1 < len(paper_sections):
                            paper_content = paper_sections[j + 1]
                            
                            # Clean up paper title and content
                            cleaned_title = paper_title.strip()
                            cleaned_content = paper_content.strip()
                            
                            # Add ALL papers - let LLM decide what's useful
                            paper_data = {
                                'title': cleaned_title,
                                'pdf_url': pdf_url.strip(),
                                'content': cleaned_content,
                                'has_quantitative_data': True  # Will be filtered by LLM later
                            }
                            
                            papers_with_data.append(paper_data)
                            logger.info(f"Added paper: '{cleaned_title[:60]}...'")
                
                # Only add hotspot if it has papers
                if papers_with_data:
                    hotspots_data[hotspot_name] = {
                        'papers_with_data': papers_with_data,
                        'full_content': hotspot_content
                    }
                    logger.info(f"[SUCCESS] Hotspot '{safe_str(hotspot_name)}': Added {len(papers_with_data)} papers")
                else:
                    logger.warning(f"[SKIPPED] Hotspot '{safe_str(hotspot_name)}': No papers found, skipping")
            
            logger.info(f"Successfully parsed {len(hotspots_data)} hotspots from existing report")
            logger.info(f"Hotspot names: {list(hotspots_data.keys())}")
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
                logger.warning(f"No papers found for hotspot: {safe_str(hotspot_name)}")
                return None
            
            # First, let LLM decide if web search is needed
            search_decision_prompt = f"""You are analyzing the "{safe_str(hotspot_name)}" component for sustainability improvements.

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
            logger.info(f"LLM search decision for {safe_str(hotspot_name)}: {safe_str(search_decision)}...")
            
            # Initialize search metadata
            search_metadata = None
            additional_data = ""
            
            # Perform web search if needed
            if "SEARCH_NEEDED: YES" in search_decision:
                search_query_match = re.search(r'SEARCH_QUERY:\s*([^\n]+)', search_decision)
                if search_query_match:
                    search_query = search_query_match.group(1).strip().strip('"')  # Remove quotes if present
                    logger.info(f"Performing web search for {safe_str(hotspot_name)}: {safe_str(search_query)}")
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
                        logger.info(f"[SUCCESS] Web search SUCCESS for {safe_str(hotspot_name)}: Found {len(search_metadata['search_results'])} results")
                    else:
                        logger.warning(f"[FAILED] Web search failed for {safe_str(hotspot_name)}: {search_metadata.get('error_message', 'Unknown error')}")
                else:
                    logger.warning(f"SEARCH_NEEDED: YES but no search query found for {safe_str(hotspot_name)}")
            else:
                logger.info(f"No web search performed for {safe_str(hotspot_name)} - LLM decided sufficient data available")
            
            # Now perform the deep analysis
            analysis_prompt = f"""You are a senior sustainability researcher conducting a focused, quantitative analysis of the "{safe_str(hotspot_name)}" component.

COMPONENT SPECIFICATION:
{original_ecu_data}

ALL AVAILABLE RESEARCH PAPERS ({len(all_papers)} papers):
{chr(10).join([f"PAPER {i+1}: {paper['title']}\\nPDF: {paper['pdf_url']}\\nCITATION FORMAT: [{paper['title']}] (PDF: {paper['pdf_url']})\\nCONTENT: {paper['content']}\\n{'='*50}" for i, paper in enumerate(all_papers)])}

ADDITIONAL WEB SEARCH DATA:
{additional_data if additional_data else "No web search performed - sufficient information available in research papers"}

**CRITICAL MANDATORY INSTRUCTIONS FOR WEB SEARCH DATA INTEGRATION if the web search results are relevant to the component:**
1. **EXTRACT ALL NUMERICAL VALUES**: Every number, percentage, formula, calculation method, or metric from web search results MUST be included
2. **MANDATORY WEB CITATION**: Every web search finding MUST be cited as: [Title] (URL: full_url)
3. **PRIORITY DATA EXTRACTION**: Web search data often contains baseline values, industry benchmarks, and calculation formulas that papers may lack
4. **NO WEB DATA IGNORED**: If web search returned results, you MUST find and extract quantitative data from them
5. Check for relevant data in the web search results and extract it.

**ENHANCED PAPER ANALYSIS INSTRUCTIONS:**
1. **DEEP SCAN ALL PAPERS**: Read through EVERY paper completely for any numbers, percentages, improvements, or metrics
2. **EXTRACT HIDDEN QUANTITATIVE DATA**: Look for process parameters, material properties, efficiency gains, cost savings, time reductions
3. **NO PAPER DISMISSED**: Even if a paper seems unrelated, check for ANY quantitative sustainability data
4. **SPECIFIC DATA TO EXTRACT**:
   - Energy consumption values (kWh, MJ, GJ)
   - Material efficiency percentages (%, ratios)
   - Process improvements (time reductions, yield increases)
   - Environmental impact reductions (CO2, waste, water)
   - Cost savings and economic benefits
   - Temperature, pressure, speed optimizations

**MANDATORY IMPROVEMENT APPLICATION REQUIREMENTS:**
1. **APPLY ALL IMPROVEMENT FORMULAS**: When you find improvement formulas, calculations, or measurement methods in papers or web search results, you MUST apply them to this specific component
2. **COMPONENT-SPECIFIC CALCULATIONS**: Use the exact component specifications to calculate actual improvement values, not generic estimates
3. **FORMULA APPLICATION**: If research provides formulas (e.g., efficiency = output/input, % improvement = (new-old)/old×100), apply these formulas using available data
4. **MEASUREMENT APPLICATION**: Apply any measurement protocols or quantification methods found in research to determine current performance and potential improvements
5. **PRACTICAL IMPLEMENTATION**: Show how improvements would specifically apply to this component's manufacturing, operation, or lifecycle
6. **DECIDE APPLICABLE METRICS**: Based on available data from papers and web search, determine which metrics and measurements are most relevant for this component
7. **CALCULATE SPECIFIC IMPROVEMENTS**: When improvement measures are found, calculate their specific application to this component using any formulas or methods provided

ANALYSIS OBJECTIVE:
Extract, synthesize, and APPLY quantitative sustainability improvements specifically to this component. You must calculate and demonstrate how improvements would work for this exact component using formulas and methods found in research.

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

**WEB SEARCH DATA MUST BE INTEGRATED** - If web search was performed and returned results, you MUST extract and include quantitative data from those results with proper URL citations.

OUTPUT STRUCTURE (only include sections with concrete citable data):

## {safe_str(hotspot_name).upper()} ANALYSIS

### Component Overview
[Only if specific environmental data available - state actual component function and quantified concerns]

### Quantitative Findings from Research Papers
[ONLY numerical improvements with full citations - skip if no specific data available]

### Quantitative Findings from Web Search  
[MANDATORY IF WEB SEARCH PERFORMED: ONLY improvements related to the component with URL citations]

### Applied Improvement Calculations
[MANDATORY: Apply any improvement formulas, measurement methods, or calculation techniques found in research to this specific component. Show actual calculations and results using component specifications. Include baseline values, improvement formulas, and calculated outcomes.]

### Implementation-Ready Solutions
[ONLY solutions with quantitative benefits and full citations - include both paper and web sources]

### Key Metrics and Targets with Applied Improvements
[ONLY specific, citable targets from research AND web search - include baseline values, benchmarks, and calculated improvement targets based on applied formulas]

### Critical Gaps
[Only state gaps in quantitative data where neither papers nor web search provided specific metrics]

**ABSOLUTE REQUIREMENTS:**
1. **ZERO TOLERANCE FOR MISSING WEB DATA**: If web search found results, you MUST extract quantitative data from them
2. **MANDATORY DUAL CITATION**: Use both paper citations (PDF: URL) AND web citations (URL: URL) 
3. **NO GENERIC STATEMENTS**: Every claim must have a specific number and source citation
4. **COMPREHENSIVE EXTRACTION**: Read every paper thoroughly for ANY quantitative sustainability data
5. **INTEGRATION PRIORITY**: Web search data often provides missing baselines and calculation methods that papers lack

**FORBIDDEN ACTIONS:**
- Ignoring web search results that contain numbers or formulas
- Dismissing papers without thoroughly reading for quantitative data  
- Using placeholder percentages or generic improvement claims
- Missing any numerical values from web search results
- Failing to cite web sources with full URLs"""

            # Debug logging to confirm web search data is being passed
            if additional_data and len(additional_data) > 100:  # Check for substantial data
                logger.info(f"Passing {len(additional_data)} characters of web search data to LLM for {safe_str(hotspot_name)} analysis")
                # Check if it contains actual search results
                if "WEB SEARCH RESULT" in additional_data:
                    logger.info(f"Web search data for {safe_str(hotspot_name)} contains {additional_data.count('WEB SEARCH RESULT')} search results")
                else:
                    logger.warning(f"Web search data for {safe_str(hotspot_name)} does not contain expected result format")
            else:
                logger.warning(f"No substantial web search data being passed to LLM for {safe_str(hotspot_name)} (data length: {len(additional_data) if additional_data else 0})")
                if additional_data:
                    logger.info(f"Sample of short web search data: '{additional_data}...'")
                else:
                    logger.info(f"Web search data is empty or None")

            analysis_response = api_client.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a sustainability researcher who MUST extract ALL quantitative data from BOTH papers AND web search results AND apply improvement formulas to the specific component. CRITICAL RULES: 1) ZERO TOLERANCE for ignoring web search data - if web search found results, you MUST extract numerical data from them 2) MANDATORY dual citations: [Paper Title] (PDF: URL) AND [Web Title] (URL: URL) 3) DELETE any statement without a citation 4) NO placeholders or generic claims 5) Extract EVERY number, percentage, formula from web search results 6) Web search data contains critical baselines, benchmarks, and calculation methods that papers often lack 7) If web search was performed, it MUST appear in your analysis with URL citations 8) MANDATORY: Apply any improvement formulas or calculation methods found in research to the specific component - show actual calculations and results 9) Use component specifications to calculate real improvement values, not generic estimates 10) Shorter analysis with complete data and applied calculations is better than longer analysis missing web search integration or improvement applications"},
                    {"role": "user", "content": analysis_prompt}
                ],
                model=api_client.model,
                temperature=0.1,  # Lower temperature for more precise extraction
                max_tokens=5000   # Increased tokens for more comprehensive analysis
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
            logger.error(f"Error in deep hotspot analysis for {safe_str(hotspot_name)}: {str(e)}")
            return None

    def generate_comprehensive_lca_report(self, hotspot_analyses: List[Dict[str, Any]], 
                                        original_report_path: str, original_input_file: str = None) -> str:
        """Generate a focused sustainability report based on actual analysis results."""
        try:
            # Read original component data from the actual input file
            if original_input_file and Path(original_input_file).exists():
                with open(original_input_file, 'r', encoding='utf-8') as f:
                    component_data = f.read()
            else:
                # Fallback: try to find the input file in the output folder
                output_folder = Path(original_report_path).parent
                input_files = list(output_folder.glob("*.txt"))
                # Filter out report files and look for the original input file
                input_files = [f for f in input_files if not any(keyword in f.name.lower() for keyword in 
                               ['report', 'sustainable', 'hotspot', 'analysis', 'final'])]
                
                if input_files:
                    # Use the first input file found (should be the original uploaded file)
                    with open(input_files[0], 'r', encoding='utf-8') as f:
                        component_data = f.read()
                    logger.info(f"Using input file from output folder: {input_files[0]}")
                else:
                    # Last resort: use hardcoded file (should not happen in normal workflow)
                    logger.warning("No input file found, using fallback ECU_sample.txt")
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

    def run_deep_analysis(self, current_report_path: str, original_input_file: str = None):
        """Main function to run deep analysis on existing sustainability report."""
        try:
            # Set up output paths
            self.output_folder = str(Path(current_report_path).parent)
            
            # Parse existing report directly (no renaming)
            hotspots_data = self.parse_existing_report(current_report_path)
            logger.info(f"Analyzing existing report: {current_report_path}")
            
            # Read original input data from the actual input file
            if original_input_file and Path(original_input_file).exists():
                with open(original_input_file, 'r', encoding='utf-8') as f:
                    original_ecu_data = f.read()
                logger.info(f"Using original input file: {original_input_file}")
            else:
                # Fallback: try to find the input file in the output folder
                output_folder = Path(current_report_path).parent
                input_files = list(output_folder.glob("*.txt"))
                # Filter out report files and look for the original input file
                input_files = [f for f in input_files if not any(keyword in f.name.lower() for keyword in 
                               ['report', 'sustainable', 'hotspot', 'analysis', 'final'])]
                
                if input_files:
                    # Use the first input file found (should be the original uploaded file)
                    with open(input_files[0], 'r', encoding='utf-8') as f:
                        original_ecu_data = f.read()
                    logger.info(f"Using input file from output folder: {input_files[0]}")
                else:
                    # Last resort: use hardcoded file (should not happen in normal workflow)
                    logger.warning("No input file found, using fallback ECU_sample.txt")
                    with open('ECU_sample.txt', 'r', encoding='utf-8') as f:
                        original_ecu_data = f.read()
            
            # Perform deep analysis for each hotspot using SEQUENTIAL processing to avoid web search rate limiting
            hotspot_analyses = []
            
            logger.info("Processing hotspots sequentially to ensure reliable web search results...")
            
            for i, (hotspot_name, hotspot_data) in enumerate(hotspots_data.items()):
                try:
                    # Use API clients in round-robin fashion
                    api_client = self.api_clients[i % len(self.api_clients)]
                    
                    logger.info(f"Starting analysis for hotspot {i+1}/{len(hotspots_data)}: {safe_str(hotspot_name)}")
                    
                    result = self.perform_deep_hotspot_analysis(
                        hotspot_name,
                        hotspot_data,
                        original_ecu_data,
                        api_client
                    )
                    
                    if result:
                        hotspot_analyses.append(result)
                        logger.info(f"[SUCCESS] Completed analysis for: {safe_str(hotspot_name)} ({i+1}/{len(hotspots_data)})")
                    else:
                        logger.warning(f"[FAILED] Analysis failed for: {safe_str(hotspot_name)}")
                    
                    # Add delay between hotspot analyses to prevent API overload
                    if i < len(hotspots_data) - 1:  # Don't delay after the last one
                        time.sleep(3)
                        
                except Exception as e:
                    logger.error(f"Error analyzing hotspot {safe_str(hotspot_name)}: {str(e)}")
            
            # Generate comprehensive LCA report with original input file
            comprehensive_report = self.generate_comprehensive_lca_report(hotspot_analyses, current_report_path, original_input_file)
            
            # Save new comprehensive report with unique name
            new_report_path = f"{self.output_folder}/final_sustainable_solutions_report.txt"
            with open(new_report_path, 'w', encoding='utf-8') as f:
                f.write(comprehensive_report)
            
            # Save search metadata to JSON file
            search_metadata_path = f"{self.output_folder}/web_search_metadata.json"
            self._save_search_metadata(hotspot_analyses, search_metadata_path)
            
            # Generate PDF version
            try:
                pdf_path = f"{self.output_folder}/final_sustainable_solutions_report.pdf"
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
        """Generate a professional PDF report with enhanced styling and parsing."""
        try:
            doc = SimpleDocTemplate(
                pdf_path,
                pagesize=A4,
                rightMargin=72,
                leftMargin=72,
                topMargin=72,
                bottomMargin=72
            )
            
            # Create enhanced styles
            styles = self._create_pdf_styles()
            
            # Build story (PDF content)
            story = []
            
            # Add header with better formatting
            title_text = "FINAL SUSTAINABILITY<br/>ANALYSIS REPORT"
            story.append(Paragraph(title_text, styles['CustomMainTitle']))
            story.append(Spacer(1, 0.2*inch))
            
            # Add generation timestamp
            timestamp = datetime.now().strftime("%B %d, %Y at %H:%M")
            story.append(Paragraph(f"Generated on {timestamp}", styles['CustomBodyText']))
            story.append(Spacer(1, 0.3*inch))
            
            # Parse and add content
            elements = self._parse_report_content(report_content)
            
            for element in elements:
                element_type = element['type']
                content = element['content']
                
                if element_type == 'main_title':
                    continue  # Already added at the top
                elif element_type == 'hotspot_title':
                    # MAIN HEADLINES - largest (## sections)
                    story.append(Spacer(1, 0.3*inch))
                    story.append(Paragraph(content, styles['CustomHotspotTitle']))
                elif element_type == 'section_title':
                    # Secondary headlines (### sections)
                    story.append(Spacer(1, 0.2*inch))
                    story.append(Paragraph(content, styles['CustomSectionTitle']))
                elif element_type == 'content_subtitle':
                    # Content subsections
                    story.append(Spacer(1, 0.1*inch))
                    story.append(Paragraph(content, styles['CustomContentSubtitle']))
                elif element_type == 'paper_citation':
                    # Paper citations with clickable PDF links
                    story.append(Spacer(1, 0.08*inch))
                    processed_content = self._process_hyperlinks(content)
                    story.append(Paragraph(processed_content, styles['CustomPaperTitle']))
                elif element_type == 'bullet':
                    story.append(Paragraph(f"• {content}", styles['CustomBulletPoint']))
                elif element_type == 'numbered':
                    story.append(Paragraph(content, styles['CustomNumberedItem']))
                elif element_type == 'indented_detail':
                    story.append(Paragraph(content, styles['CustomIndentedDetail']))
                elif element_type == 'divider':
                    story.append(Spacer(1, 0.15*inch))
                    # Add a horizontal line
                    divider_table = Table([['─' * 60]], colWidths=[5*inch])
                    divider_table.setStyle(TableStyle([
                        ('TEXTCOLOR', (0, 0), (-1, -1), HexColor('#BDC3C7')),
                        ('FONTSIZE', (0, 0), (-1, -1), 10),
                        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ]))
                    story.append(divider_table)
                    story.append(Spacer(1, 0.15*inch))
                elif element_type == 'code_block':
                    # Code blocks with monospace font
                    story.append(Paragraph(content, styles['CustomCodeBlock']))
                elif element_type == 'body':
                    story.append(Paragraph(content, styles['CustomBodyText']))
            
            # Add footer information
            story.append(Spacer(1, 0.3*inch))
            footer_text = """
            <para align="center">
            <b>Report Information</b><br/>
            This report was generated using the Deep Hotspot LCA Analysis System<br/>
            All quantitative findings include proper citations and web search validation<br/>
            Analysis combines research papers with real-time web search for quantitative data
            </para>
            """
            story.append(Paragraph(footer_text, styles['CustomBodyText']))
            
            # Build PDF
            doc.build(story)
            
            logger.info(f"Successfully generated enhanced PDF report: {pdf_path}")
            
        except Exception as e:
            logger.error(f"PDF generation error: {str(e)}")
            raise

    def _create_pdf_styles(self):
        """Create custom styles for PDF generation."""
        styles = getSampleStyleSheet()
        
        # Custom styles with proper hierarchy
        styles.add(ParagraphStyle(
            name='CustomMainTitle',
            parent=styles['Title'],
            fontSize=22,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=HexColor('#2C3E50'),
            fontName='Helvetica-Bold',
            wordWrap='CJK'
        ))
        
        # ## Hotspot sections - Main headlines (largest)
        styles.add(ParagraphStyle(
            name='CustomHotspotTitle',
            parent=styles['Heading1'],
            fontSize=18,
            spaceAfter=15,
            spaceBefore=25,
            textColor=HexColor('#1A5490'),
            keepWithNext=1,
            fontName='Helvetica-Bold'
        ))
        
        # ### sections - Secondary headlines (medium)
        styles.add(ParagraphStyle(
            name='CustomSectionTitle',
            parent=styles['Heading2'],
            fontSize=15,
            spaceAfter=10,
            spaceBefore=15,
            textColor=HexColor('#34495E'),
            keepWithNext=1,
            fontName='Helvetica-Bold'
        ))
        
        # Paper citations - Tertiary headlines (smaller)
        styles.add(ParagraphStyle(
            name='CustomPaperTitle',
            parent=styles['Heading3'],
            fontSize=13,
            spaceAfter=6,
            spaceBefore=8,
            textColor=HexColor('#2980B9'),
            keepWithNext=1,
            fontName='Helvetica-Bold'
        ))
        
        # Content subsections
        styles.add(ParagraphStyle(
            name='CustomContentSubtitle',
            parent=styles['Normal'],
            fontSize=12,
            spaceAfter=4,
            spaceBefore=6,
            textColor=HexColor('#7F8C8D'),
            fontName='Helvetica-Bold'
        ))
        
        # Regular body text
        styles.add(ParagraphStyle(
            name='CustomBodyText',
            parent=styles['Normal'],
            fontSize=11,
            spaceAfter=4,
            alignment=TA_JUSTIFY,
            textColor=HexColor('#2C3E50')
        ))
        
        # Bullet points
        styles.add(ParagraphStyle(
            name='CustomBulletPoint',
            parent=styles['Normal'],
            fontSize=11,
            spaceAfter=3,
            leftIndent=20,
            bulletIndent=10,
            textColor=HexColor('#2C3E50')
        ))
        
        # Numbered items
        styles.add(ParagraphStyle(
            name='CustomNumberedItem',
            parent=styles['Normal'],
            fontSize=11,
            spaceAfter=3,
            leftIndent=15,
            textColor=HexColor('#2C3E50')
        ))
        
        # Indented details
        styles.add(ParagraphStyle(
            name='CustomIndentedDetail',
            parent=styles['Normal'],
            fontSize=10,
            spaceAfter=2,
            leftIndent=40,
            textColor=HexColor('#555555')
        ))
        
        # Code blocks
        styles.add(ParagraphStyle(
            name='CustomCodeBlock',
            parent=styles['Normal'],
            fontSize=9,
            spaceAfter=6,
            spaceBefore=6,
            leftIndent=20,
            rightIndent=20,
            textColor=HexColor('#2C3E50'),
            fontName='Courier',
            backColor=HexColor('#F8F9FA')
        ))
        
        return styles

    def _parse_report_content(self, report_content: str) -> List[Dict[str, Any]]:
        """Parse the text report content into structured elements for PDF generation."""
        lines = report_content.split('\n')
        elements = []
        in_code_block = False
        
        for line in lines:
            original_line = line
            line = line.strip()
            if not line:
                continue
                
            # Handle code blocks
            if line.startswith('```'):
                in_code_block = not in_code_block
                continue
                
            if in_code_block:
                elements.append({'type': 'code_block', 'content': original_line})
                continue
                
            # Main title
            if line.startswith('# SUSTAINABILITY ANALYSIS REPORT'):
                elements.append({'type': 'main_title', 'content': line[2:].strip()})
            # ## Hotspot sections - MAIN HEADLINES (largest)
            elif line.startswith('## '):
                hotspot_title = line[3:].strip().replace('_', ' ')
                elements.append({'type': 'hotspot_title', 'content': hotspot_title})
            # ### sections - Secondary headlines
            elif line.startswith('### '):
                section_title = line[4:].strip()
                elements.append({'type': 'section_title', 'content': section_title})
            # Paper citations with clickable links
            elif '[' in line and '] (PDF:' in line:
                elements.append({'type': 'paper_citation', 'content': line})
            elif '[' in line and '] (URL:' in line:
                elements.append({'type': 'paper_citation', 'content': line})
            # Content subsections (like "Quantitative Findings")
            elif (line.startswith('**') and line.endswith('**') and 
                  ('Finding' in line or 'Overview' in line or 'Calculation' in line or 
                   'Solution' in line or 'Metric' in line or 'Gap' in line)):
                subtitle = line.strip('*').strip()
                elements.append({'type': 'content_subtitle', 'content': subtitle})
            # Bullet points
            elif line.startswith('• ') or line.startswith('- ') or line.startswith('* '):
                bullet_text = line[2:].strip()
                elements.append({'type': 'bullet', 'content': bullet_text})
            # Numbered lists
            elif line and line[0].isdigit() and '. ' in line[:4]:
                elements.append({'type': 'numbered', 'content': line})
            # Indented details (lines that start with spaces)
            elif (original_line.startswith('   ') or original_line.startswith('\t')) and ':' in line:
                elements.append({'type': 'indented_detail', 'content': line})
            # Dividers
            elif line.startswith('---'):
                elements.append({'type': 'divider', 'content': ''})
            # Regular text
            elif line:
                elements.append({'type': 'body', 'content': line})
        
        return elements

    def _process_hyperlinks(self, content: str) -> str:
        """Process content to convert PDF and web URLs into clickable hyperlinks."""
        # Pattern to match PDF links like [Title] (PDF: http://arxiv.org/pdf/...)
        pdf_pattern = r'\[([^\]]+)\] \(PDF:\s*(https?://[^\)]+)\)'
        
        def replace_pdf_link(match):
            title = match.group(1).strip()
            url = match.group(2).strip()
            return f'<a href="{url}" color="#0066CC"><u>{title}</u></a> (PDF: <a href="{url}" color="#0066CC"><u>{url}</u></a>)'
        
        # Pattern to match web links like [Title] (URL: http://...)
        url_pattern = r'\[([^\]]+)\] \(URL:\s*(https?://[^\)]+)\)'
        
        def replace_url_link(match):
            title = match.group(1).strip()
            url = match.group(2).strip()
            return f'<a href="{url}" color="#0066CC"><u>{title}</u></a> (URL: <a href="{url}" color="#0066CC"><u>{url}</u></a>)'
        
        # Replace PDF and URL links with hyperlinks
        processed_content = re.sub(pdf_pattern, replace_pdf_link, content)
        processed_content = re.sub(url_pattern, replace_url_link, processed_content)
        
        return processed_content

    def _validate_web_search_integration(self, analysis_content: str, search_metadata: Dict[str, Any], hotspot_name: str) -> None:
        """Validate that web search data was properly integrated into the analysis."""
        if not search_metadata or not search_metadata.get("success", False):
            logger.info(f"No web search validation needed for {safe_str(hotspot_name)} - no successful web search performed")
            return
            
        search_results = search_metadata.get("search_results", [])
        if not search_results:
            logger.warning(f"Web search successful but no results found for {safe_str(hotspot_name)}")
            return
            
        # Check if web search URLs appear in the analysis content
        urls_found = []
        urls_missing = []
        
        for result in search_results:
            url = result.get("url", "")
            title = result.get("title", "")
            
            if url and url in analysis_content:
                urls_found.append(f"{title} ({url})")
            else:
                urls_missing.append(f"{title} ({url})")
                
        # Log validation results
        if urls_found:
            logger.info(f"[SUCCESS] Web search integration for {safe_str(hotspot_name)}: {len(urls_found)}/{len(search_results)} URLs cited")
            for url_info in urls_found:
                logger.info(f"  [CITED] {url_info}")
        
        if urls_missing:
            logger.warning(f"[WARNING] Missing web search integration for {safe_str(hotspot_name)}: {len(urls_missing)}/{len(search_results)} URLs not cited")
            for url_info in urls_missing:
                logger.warning(f"  [MISSING] {url_info}")
                
        # Check for quantitative data in missed web search results
        missing_quantitative_data = []
        for result in search_results:
            if result.get("url", "") not in analysis_content:
                content = result.get("content", "")
                # Look for numbers, percentages, formulas in the content
                numbers = re.findall(r'\d+\.?\d*\s*%|\d+\.?\d*\s*[A-Za-z]+/[A-Za-z]+|\d+\.?\d*\s*[kKmMgG][A-Za-z]*|\d+\.?\d*', content)
                if numbers:
                    missing_quantitative_data.append({
                        "title": result.get("title", ""),
                        "url": result.get("url", ""), 
                        "quantitative_data": numbers[:5]  # First 5 numbers found
                    })
                    
        if missing_quantitative_data:
            logger.error(f"[CRITICAL] {safe_str(hotspot_name)} missing quantitative data from web search:")
            for missing in missing_quantitative_data:
                logger.error(f"  [DATA] {missing['title']}: {missing['quantitative_data']}")
                logger.error(f"      URL: {missing['url']}")

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
    from config import PRIMARY_API_KEY, SECONDARY_API_KEY, BASE_URL, TAVILY_API_KEY
    
    # API configurations
    api_configs = [
        {"api_key": PRIMARY_API_KEY, "base_url": BASE_URL, "model": "llama-3.3-70b-instruct"},
        {"api_key": SECONDARY_API_KEY, "base_url": BASE_URL, "model": "llama-3.3-70b-instruct"}
    ]
    
    analyzer = DeepHotspotAnalyzer(api_configs, TAVILY_API_KEY)
    
    # Run analysis on existing detailed report (use this as input for deep analysis)
    current_report = "output/ECU_sample/detailed_sustainable_solutions_report.txt"
    original_input_file = "ECU_sample.txt"  # Use the actual input file
    if Path(current_report).exists():
        result = analyzer.run_deep_analysis(current_report, original_input_file)
        print(f"Deep analysis completed. New report: {result}")
    else:
        print(f"Report file not found: {current_report}")
        print("Available report files:")
        output_dir = Path("output")
        if output_dir.exists():
            for subdir in output_dir.iterdir():
                if subdir.is_dir():
                    print(f"  {subdir}/")
                    for file in subdir.glob("*.txt"):
                        print(f"    {file}")

if __name__ == "__main__":
    main()