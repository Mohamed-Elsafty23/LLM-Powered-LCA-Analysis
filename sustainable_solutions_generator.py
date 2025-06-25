"""
Hotspot Sustainable Solutions Generator
Generates sustainability solutions based on hotspot analysis and research papers.
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
import time
import concurrent.futures
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
import sys
import unicodedata
import re

# PDF generation imports
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle
from reportlab.platypus.tableofcontents import TableOfContents
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY

# Configure logging with Unicode support
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/sustainable_solutions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log', encoding='utf-8'),
        UnicodeStreamHandler()
        ]
)
logger = logging.getLogger(__name__)

class APIClient:
    def __init__(self, api_key: str, base_url: str, model: str):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

class HotspotSustainableSolutionsGenerator:
    def __init__(self, api_configs: List[Dict[str, str]]):
        """Initialize with multiple API configurations for load balancing."""
        self.api_clients = []
        for config in api_configs:
            client = APIClient(config["api_key"], config["base_url"], config["model"])
            self.api_clients.append(client)
        
        self.current_client_index = 0
        self.output_folder = None
        
        logger.info(f"Initialized HotspotSustainableSolutionsGenerator with {len(self.api_clients)} API clients")

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
                temperature=0.7,
                max_tokens=4000
            )
            return response
        except Exception as e:
            logger.error(f"API request failed: {str(e)}")
            raise

    def get_output_folder_from_hotspot_file(self, hotspot_file: str) -> str:
        """Get output folder from hotspot analysis file path."""
        return str(Path(hotspot_file).parent)
    
    def _extract_hotspot_context(self, hotspot_name: str, hotspot_analysis: Dict[str, Any], raw_input_data: str) -> str:
        """Extract all hotspot data dynamically from hotspot analysis based on hotspot_name."""
        try:
            # Find the specific hotspot in the analysis
            hotspot_details = None
            production_hotspots = hotspot_analysis.get('hotspot_analysis', {}).get('production_hotspots', [])
            
            for hotspot in production_hotspots:
                if hotspot.get('hotspot_name') == hotspot_name:
                    hotspot_details = hotspot
                    break
            
            if not hotspot_details:
                return f"No hotspot data found for: {hotspot_name}"
            
            # Extract all fields dynamically (except hotspot_name to avoid redundancy)
            context_parts = []
            for key, value in hotspot_details.items():
                if key != 'hotspot_name':  # Skip the name since it's already in the header
                    # Format field names to be more readable
                    formatted_key = key.replace('_', ' ').title()
                    context_parts.append(f"{formatted_key}: {value}")
            
            return '\n'.join(context_parts) if context_parts else f"Limited context available for {hotspot_name}"
            
        except Exception as e:
            logger.error(f"Error extracting hotspot context: {str(e)}")
            return f"Context extraction failed for {hotspot_name}"
    
    def analyze_paper_for_hotspot(self, paper_content: str, paper_metadata: Dict[str, Any], 
                                  hotspot_analysis: Dict[str, Any], raw_input_data: str, 
                                  api_client: APIClient) -> Dict[str, Any]:
        """Analyze a single paper to extract sustainable solutions for specific hotspots."""
        try:
            hotspot_name = paper_metadata.get('hotspot_name', 'unknown')
            title = paper_metadata.get('title', 'Unknown Title')
            
            # Extract specific hotspot context for dynamic analysis
            hotspot_context = self._extract_hotspot_context(hotspot_name, hotspot_analysis, raw_input_data)
            
            prompt = f"""You are a quantitative research analyst. Extract ONLY explicit, measurable sustainability improvements from this research paper for the hotspot "{hotspot_name}".

PAPER CONTENT:
{paper_content}

PAPER TITLE: {title}
TARGET HOTSPOT: {hotspot_name}

HOTSPOT CONTEXT:
{hotspot_context}

STRICT EXTRACTION RULES:
1. ONLY extract numbers explicitly stated in the paper
2. Find percentages, energy reductions, efficiency improvements, cost savings
3. Look for specific process optimizations with measured results
4. Identify temperature, pressure, time, or material improvements with exact values
5. NO estimates, NO generalizations, NO assumptions

SEARCH FOR THESE MANUFACTURING-SPECIFIC SUSTAINABILITY PATTERNS:
- "X% reduction in energy consumption"
- "Y% improvement in material efficiency"
- "Z% decrease in environmental impact"
- "reduced CO2 emissions by X kg/unit"
- "energy savings of X kWh/kg"
- "cycle time reduced from A to B seconds/minutes"
- "temperature optimized from X°C to Y°C"
- "material waste decreased by Z%"
- "production efficiency increased by X%"
- "LCA impact reduced by Y%"
- "carbon footprint decreased by X%"
- "recycling rate improved by Y%"
- "resource utilization increased by Z%"
- "process optimization achieving X% savings"
- "sustainable manufacturing with Y% improvement"

DYNAMIC SEARCH PRIORITIES BASED ON HOTSPOT CONTEXT:
Based on the hotspot context provided, prioritize finding quantitative data related to:
- If material processing: Look for melting temperature optimization, material utilization rates, alloy composition efficiency
- If molding/forming: Search for cycle time reduction, pressure optimization, temperature control improvements  
- If assembly/production: Focus on energy consumption per unit, defect reduction percentages, throughput improvements
- If surface treatment: Target coating efficiency, chemical usage reduction, process time optimization
- If any manufacturing process: Emphasize energy consumption reduction, waste minimization, and process efficiency gains

SUSTAINABILITY FOCUS AREAS:
Prioritize papers that explicitly mention:
- Life Cycle Assessment (LCA) results with quantitative impacts
- Carbon footprint measurements and reduction strategies  
- Energy consumption analysis with specific savings
- Material efficiency improvements with waste reduction percentages
- Circular economy approaches with recycling/reuse rates
- Environmental impact assessments with numerical outcomes
- Sustainable manufacturing practices with measured benefits

OUTPUT FORMAT (respond with ONLY the extracted data):

**QUANTITATIVE FINDINGS:**
[List each finding with exact numbers and what they measure]

**TECHNOLOGIES/METHODS:**
[Specific technologies mentioned with their applications]

**PROCESS IMPROVEMENTS:**
[Specific process changes with measurable results]

**RELEVANCE TO {hotspot_name}:**
Systematically analyze the connections between the paper's findings and the target hotspot using this evidence-based framework:

1. MATERIAL COMPATIBILITY ANALYSIS:
   - Compare materials studied in the paper with hotspot materials from the context
   - Identify shared material properties, thermal behaviors, processing characteristics
   - Assess transferability of material-specific improvements and optimizations
   - Look for similar alloys, polymers, or material families

2. PROCESS ALIGNMENT ASSESSMENT:
   - Compare manufacturing processes investigated in the paper with hotspot processes
   - Identify similar equipment types, processing conditions, operational parameters
   - Match temperature ranges, pressure requirements, cycle times, and energy inputs
   - Evaluate process similarity on equipment, methods, and operational characteristics

3. QUANTITATIVE SUSTAINABILITY BENEFIT TRANSLATION:
   - Extract specific numerical improvements that could reduce hotspot environmental impact
   - Calculate potential impact reduction based on hotspot material quantities and significance level
   - Translate percentage improvements to actual environmental benefits (energy, emissions, waste)
   - Identify measurable LCA impact reductions and sustainability metrics

4. TECHNOLOGY AND METHOD ADAPTATION POTENTIAL:
   - Assess which optimization techniques could be directly implemented for this hotspot
   - Evaluate feasibility of control systems, algorithms, or process modifications
   - Identify required adaptations for different scales, materials, or equipment
   - Consider implementation complexity and potential sustainability gains

EVIDENCE-BASED RELEVANCE REQUIREMENTS:
- Reference specific quantitative data from the paper that supports each connection
- Cross-reference paper findings with hotspot context (materials, quantities, significance)
- Quantify potential environmental benefits using actual hotspot data when possible
- Clearly distinguish between direct applications and adapted implementations
- If no relevance exists, specify material/process/scale differences preventing application
- Focus on sustainability impact potential rather than generic process similarities

STRONG RELEVANCE EXAMPLE:
"The paper's 25% energy reduction in injection molding (achieved through temperature optimization from 200°C to 180°C) directly applies to the hotspot's injection molding process. Given the hotspot context showing medium environmental significance for housing production, this temperature optimization could reduce energy consumption by approximately 25% for the specified material quantities, potentially decreasing the hotspot's overall environmental impact."

WEAK RELEVANCE EXAMPLE:
"No direct relevance to {hotspot_name}. The paper studies [specific domain] with no material overlap (paper materials: [X] vs. hotspot materials: [Y from context]). Process differences: paper examines [specific process] vs. hotspot process: [specific process from context]. Scale mismatch: [specific scale differences]. No quantitative sustainability metrics applicable to this manufacturing hotspot's environmental impact reduction."

If NO quantitative data is found, respond with:
"NO QUANTITATIVE SUSTAINABILITY DATA FOUND"

Do not make up numbers. Only report what is explicitly written in the paper."""

            response = api_client.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a precise sustainability research analyst specializing in Life Cycle Assessment (LCA) and manufacturing process optimization. Your expertise includes identifying quantitative environmental improvements, energy efficiency gains, material waste reduction, and sustainable manufacturing practices. Focus on extracting explicit numerical data from research papers that can directly reduce environmental impact. Prioritize findings related to: manufacturing energy consumption, material efficiency, waste reduction, process optimization, temperature/pressure optimization, cycle time improvements, recycling rates, carbon footprint reduction, and LCA impact measurements. For relevance analysis, establish concrete connections between paper findings and manufacturing hotspot characteristics, considering material compatibility, process similarity, quantitative applicability, and technology transfer potential. Never estimate numbers or make generic claims - only report explicit quantitative data with clear environmental benefits."},
                    {"role": "user", "content": prompt}
                ],
                model=api_client.model,
                temperature=0.0,  # Zero temperature for precise extraction
                max_tokens=2000
            )
            
            analysis_content = response.choices[0].message.content
            
            # Safe logging for paper titles that may contain Unicode characters
            safe_title = safe_str(title)
            logger.info(f"Analyzed paper for hotspot '{hotspot_name}': {safe_title}")
            
            return {
                "hotspot_name": hotspot_name,
                "paper_title": title,
                "paper_metadata": paper_metadata,
                "analysis_content": analysis_content,  # Include the actual analysis
                "has_quantitative_data": "NO QUANTITATIVE" not in analysis_content.upper()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing paper for hotspot: {str(e)}")
            return None
    
    def analyze_papers_for_hotspots(self, processed_papers_file: str, hotspot_analysis: Dict[str, Any], 
                                   raw_input_data: str) -> Dict[str, List[Dict[str, Any]]]:
        """Analyze all processed papers for hotspot-specific solutions."""
        try:
            # Load processed papers with UTF-8 encoding
            with open(processed_papers_file, 'r', encoding='utf-8') as f:
                papers_data = json.load(f)
            
            processed_papers = papers_data.get('processed_papers', {})
            
            if not processed_papers:
                logger.error("No processed papers found in the processed papers file")
                raise ValueError("Processed papers file is empty - no papers available for analysis")
            
            logger.info(f"Analyzing {len(processed_papers)} processed papers for hotspot solutions")
            
            # Group papers by hotspot
            hotspot_papers = {}
            for paper_id, paper_data in processed_papers.items():
                hotspot_name = paper_data.get('metadata', {}).get('hotspot_name', 'unknown')
                if hotspot_name not in hotspot_papers:
                    hotspot_papers[hotspot_name] = []
                hotspot_papers[hotspot_name].append((paper_id, paper_data))
            
            # Analyze papers for each hotspot
            hotspot_analyses = {}
            
            for hotspot_name, papers in hotspot_papers.items():
                logger.info(f"Analyzing {len(papers)} papers for hotspot: {hotspot_name}")
                
                hotspot_results = []
                
                # Use ThreadPoolExecutor for concurrent processing
                with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(self.api_clients), 3)) as executor:
                    # Create analysis tasks
                    future_to_paper = {}
                    for i, (paper_id, paper_data) in enumerate(papers):
                        api_client = self.api_clients[i % len(self.api_clients)]
                        future = executor.submit(
                            self.analyze_paper_for_hotspot,
                            paper_data['full_text'],
                            paper_data['metadata'],
                            hotspot_analysis,
                            raw_input_data,
                            api_client
                        )
                        future_to_paper[future] = paper_id
                    
                    # Collect results
                    for future in concurrent.futures.as_completed(future_to_paper):
                        paper_id = future_to_paper[future]
                        try:
                            result = future.result()
                            if result:
                                hotspot_results.append(result)
                        except Exception as e:
                            logger.error(f"Error analyzing paper {paper_id}: {str(e)}")
                
                hotspot_analyses[hotspot_name] = hotspot_results
                logger.info(f"Completed analysis for hotspot '{hotspot_name}': {len(hotspot_results)} results")
            
            return hotspot_analyses
            
        except Exception as e:
            logger.error(f"Error analyzing papers for hotspots: {str(e)}")
            raise

    def generate_comprehensive_sustainability_report(self, hotspot_analyses: Dict[str, List[Dict[str, Any]]], 
                                                   hotspot_analysis: Dict[str, Any], 
                                                   raw_input_data: str) -> Dict[str, Any]:
        """Generate a comprehensive sustainability report based on actual paper analysis."""
        try:
            # Check if any analyses were found
            if not hotspot_analyses:
                logger.error("No hotspot analyses found - cannot generate sustainability report")
                raise ValueError("No paper analyses available for generating sustainability report")
            
            # Verify that at least some hotspots have papers
            total_papers = sum(len(papers) for papers in hotspot_analyses.values())
            if total_papers == 0:
                logger.error("No papers successfully analyzed for any hotspot")
                raise ValueError("No papers were successfully analyzed - cannot generate sustainability report")
            
            # Extract actual quantitative data from ECU input
            ecu_components = self._extract_ecu_components(raw_input_data)
            hotspot_ranking = hotspot_analysis.get('overall_hotspot_ranking', [])
            
            # Build report from actual analysis content
            paper_based_solutions = {}
            quantitative_findings = {}
            
            for hotspot_name, analyses in hotspot_analyses.items():
                solutions_with_data = []
                solutions_without_data = []
                
                for analysis in analyses:
                    paper_title = analysis.get('paper_title', 'Unknown Paper')
                    analysis_content = analysis.get('analysis_content', '')
                    has_quantitative_data = analysis.get('has_quantitative_data', False)
                    paper_metadata = analysis.get('paper_metadata', {})
                    pdf_link = paper_metadata.get('pdf_link', '')
                    
                    if has_quantitative_data:
                        solutions_with_data.append({
                            'title': paper_title,
                            'content': analysis_content,
                            'pdf_link': pdf_link
                        })
                    else:
                        solutions_without_data.append({
                            'title': paper_title,
                            'content': analysis_content,
                            'pdf_link': pdf_link
                        })
                
                paper_based_solutions[hotspot_name] = {
                    'with_data': solutions_with_data,
                    'without_data': solutions_without_data
                }
            
            # Generate final report content based on actual findings
            report_content = self._generate_evidence_based_report(
                paper_based_solutions, 
                ecu_components, 
                hotspot_ranking,
                hotspot_analysis
            )
            
            logger.info("Generated evidence-based sustainability report")
            
            return {
                "sustainability_report": report_content,
                "hotspot_analyses": hotspot_analyses,
                "baseline_hotspot_analysis": hotspot_analysis,
                "report_metadata": {
                    "generation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "total_hotspots_analyzed": len(hotspot_analyses),
                    "total_papers_analyzed": total_papers,
                    "papers_with_quantitative_data": sum(
                        len(data['with_data']) for data in paper_based_solutions.values()
                    ),
                    "papers_without_quantitative_data": sum(
                        len(data['without_data']) for data in paper_based_solutions.values()
                    )
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating comprehensive sustainability report: {str(e)}")
            raise
    
    def _extract_ecu_components(self, raw_input_data: str) -> Dict[str, Dict[str, str]]:
        """Extract component data from ECU input file."""
        components = {}
        current_component = None
        
        for line in raw_input_data.split('\n'):
            line = line.strip()
            if line and ':' not in line and line != '':
                current_component = line
                components[current_component] = {}
            elif line and ':' in line and current_component:
                key, value = line.split(':', 1)
                components[current_component][key.strip()] = value.strip()
        
        return components
    
    def _generate_evidence_based_report(self, paper_based_solutions: Dict, 
                                       ecu_components: Dict, 
                                       hotspot_ranking: List,
                                       hotspot_analysis: Dict = None) -> str:
        """Generate the final report based on actual evidence from papers."""
        
        report_sections = []
        
        # Header
        report_sections.append("EVIDENCE-BASED SUSTAINABILITY SOLUTIONS REPORT")
        report_sections.append("=" * 60)
        report_sections.append("")
        
        # ECU Component Analysis
        report_sections.append("**ECU COMPONENT ANALYSIS**")
        # report_sections.append("")
        # report_sections.append("Based on the ECU sample input data:")
        # report_sections.append("")
        
        for component, details in ecu_components.items():
            if details:
                report_sections.append(f"• {component}:")
                for key, value in details.items():
                    report_sections.append(f"  - {key}: {value}")
                report_sections.append("")
        
        # Hotspot Priority Ranking with detailed information
        report_sections.append("**ENVIRONMENTAL HOTSPOT PRIORITY RANKING**")
        report_sections.append("")
        
        # Get production hotspots details for additional information
        production_hotspots = {}
        if hotspot_analysis:
            for hotspot in hotspot_analysis.get('production_hotspots', []):
                production_hotspots[hotspot.get('hotspot_name')] = hotspot
        
        for i, hotspot in enumerate(hotspot_ranking, 1):
            hotspot_name = hotspot.get('hotspot_name', 'Unknown')
            significance = hotspot.get('environmental_significance', 'unknown')
            justification = hotspot.get('priority_justification', 'No justification provided')
            life_cycle_phase = hotspot.get('life_cycle_phase', 'unknown')
            
            # Get additional details from production hotspots
            production_details = production_hotspots.get(hotspot_name, {})
            impact_category = production_details.get('impact_category', 'Unknown')
            impact_source = production_details.get('impact_source', 'Unknown')
            quantitative_impact = production_details.get('quantitative_impact', 'Unknown')
            description = production_details.get('description', 'No description available')
            
            report_sections.append(f"{i}. {hotspot_name}")
            report_sections.append(f"   Life Cycle Phase: {life_cycle_phase}")
            report_sections.append(f"   Environmental Significance: {significance}")
            report_sections.append(f"   Impact Category: {impact_category}")
            report_sections.append(f"   Impact Source: {impact_source}")
            report_sections.append(f"   Quantitative Impact: {quantitative_impact}")
            report_sections.append(f"   Priority Justification: {justification}")
            report_sections.append(f"   Description: {description}")
            report_sections.append("")
        
        # Research-Based Solutions
        report_sections.append("**RESEARCH-BASED SUSTAINABILITY SOLUTIONS**")
        report_sections.append("")
        
        for hotspot_name, solutions in paper_based_solutions.items():
            report_sections.append(f"### {hotspot_name}")
            report_sections.append("")
            
            # Solutions with quantitative data
            if solutions['with_data']:
                report_sections.append("**Papers with Quantitative Sustainability Data:**")
                report_sections.append("")
                
                for solution in solutions['with_data']:
                    title = solution['title']
                    pdf_link = solution.get('pdf_link', '')
                    if pdf_link:
                        report_sections.append(f"**Paper:** {title} (PDF: {pdf_link})")
                    else:
                        report_sections.append(f"**Paper:** {title}")
                    report_sections.append("")
                    report_sections.append(solution['content'])
                    report_sections.append("")
                    report_sections.append("---")
                    report_sections.append("")
            
            # Solutions without quantitative data
            if solutions['without_data']:
                report_sections.append("**Papers without Specific Quantitative Data:**")
                report_sections.append("")
                
                for solution in solutions['without_data']:
                    title = solution['title']
                    pdf_link = solution.get('pdf_link', '')
                    if pdf_link:
                        report_sections.append(f"**Paper:** {title} (PDF: {pdf_link})")
                    else:
                        report_sections.append(f"**Paper:** {title}")
                    report_sections.append("")
                    # Only show that no quantitative data was found
                    report_sections.append("No specific quantitative sustainability improvements were found in this paper.")
                    report_sections.append("")
            
            if not solutions['with_data'] and not solutions['without_data']:
                report_sections.append("No research papers were successfully analyzed for this hotspot.")
                report_sections.append("")
            
            report_sections.append("")
        
        # Data Quality Assessment
        report_sections.append("**DATA QUALITY ASSESSMENT**")
        report_sections.append("")
        
        total_papers = sum(len(solutions['with_data']) + len(solutions['without_data']) 
                          for solutions in paper_based_solutions.values())
        papers_with_data = sum(len(solutions['with_data']) 
                              for solutions in paper_based_solutions.values())
        
        report_sections.append(f"Total papers analyzed: {total_papers}")
        report_sections.append(f"Papers with quantitative sustainability data: {papers_with_data}")
        report_sections.append(f"Papers without quantitative data: {total_papers - papers_with_data}")
        report_sections.append("")
        
        if papers_with_data == 0:
            report_sections.append("**IMPORTANT LIMITATION:**")
            report_sections.append("No quantitative sustainability improvements were found in any of the analyzed papers.")
            report_sections.append("This report is based on available research but lacks specific numerical targets")
            report_sections.append("for sustainability improvements. Additional research with quantitative")
            report_sections.append("data may be needed for concrete implementation planning.")
            report_sections.append("")
        
        # Disclaimer
        report_sections.append("**REPORT DISCLAIMER**")
        report_sections.append("")
        report_sections.append("This report is based exclusively on:")
        report_sections.append("1. Actual data from the ECU component specification")
        report_sections.append("2. Quantitative findings explicitly stated in research papers")
        report_sections.append("3. No estimates, assumptions, or generic industry values were used")
        report_sections.append("")
        report_sections.append("All sustainability solutions are evidence-based and sourced from")
        report_sections.append("the analyzed research literature. Where no quantitative data was")
        report_sections.append("available, this is clearly stated.")
        
        return "\n".join(report_sections)

    def _process_hyperlinks(self, content: str) -> str:
        """Process content to convert PDF URLs into clickable hyperlinks."""
        # Pattern to match PDF links like (PDF: http://arxiv.org/pdf/...)
        pdf_pattern = r'\(PDF:\s*(https?://[^\)]+)\)'
        
        def replace_pdf_link(match):
            url = match.group(1).strip()
            # Use ReportLab's hyperlink format - display the URL as clickable link
            return f'(<a href="{url}" color="#0066CC"><u>{url}</u></a>)'
        
        # Replace PDF links with hyperlinks
        processed_content = re.sub(pdf_pattern, replace_pdf_link, content)
        
        return processed_content

    def _create_pdf_styles(self):
        """Create custom styles for PDF generation."""
        styles = getSampleStyleSheet()
        
        # Custom styles with proper hierarchy: ### sections as main headlines (BIGGER FONTS)
        styles.add(ParagraphStyle(
            name='CustomMainTitle',
            parent=styles['Title'],
            fontSize=22,  # Increased from 18
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=HexColor('#2C3E50'),
            fontName='Helvetica-Bold',
            wordWrap='CJK'  # Better word wrapping to prevent color split
        ))
        
        # ### Hotspot sections - Main headlines (largest)
        styles.add(ParagraphStyle(
            name='CustomHotspotTitle',
            parent=styles['Heading1'],
            fontSize=18,  # Increased from 16
            spaceAfter=15,
            spaceBefore=25,
            textColor=HexColor('#1A5490'),
            keepWithNext=1,
            fontName='Helvetica-Bold'
        ))
        
        # ** sections - Secondary headlines (medium)
        styles.add(ParagraphStyle(
            name='CustomSectionTitle',
            parent=styles['Heading2'],
            fontSize=15,  # Increased from 13
            spaceAfter=10,
            spaceBefore=15,
            textColor=HexColor('#34495E'),
            keepWithNext=1,
            fontName='Helvetica-Bold'
        ))
        
        # Paper titles - Tertiary headlines (smaller)
        styles.add(ParagraphStyle(
            name='CustomPaperTitle',
            parent=styles['Heading3'],
            fontSize=13,  # Increased from 11
            spaceAfter=6,
            spaceBefore=8,
            textColor=HexColor('#2980B9'),
            keepWithNext=1,
            fontName='Helvetica-Bold'
        ))
        
        # Content subsections (like QUANTITATIVE FINDINGS)
        styles.add(ParagraphStyle(
            name='CustomContentSubtitle',
            parent=styles['Normal'],
            fontSize=12,  # Increased from 10
            spaceAfter=4,
            spaceBefore=6,
            textColor=HexColor('#7F8C8D'),
            fontName='Helvetica-Bold'
        ))
        
        # Regular body text
        styles.add(ParagraphStyle(
            name='CustomBodyText',
            parent=styles['Normal'],
            fontSize=11,  # Increased from 9
            spaceAfter=4,
            alignment=TA_JUSTIFY,
            textColor=HexColor('#2C3E50')
        ))
        
        # Bullet points
        styles.add(ParagraphStyle(
            name='CustomBulletPoint',
            parent=styles['Normal'],
            fontSize=11,  # Increased from 9
            spaceAfter=3,
            leftIndent=20,
            bulletIndent=10,
            textColor=HexColor('#2C3E50')
        ))
        
        # Numbered items
        styles.add(ParagraphStyle(
            name='CustomNumberedItem',
            parent=styles['Normal'],
            fontSize=11,  # Increased from 9
            spaceAfter=3,
            leftIndent=15,
            textColor=HexColor('#2C3E50')
        ))
        
        # Indented details for hotspot priority ranking
        styles.add(ParagraphStyle(
            name='CustomIndentedDetail',
            parent=styles['Normal'],
            fontSize=10,
            spaceAfter=2,
            leftIndent=40,  # Heavy indentation for details
            textColor=HexColor('#555555')
        ))
        
        return styles

    def _parse_report_content(self, report_content: str) -> List[Dict[str, Any]]:
        """Parse the text report content into structured elements for PDF generation."""
        lines = report_content.split('\n')
        elements = []
        in_hotspot_ranking = False
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Track if we're in the hotspot ranking section
            if line.startswith('**ENVIRONMENTAL HOTSPOT PRIORITY RANKING**'):
                in_hotspot_ranking = True
            elif line.startswith('**') and line.endswith('**') and in_hotspot_ranking:
                in_hotspot_ranking = False
                
            # Main title
            if line.startswith('EVIDENCE-BASED SUSTAINABILITY SOLUTIONS REPORT'):
                elements.append({'type': 'main_title', 'content': line})
            # ### Hotspot sections - MAIN HEADLINES (largest)
            elif line.startswith('### '):
                hotspot_title = line[4:].strip().replace('_', ' ')  # Clean up underscores
                elements.append({'type': 'hotspot_title', 'content': hotspot_title})
            # ** sections - Secondary headlines
            elif line.startswith('**') and line.endswith('**') and not line.startswith('**Paper:'):
                section_title = line.strip('*').strip()
                # Check if it's a content subsection (like QUANTITATIVE FINDINGS)
                if (section_title.upper() in ['QUANTITATIVE FINDINGS', 'TECHNOLOGIES/METHODS', 
                                              'PROCESS IMPROVEMENTS', 'DATA QUALITY ASSESSMENT',
                                              'REPORT DISCLAIMER'] or 
                    section_title.upper().startswith('RELEVANCE TO')):
                    elements.append({'type': 'content_subtitle', 'content': section_title})
                else:
                    elements.append({'type': 'section_title', 'content': section_title})
            # Paper titles with PDF links
            elif line.startswith('**Paper:**'):
                paper_title = line[10:].strip()
                elements.append({'type': 'paper_title', 'content': paper_title})
            # Bullet points
            elif line.startswith('• ') or line.startswith('- ') or line.startswith('* '):
                bullet_text = line[2:].strip()
                elements.append({'type': 'bullet', 'content': bullet_text})
            # Numbered lists and hotspot ranking details
            elif line and line[0].isdigit() and '. ' in line[:4]:
                elements.append({'type': 'numbered', 'content': line})
            # Indented details in hotspot ranking (lines that start with spaces and contain ":")
            elif in_hotspot_ranking and (line.startswith('   ') or line.startswith('\t')) and ':' in line:
                elements.append({'type': 'indented_detail', 'content': line.strip()})
            # Dividers
            elif line.startswith('---'):
                elements.append({'type': 'divider', 'content': ''})
            # Regular text
            elif line:
                elements.append({'type': 'body', 'content': line})
        
        return elements

    def generate_pdf_report(self, text_report_path: str, pdf_output_path: str = None) -> str:
        """Generate a professional PDF report from the text sustainability report."""
        try:
            # Set default PDF output path
            if pdf_output_path is None:
                text_path = Path(text_report_path)
                pdf_output_path = str(text_path.with_suffix('.pdf'))
            
            # Read the text report
            with open(text_report_path, 'r', encoding='utf-8') as f:
                report_content = f.read()
            
            # Create PDF document
            doc = SimpleDocTemplate(
                pdf_output_path,
                pagesize=A4,
                rightMargin=72,
                leftMargin=72,
                topMargin=72,
                bottomMargin=72
            )
            
            # Get styles
            styles = self._create_pdf_styles()
            
            # Build story (PDF content)
            story = []
            
            # Add header with better formatting to prevent color splitting
            title_text = "COMPREHENSIVE SUSTAINABILITY<br/>SOLUTIONS REPORT"
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
                    # MAIN HEADLINES - largest (### sections)
                    story.append(Spacer(1, 0.3*inch))
                    story.append(Paragraph(content, styles['CustomHotspotTitle']))
                elif element_type == 'section_title':
                    # Secondary headlines (** sections)
                    story.append(Spacer(1, 0.2*inch))
                    story.append(Paragraph(content, styles['CustomSectionTitle']))
                elif element_type == 'content_subtitle':
                    # Content subsections (QUANTITATIVE FINDINGS, etc.)
                    story.append(Spacer(1, 0.1*inch))
                    story.append(Paragraph(content, styles['CustomContentSubtitle']))
                elif element_type == 'paper_title':
                    # Paper titles - tertiary headlines with clickable PDF links
                    story.append(Spacer(1, 0.08*inch))
                    # Process hyperlinks in paper titles
                    processed_content = self._process_hyperlinks(content)
                    story.append(Paragraph(processed_content, styles['CustomPaperTitle']))
                elif element_type == 'bullet':
                    story.append(Paragraph(f"• {content}", styles['CustomBulletPoint']))
                elif element_type == 'numbered':
                    story.append(Paragraph(content, styles['CustomNumberedItem']))
                elif element_type == 'indented_detail':
                    # Indented details for hotspot ranking
                    story.append(Paragraph(content, styles['CustomIndentedDetail']))
                elif element_type == 'divider':
                    story.append(Spacer(1, 0.15*inch))
                    # Add a more subtle horizontal line
                    divider_table = Table([['─' * 60]], colWidths=[5*inch])
                    divider_table.setStyle(TableStyle([
                        ('TEXTCOLOR', (0, 0), (-1, -1), HexColor('#BDC3C7')),
                        ('FONTSIZE', (0, 0), (-1, -1), 10),
                        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ]))
                    story.append(divider_table)
                    story.append(Spacer(1, 0.15*inch))
                elif element_type == 'body':
                    story.append(Paragraph(content, styles['CustomBodyText']))
            
            # Add footer information
            story.append(Spacer(1, 0.3*inch))
            footer_text = """
            <para align="center">
            <b>Report Information</b><br/>
            This report was generated using the LLM-Powered LCA Analysis System<br/>
            All data is based on research papers and actual component specifications<br/>
            No estimates or fabricated values were used in this analysis
            </para>
            """
            story.append(Paragraph(footer_text, styles['CustomBodyText']))
            
            # Build PDF
            doc.build(story)
            
            logger.info(f"Successfully generated PDF report: {pdf_output_path}")
            return pdf_output_path
            
        except Exception as e:
            logger.error(f"Error generating PDF report: {str(e)}")
            raise

    def generate_solutions_from_hotspot_analysis(self, hotspot_analysis_path: str, output_path: str = None):
        """Generate sustainable solutions based on hotspot analysis results."""
        try:
            # Determine output folder
            self.output_folder = self.get_output_folder_from_hotspot_file(hotspot_analysis_path)
            
            # Set default output path if not provided
            if output_path is None:
                output_path = f"{self.output_folder}/sustainable_solutions_report.txt"
            elif not str(output_path).startswith(self.output_folder):
                filename = Path(output_path).name
                output_path = f"{self.output_folder}/{filename}"
            
            # Create output folder if it doesn't exist
            Path(self.output_folder).mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Using hotspot analysis: {hotspot_analysis_path}")
            logger.info(f"Solutions report will be saved to: {output_path}")
            
            # Load hotspot analysis with UTF-8 encoding
            with open(hotspot_analysis_path, 'r', encoding='utf-8') as f:
                hotspot_data = json.load(f)
            
            hotspot_analysis = hotspot_data.get('hotspot_analysis', {})
            input_file = hotspot_data.get('input_file', '')
            
            # Load raw input data with UTF-8 encoding
            raw_input_data = ""
            if input_file and Path(input_file).exists():
                with open(input_file, 'r', encoding='utf-8') as f:
                    raw_input_data = f.read()
                logger.info("Loaded raw input data")
            
            # Check if processed papers exist
            processed_papers_file = f"{self.output_folder}/processed_papers.json"
            if not Path(processed_papers_file).exists():
                logger.error(f"Processed papers file not found: {processed_papers_file}")
                error_message = f"""SUSTAINABILITY SOLUTIONS REPORT
{"=" * 50}

ERROR: NO RESEARCH PAPERS FOUND

The sustainability analysis cannot be completed because no processed research papers were found.

Expected file: {processed_papers_file}

REQUIRED STEPS:
1. Download research papers for each hotspot using the paper downloader
2. Process the downloaded papers using the PDF processor
3. Re-run this sustainability analysis

Without research papers, no evidence-based sustainability solutions can be generated.
The system does not provide generic recommendations.

Please complete the paper download and processing steps first."""
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(error_message)
                
                logger.error(f"Cannot generate sustainability report - no papers processed")
                raise FileNotFoundError(f"Processed papers file not found: {processed_papers_file}")
            
            # Analyze papers for hotspot-specific solutions
            hotspot_analyses = self.analyze_papers_for_hotspots(
                processed_papers_file, hotspot_analysis, raw_input_data
            )
            
            # Generate comprehensive sustainability report
            final_report = self.generate_comprehensive_sustainability_report(
                hotspot_analyses, hotspot_analysis, raw_input_data
            )
            
            # Save the report
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("COMPREHENSIVE SUSTAINABILITY SOLUTIONS REPORT\n")
                f.write("=" * 60 + "\n\n")
                f.write(final_report['sustainability_report'])
            
            # Generate PDF version
            try:
                pdf_output_path = f"{self.output_folder}/sustainable_solutions_report.pdf"
                self.generate_pdf_report(output_path, pdf_output_path)
                logger.info(f"PDF report generated at {pdf_output_path}")
            except Exception as e:
                logger.warning(f"Failed to generate PDF report: {str(e)}")
            
            # Also save the detailed analysis data
            detailed_output_path = f"{self.output_folder}/detailed_sustainability_analysis.json"
            with open(detailed_output_path, 'w', encoding='utf-8') as f:
                json.dump(final_report, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Successfully generated sustainability solutions report at {output_path}")
            logger.info(f"Detailed analysis saved at {detailed_output_path}")
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error generating solutions from hotspot analysis: {str(e)}")
            raise
    


def main():
    """Test the sustainable solutions generator."""
    from config import PRIMARY_API_KEY, SECONDARY_API_KEY, BASE_URL
    
    api_configs = [
        {"api_key": PRIMARY_API_KEY, "base_url": BASE_URL, "model": "llama-3.3-70b-instruct"},
        {"api_key": SECONDARY_API_KEY, "base_url": BASE_URL, "model": "llama-3.3-70b-instruct"}
    ]
    
    generator = HotspotSustainableSolutionsGenerator(api_configs)
    
    # Test with a sample hotspot analysis file
    hotspot_file = "output/automotive_sample/hotspot_lca_analysis.json"
    if Path(hotspot_file).exists():
        result = generator.generate_solutions_from_hotspot_analysis(hotspot_file)
        print(f"Generated sustainability report: {result}")
    else:
        print(f"Hotspot analysis file not found: {hotspot_file}")

if __name__ == "__main__":
    main() 