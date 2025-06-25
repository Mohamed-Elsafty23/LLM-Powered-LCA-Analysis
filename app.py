import streamlit as st
import json
import time
from pathlib import Path
import logging
import warnings
# Remove the import of new_lca_workflow and implement the workflow directly in this file
from lca_implementation import HotspotLCAAnalyzer
from arxiv_paper_downloader import ArxivPaperDownloader
from pdf_processor import PDFProcessor
from sustainable_solutions_generator import HotspotSustainableSolutionsGenerator
from deep_hotspot_analyzer import DeepHotspotAnalyzer
from config import PRIMARY_API_KEY, SECONDARY_API_KEY, BASE_URL
import queue
import threading
import sys
from datetime import datetime
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import base64
from io import BytesIO
from lca_visualizations import create_all_visualizations as create_lca_visualizations, get_latest_visualizations as get_latest_lca_visualizations
from sustainable_solutions_visualizations import create_all_visualizations as create_solutions_visualizations, get_latest_visualizations as get_latest_solutions_visualizations
import re
from typing import Dict

# Suppress Faiss GPU warning
warnings.filterwarnings('ignore', message='.*Failed to load GPU Faiss.*')

# Create a queue for log messages
log_queue = queue.Queue()

# Custom StreamHandler to capture logs
class StreamlitHandler(logging.Handler):
    def emit(self, record):
        try:
            msg = self.format(record)
            # Ensure the message is properly encoded
            if isinstance(msg, str):
                msg = msg.encode('utf-8', errors='replace').decode('utf-8')
            log_queue.put(msg)
        except Exception as e:
            self.handleError(record)
            logger.error(f"Error in StreamlitHandler: {str(e)}")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/app.log', encoding='utf-8'),
        StreamlitHandler()
    ]
)
logger = logging.getLogger(__name__)

class NewLCAWorkflow:
    def __init__(self):
        """Initialize the new LCA workflow with all required components."""
        # API configuration
        self.api_configs = [
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
        
        # Initialize workflow components
        self.hotspot_analyzer = HotspotLCAAnalyzer([PRIMARY_API_KEY, SECONDARY_API_KEY], BASE_URL)
        self.arxiv_downloader = ArxivPaperDownloader(max_results_per_query=10)
        self.pdf_processor = PDFProcessor()
        self.solutions_generator = HotspotSustainableSolutionsGenerator(self.api_configs)
        self.deep_analyzer = DeepHotspotAnalyzer(self.api_configs, "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")
        
        logger.info("Initialized NewLCAWorkflow with all components")
    
    def execute_complete_workflow(self, input_file: str) -> Dict[str, str]:
        """
        Execute the complete new LCA workflow.
        
        Args:
            input_file: Path to the raw input data file
            
        Returns:
            Dict containing paths to all generated outputs
        """
        try:
            logger.info(f"Starting complete LCA workflow for input file: {input_file}")
            workflow_start_time = time.time()
            
            # Step 1: Perform hotspot analysis directly on raw input data
            logger.info("=" * 60)
            logger.info("STEP 1: HOTSPOT ANALYSIS")
            logger.info("=" * 60)
            
            hotspot_results = self.hotspot_analyzer.analyze_from_raw_input(input_file)
            hotspot_file = f"{self.hotspot_analyzer.output_folder}/hotspot_lca_analysis.json"
            
            logger.info(f"[SUCCESS] Hotspot analysis completed. Results saved to: {hotspot_file}")
            
            # Step 2: Extract search queries from hotspot analysis
            logger.info("=" * 60)
            logger.info("STEP 2: SEARCH QUERY EXTRACTION")
            logger.info("=" * 60)
            
            search_queries = hotspot_results.get('search_queries', {}).get('hotspot_queries', {})
            logger.info(f"[SUCCESS] Extracted {len(search_queries)} search queries for hotspots:")
            for hotspot_name, query in search_queries.items():
                logger.info(f"  - {hotspot_name}: {query}")
            
            # Step 3: Download ArXiv papers using multi-query strategy
            logger.info("=" * 60)
            logger.info("STEP 3: ARXIV PAPER DOWNLOAD")
            logger.info("=" * 60)
            
            downloaded_papers = self.arxiv_downloader.search_and_download_papers(
                search_queries, self.hotspot_analyzer.output_folder
            )
            
            total_downloaded = sum(len(papers) for papers in downloaded_papers.values())
            logger.info(f"[SUCCESS] Downloaded {total_downloaded} papers for {len(search_queries)} hotspots")
            
            # Step 4: Process downloaded papers (extract text and remove references)
            logger.info("=" * 60)
            logger.info("STEP 4: PDF PROCESSING")
            logger.info("=" * 60)
            
            processed_papers_file = self.pdf_processor.process_papers_for_project(
                self.hotspot_analyzer.output_folder
            )
            
            logger.info(f"[SUCCESS] Papers processed and references removed. Results saved to: {processed_papers_file}")
            
            # Step 5: Generate comprehensive sustainability report
            logger.info("=" * 60)
            logger.info("STEP 5: DETAILED SUSTAINABILITY REPORT GENERATION")
            logger.info("=" * 60)
            
            detailed_solutions_report_file = self.solutions_generator.generate_solutions_from_hotspot_analysis(
                hotspot_file
            )
            
            logger.info(f"[SUCCESS] Detailed sustainability solutions report generated: {detailed_solutions_report_file}")
            
            # Step 6: Generate deep hotspot analysis report
            logger.info("=" * 60)
            logger.info("STEP 6: DEEP HOTSPOT ANALYSIS REPORT GENERATION")
            logger.info("=" * 60)
            
            solutions_report_file = self.deep_analyzer.run_deep_analysis(detailed_solutions_report_file)
            
            logger.info(f"[SUCCESS] Final sustainability solutions report generated: {solutions_report_file}")
            
            # Calculate total workflow time
            workflow_time = time.time() - workflow_start_time
            logger.info("=" * 60)
            logger.info("WORKFLOW COMPLETION SUMMARY")
            logger.info("=" * 60)
            
            logger.info(f"[SUCCESS] Total workflow time: {workflow_time:.2f} seconds")
            logger.info(f"[SUCCESS] Input file: {input_file}")
            logger.info(f"[SUCCESS] Output folder: {self.hotspot_analyzer.output_folder}")
            
            # Return all output file paths
            outputs = {
                "input_file": input_file,
                "output_folder": self.hotspot_analyzer.output_folder,
                "hotspot_analysis": hotspot_file,
                "processed_papers": processed_papers_file,
                "detailed_sustainability_report": detailed_solutions_report_file,
                "sustainability_report": solutions_report_file,
                "hotspot_lca_analysis_with_papers": f"{self.hotspot_analyzer.output_folder}/hotspot_lca_analysis_with_papers.json",
                "downloaded_papers_folder": f"{self.hotspot_analyzer.output_folder}/downloaded_papers",
                "query_mapping": f"{self.hotspot_analyzer.output_folder}/downloaded_papers/query_paper_mapping.json"
            }
            
            logger.info("Generated files:")
            for output_type, file_path in outputs.items():
                if Path(file_path).exists():
                    logger.info(f"  [SUCCESS] {output_type}: {file_path}")
                else:
                    logger.warning(f"  [MISSING] {output_type}: {file_path} (not found)")
            
            return outputs
            
        except Exception as e:
            logger.error(f"Error in complete workflow execution: {str(e)}")
            raise
    
    def execute_workflow_from_hotspot_analysis(self, hotspot_analysis_file: str) -> Dict[str, str]:
        """
        Execute workflow starting from existing hotspot analysis (skip steps 1-2).
        
        Args:
            hotspot_analysis_file: Path to existing hotspot analysis file
            
        Returns:
            Dict containing paths to generated outputs
        """
        try:
            logger.info(f"Starting workflow from existing hotspot analysis: {hotspot_analysis_file}")
            
            # Load hotspot analysis
            with open(hotspot_analysis_file, 'r') as f:
                hotspot_data = json.load(f)
            
            search_queries = hotspot_data.get('search_queries', {}).get('hotspot_queries', {})
            output_folder = Path(hotspot_analysis_file).parent
            
            logger.info(f"Found {len(search_queries)} search queries")
            logger.info(f"Using output folder: {output_folder}")
            
            # Execute steps 3-5
            downloaded_papers = self.arxiv_downloader.search_and_download_papers(
                search_queries, str(output_folder)
            )
            
            processed_papers_file = self.pdf_processor.process_papers_for_project(str(output_folder))
            
            solutions_report_file = self.solutions_generator.generate_solutions_from_hotspot_analysis(
                hotspot_analysis_file
            )
            
            outputs = {
                "hotspot_analysis": hotspot_analysis_file,
                "processed_papers": processed_papers_file,
                "sustainability_report": solutions_report_file,
                "output_folder": str(output_folder)
            }
            
            logger.info("Workflow completed successfully from existing hotspot analysis")
            return outputs
            
        except Exception as e:
            logger.error(f"Error in workflow from hotspot analysis: {str(e)}")
            raise

def get_api_keys():
    """Get API keys from either config file or Streamlit secrets."""
    try:
        # First try to get from config file
        primary_key = PRIMARY_API_KEY
        secondary_key = SECONDARY_API_KEY
        base_url = BASE_URL
        
        # If any of the keys are None or empty, try to get from Streamlit secrets
        if not primary_key or not secondary_key:
            if 'PRIMARY_API_KEY' in st.secrets:
                primary_key = st.secrets['PRIMARY_API_KEY']
            if 'SECONDARY_API_KEY' in st.secrets:
                secondary_key = st.secrets['SECONDARY_API_KEY']  
                
            if 'BASE_URL' in st.secrets:
                base_url = st.secrets['BASE_URL']
        
        # Validate that we have the required keys
        if not primary_key or not secondary_key:
            raise ValueError("API keys not found in either config file or Streamlit secrets")
        
        return primary_key, secondary_key, base_url
    except Exception as e:
        logger.error(f"Error getting API keys: {str(e)}")
        raise

def get_download_link(val, filename):
    """Generate a download link for a file."""
    b64 = base64.b64encode(val)  # val looks like b'...'
    return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="{filename}">Download {filename}</a>'

def create_hotspot_chart(hotspot_data):
    """Create an interactive chart for hotspot analysis results."""
    # Extract hotspot data for visualization
    chart_data = []
    
    # Process different hotspot categories
    for category in ['production_hotspots', 'distribution_hotspots', 'use_hotspots', 'end_of_life_hotspots']:
        if category in hotspot_data:
            for hotspot in hotspot_data[category]:
                if isinstance(hotspot, dict):
                    chart_data.append({
                        'Category': category.replace('_hotspots', '').replace('_', ' ').title(),
                        'Hotspot': hotspot.get('hotspot_name', 'Unknown'),
                        'Significance': hotspot.get('environmental_significance', 'medium'),
                        'Impact Category': hotspot.get('impact_category', 'Unknown')
                    })
    
    if chart_data:
        df = pd.DataFrame(chart_data)
        
        # Create significance mapping for colors
        significance_colors = {'high': 'red', 'medium': 'orange', 'low': 'green'}
        df['Color'] = df['Significance'].map(significance_colors)
        
        fig = px.bar(df, x='Category', y='Hotspot', color='Significance',
                    title='Environmental Hotspots by Life Cycle Phase',
                    labels={'Hotspot': 'Hotspot Name', 'Category': 'Life Cycle Phase'},
                    color_discrete_map=significance_colors)
        return fig
    return None

def initialize_new_workflow():
    """Initialize the new LCA workflow components."""
    try:
        # Get API keys from either source
        primary_key, secondary_key, base_url = get_api_keys()
        
        # Initialize new workflow
        workflow = NewLCAWorkflow()
        
        logger.info("New LCA workflow initialized successfully")
        return workflow
    except Exception as e:
        logger.error(f"Error initializing new workflow: {str(e)}")
        raise

def get_output_folder_from_input(input_file: str) -> str:
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
    
    return output_folder

def get_project_name_from_upload(uploaded_file) -> str:
    """
    Extract project name from uploaded file name.
    
    Args:
        uploaded_file: Streamlit uploaded file object
        
    Returns:
        str: Project name for folder creation
    """
    if not uploaded_file or not uploaded_file.name:
        return "automotive_sample"
    
    # Extract filename without extension
    file_path = Path(uploaded_file.name)
    return file_path.stem

def update_logs(log_container):
    """Update the log display with new messages."""
    log_text = ""
    while True:
        try:
            # Get new log messages
            while not log_queue.empty():
                log_message = log_queue.get_nowait()
                log_text += log_message + "\n"
                # Update the container with all messages
                log_container.text(log_text)
            time.sleep(0.1)
        except Exception as e:
            logger.error(f"Error updating logs: {str(e)}")
            break

def format_hotspot_report(report):
    """Format the hotspot LCA report for better display."""
    def format_key(key):
        """Format key for display by replacing underscores with spaces and capitalizing."""
        return key.replace('_', ' ').title()

    def display_hotspot_list(hotspots, phase_name):
        """Display a list of hotspots for a specific phase."""
        if not hotspots:
            st.write("No significant hotspots identified")
            return
            
        st.markdown(f"#### {format_key(phase_name)}")
        
        for i, hotspot in enumerate(hotspots, 1):
            if isinstance(hotspot, dict):
                with st.expander(f"Hotspot {i}: {hotspot.get('hotspot_name', 'Unknown')}"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown(f"**Impact Category:** {hotspot.get('impact_category', 'N/A')}")
                        st.markdown(f"**Environmental Significance:** {hotspot.get('environmental_significance', 'N/A')}")
                    
                    with col2:
                        st.markdown(f"**Impact Source:** {hotspot.get('impact_source', 'N/A')}")
                        if 'quantitative_impact' in hotspot:
                            st.markdown(f"**Quantitative Impact:** {hotspot['quantitative_impact']}")
                    
                    if 'description' in hotspot:
                        st.markdown(f"**Description:** {hotspot['description']}")

    # Main report structure
    if 'hotspot_analysis' in report:
        hotspot_data = report['hotspot_analysis']
    else:
        hotspot_data = report
    
    # Display overall hotspot ranking first
    if 'overall_hotspot_ranking' in hotspot_data:
        st.header("🔥 Overall Hotspot Ranking")
        ranking = hotspot_data['overall_hotspot_ranking']
        
        if ranking:
            # Create a DataFrame for better display
            ranking_df = pd.DataFrame(ranking)
            if not ranking_df.empty:
                # Display as an interactive table
                st.dataframe(ranking_df, use_container_width=True)
                
                # Create chart if we have significance data
                if 'environmental_significance' in ranking_df.columns:
                    fig = create_hotspot_chart({'overall_hotspot_ranking': ranking})
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
    
    # Create tabs for different lifecycle phases
    tab_names = []
    tab_data = []
    
    for phase in ['production_hotspots', 'distribution_hotspots', 'use_hotspots', 'end_of_life_hotspots']:
        if phase in hotspot_data and hotspot_data[phase]:
            tab_names.append(format_key(phase.replace('_hotspots', '')))
            tab_data.append((phase, hotspot_data[phase]))
    
    if tab_names:
        tabs = st.tabs(tab_names)
        
        for tab, (phase, hotspots) in zip(tabs, tab_data):
            with tab:
                display_hotspot_list(hotspots, phase)
    
    # Display search queries if available
    if 'search_queries' in hotspot_data:
        st.header("🔍 Generated Search Queries")
        search_queries = hotspot_data['search_queries']
        
        if 'hotspot_queries' in search_queries:
            queries = search_queries['hotspot_queries']
            for hotspot_name, query in queries.items():
                with st.expander(f"Search Query for: {hotspot_name}"):
                    st.code(query, language='text')

def format_solutions_report(report_text):
    """Format the sustainable solutions report for display."""
    def parse_sections(content):
        """Parse the content into sections based on headers and formatting."""
        if not content:
            return []
        
        # Split content by double newlines to get potential sections
        raw_sections = content.split('\n\n')
        sections = []
        current_section = []
        
        for section in raw_sections:
            section = section.strip()
            if not section:
                continue
                
            # Check if it's a header (starts with ** or ###)
            if section.startswith('**') or section.startswith('###'):
                if current_section:
                    sections.append('\n'.join(current_section))
                    current_section = []
                current_section.append(section)
            else:
                current_section.append(section)
        
        if current_section:
            sections.append('\n'.join(current_section))
        
        return sections

    def format_references(section):
        """Format the references section with proper citations."""
        # Display the title
        st.markdown("**References**")
        
        # Split the section into lines and process each reference
        lines = section.split('\n')
        for line in lines:
            if line.strip() and not line.startswith('**References**'):
                # Add a new line before each reference
                st.markdown("\n" + line.strip())

    def format_section(section):
        """Format a section with appropriate styling."""
        # Special handling for References section
        if section.startswith('**References**'):
            format_references(section)
            return
            
        # Check if it's a header
        if section.startswith('**'):
            st.markdown(section, unsafe_allow_html=True)
        elif section.startswith('###'):
            st.markdown(section, unsafe_allow_html=True)
        # Check if it's a list item
        elif section.startswith('*'):
            st.markdown(section, unsafe_allow_html=True)
        # Regular paragraph
        else:
            st.markdown(section, unsafe_allow_html=True)

    # Parse sections
    sections = parse_sections(report_text)
    
    # Display title (assuming first line is title)
    title = report_text.split('\n')[0].strip()
    st.title(title)
    st.markdown("---")

    # Display each section
    for section in sections:
        if section.strip():
            format_section(section)
            st.markdown("---")

def load_visualizations(output_folder=None):
    """Load the latest visualizations for both hotspot LCA and sustainable solutions from project-specific folders."""
    lca_viz = {}
    solutions_viz = {}
    
    if output_folder:
        # Load hotspot LCA visualizations from project-specific folder
        lca_dir = Path(output_folder) / "visualizations" / "hotspot_lca"
        if lca_dir.exists():
            # Get all timestamped directories and sort by name (newest first)
            timestamp_dirs = [d for d in lca_dir.iterdir() if d.is_dir() and all(part.isdigit() for part in d.name.split('_'))]
            if timestamp_dirs:
                newest_dir = sorted(timestamp_dirs, key=lambda x: x.name, reverse=True)[0]
                for html_file in newest_dir.glob("*.html"):
                    with open(html_file, 'r', encoding='utf-8') as f:
                        lca_viz[html_file.stem] = f.read()
        
        # Load sustainable solutions visualizations from project-specific folder
        solutions_dir = Path(output_folder) / "visualizations" / "sustainable_solutions"
        if solutions_dir.exists():
            # Get all timestamped directories and sort by name (newest first)
            timestamp_dirs = [d for d in solutions_dir.iterdir() if d.is_dir() and all(part.isdigit() for part in d.name.split('_'))]
            if timestamp_dirs:
                newest_dir = sorted(timestamp_dirs, key=lambda x: x.name, reverse=True)[0]
                for html_file in newest_dir.glob("*.html"):
                    with open(html_file, 'r', encoding='utf-8') as f:
                        solutions_viz[html_file.stem] = f.read()
    else:
        # Fallback to default global visualization folders
        # Load hotspot LCA visualizations from global folder
        lca_dir = Path("visualizations/hotspot_lca")
        if lca_dir.exists():
            # Get all timestamped directories and sort by name (newest first)
            timestamp_dirs = [d for d in lca_dir.iterdir() if d.is_dir() and all(part.isdigit() for part in d.name.split('_'))]
            if timestamp_dirs:
                newest_dir = sorted(timestamp_dirs, key=lambda x: x.name, reverse=True)[0]
                for html_file in newest_dir.glob("*.html"):
                    with open(html_file, 'r', encoding='utf-8') as f:
                        lca_viz[html_file.stem] = f.read()
        
        # Load sustainable solutions visualizations from global folder
        solutions_dir = Path("visualizations/sustainable_solutions")
        if solutions_dir.exists():
            # Get all timestamped directories and sort by name (newest first)
            timestamp_dirs = [d for d in solutions_dir.iterdir() if d.is_dir() and all(part.isdigit() for part in d.name.split('_'))]
            if timestamp_dirs:
                newest_dir = sorted(timestamp_dirs, key=lambda x: x.name, reverse=True)[0]
                for html_file in newest_dir.glob("*.html"):
                    with open(html_file, 'r', encoding='utf-8') as f:
                        solutions_viz[html_file.stem] = f.read()
    
    return lca_viz, solutions_viz

def check_existing_files(output_folder: str) -> Dict[str, bool]:
    """
    Check which analysis files already exist in the output folder for the new workflow.
    
    Args:
        output_folder: Path to the project output folder
        
    Returns:
        Dict: Status of each analysis step
    """
    output_path = Path(output_folder)
    
    # Define required files for each step in the new workflow
    required_files = {
        "hotspot_analysis": output_path / "hotspot_lca_analysis.json",
        "downloaded_papers": output_path / "downloaded_papers",
        "processed_papers": output_path / "processed_papers.json", 
        "detailed_sustainable_solutions": output_path / "detailed_sustainable_solutions_report.txt",
        "sustainable_solutions": output_path / "sustainable_solutions_report.txt",
        "lca_visualizations": output_path / "visualizations" / "hotspot_lca",
        "solutions_visualizations": output_path / "visualizations" / "sustainable_solutions"
    }
    
    # Check which files exist and are not empty
    file_status = {}
    for step, file_path in required_files.items():
        if step in ["downloaded_papers", "lca_visualizations", "solutions_visualizations"]:
            # For directories, check if they exist and contain files
            exists = file_path.exists() and any(file_path.iterdir()) if file_path.exists() else False
        else:
            # For regular files, check if they exist and are not empty
            exists = file_path.exists() and file_path.stat().st_size > 0 if file_path.exists() else False
        file_status[step] = exists
    
    return file_status

def get_steps_to_run(file_status: Dict[str, bool]) -> Dict[str, bool]:
    """
    Determine which steps need to be run based on existing files for the new workflow.
    
    Args:
        file_status: Dictionary of file existence status
        
    Returns:
        Dict: Which steps should be executed
    """
    # Check if hotspot analysis file exists
    hotspot_analysis_exists = file_status["hotspot_analysis"]
    
    # If hotspot analysis exists, we can skip to paper download or sustainability solutions
    if hotspot_analysis_exists:
        steps_to_run = {
            "Hotspot Analysis": False,
            "ArXiv Paper Download": not file_status["downloaded_papers"],
            "PDF Processing": not file_status["processed_papers"],
            "Detailed Sustainability Report": not file_status["detailed_sustainable_solutions"],
            "Deep Hotspot Analysis": not file_status["sustainable_solutions"],
            # "Visualization Generation": not (file_status["lca_visualizations"] and file_status["solutions_visualizations"])
        }
    else:
        # If hotspot analysis is missing, run all steps
        steps_to_run = {
            "Hotspot Analysis": True,
            "ArXiv Paper Download": True,
            "PDF Processing": True,
            "Detailed Sustainability Report": True,
            "Deep Hotspot Analysis": True,
            # "Visualization Generation": True
        }
    
    return steps_to_run

def main():
    # Set page config with custom theme
    st.set_page_config(
        page_title="LLM-powered LCA Hotspot Analysis Tool",
        # page_icon="🔥",
        layout="wide"
    )
    
    # Initialize session state
    if 'analysis_completed' not in st.session_state:
        st.session_state.analysis_completed = False
    if 'current_file_hash' not in st.session_state:
        st.session_state.current_file_hash = None
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    if 'output_folder' not in st.session_state:
        st.session_state.output_folder = "output/automotive_sample"  # Default folder
    
    # Add custom CSS for better formatting
    st.markdown("""
        <style>
        .stApp {
            max-width: 1200px;
            margin: 0 auto;
        }
        .report-text {
            font-size: 16px;
            line-height: 1.6;
            margin: 20px 0;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 2rem;
            background-color: white;
            border-bottom: 1px solid #e5e7eb;
        }
        .stTabs [data-baseweb="tab"] {
            height: 3rem;
            white-space: pre-wrap;
            background-color: white;
            border-radius: 0;
            gap: 1rem;
            padding-top: 10px;
            padding-bottom: 10px;
            color: #4b5563;
            font-weight: 500;
        }
        .stTabs [aria-selected="true"] {
            background-color: white;
            color: #1f2937;
            border-bottom: 2px solid #4CAF50;
            font-weight: 600;
        }
        .stTabs [aria-selected="false"] {
            background-color: white;
            color: #6b7280;
        }
        .stTabs [data-baseweb="tab-panel"] {
            background-color: white;
            padding: 1rem 0;
        }
        .download-link {
            display: inline-block;
            padding: 10px 20px;
            background-color: #4CAF50;
            color: white;
            text-decoration: none;
            border-radius: 5px;
            margin: 10px 0;
        }
        .step-container {
            padding: 20px;
            margin: 10px 0;
            border-radius: 5px;
            background-color: #f8f9fa;
        }
        .step-title {
            font-size: 1.2em;
            font-weight: bold;
            color: #2c3e50;
        }
        .step-status {
            margin-left: 10px;
            font-weight: normal;
        }
        /* Enhanced styles for report formatting */
        h1 {
            font-size: 2.2em;
            color: #1f2937;
            margin-bottom: 1em;
            border-bottom: 2px solid #e5e7eb;
            padding-bottom: 0.5em;
        }
        h2 {
            font-size: 1.8em;
            color: #374151;
            margin-top: 1.5em;
            margin-bottom: 0.8em;
            border-bottom: 1px solid #e5e7eb;
            padding-bottom: 0.3em;
        }
        h3 {
            font-size: 1.4em;
            color: #4b5563;
            margin-top: 1.2em;
            margin-bottom: 0.6em;
        }
        strong {
            color: #1f2937;
            font-weight: 600;
        }
        p {
            margin: 0.8em 0;
            line-height: 1.6;
        }
        /* Add styles for numerical values */
        .number-value {
            font-family: 'Courier New', monospace;
            color: #059669;
            font-weight: 500;
        }
        /* Add styles for section headers */
        .section-header {
            background-color: #f3f4f6;
            padding: 0.5em 1em;
            border-radius: 4px;
            margin: 1em 0;
        }
        /* Add styles for lists */
        ul, ol {
            margin: 0.8em 0;
            padding-left: 1.5em;
        }
        li {
            margin: 0.4em 0;
        }
        /* Enhanced styles for LCA report formatting */
        .lca-section {
            margin: 1.5em 0;
            padding: 1em;
            background-color: #f8f9fa;
            border-radius: 8px;
        }
        
        .lca-subsection {
            margin: 1em 0;
            padding: 0.8em;
            background-color: #ffffff;
            border-left: 4px solid #4CAF50;
        }
        
        .lca-value {
            font-family: 'Courier New', monospace;
            color: #059669;
            font-weight: 500;
            padding: 0.2em 0.4em;
            background-color: #f0fdf4;
            border-radius: 4px;
        }
        
        /* Enhanced styles for solutions report formatting */
        .solution-content {
            margin: 1em 0;
            padding: 1em;
            background-color: #f8f9fa;
            border-radius: 8px;
            line-height: 1.6;
        }
        
        .solution-header {
            margin: 1.5em 0 1em 0;
            padding: 0.5em 1em;
            background-color: #e8f5e9;
            border-radius: 4px;
            font-weight: 600;
        }
        
        /* Improved list formatting */
        ul, ol {
            margin: 0.8em 0;
            padding-left: 1.5em;
        }
        
        li {
            margin: 0.4em 0;
            line-height: 1.6;
        }
        
        /* Improved table formatting */
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 1em 0;
        }
        
        th, td {
            padding: 0.8em;
            border: 1px solid #e5e7eb;
            text-align: left;
        }
        
        th {
            background-color: #f3f4f6;
            font-weight: 600;
        }
        
        tr:nth-child(even) {
            background-color: #f9fafb;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.title("LLM-powered LCA Hotspot Analysis Tool")
    st.markdown("""
    This tool performs **Hotspot-driven Life Cycle Assessment (LCA)** analysis using the new LLM-powered workflow.
    
    **Workflow Steps:**
    1. 🔥 **Hotspot Analysis** - Identifies environmental hotspots directly from raw input data
    2. 📚 **ArXiv Paper Download** - Downloads relevant research papers for each hotspot
    3. 📄 **PDF Processing** - Extracts and processes paper content 
    4. 📊 **Detailed Sustainability Report** - Generates comprehensive sustainability analysis with research data
    5. 🎯 **Deep Hotspot Analysis** - Performs focused quantitative analysis with web search augmentation
    <!-- 6. 📊 **Visualizations** - Creates interactive charts and reports -->
    
    Upload a text file containing your raw input data to begin the analysis.
    """)
    
    # File upload
    uploaded_file = st.file_uploader("Upload your input file", type=['txt'])
    
    if uploaded_file is not None:
        import hashlib
        
        # Calculate hash of the uploaded file content
        file_content = uploaded_file.getvalue()
        file_hash = hashlib.md5(file_content).hexdigest()
        
        # Check if this is a new file or the analysis is already completed
        if st.session_state.current_file_hash != file_hash:
            # New file uploaded, reset session state
            st.session_state.analysis_completed = False
            st.session_state.current_file_hash = file_hash
            st.session_state.analysis_results = None
            st.session_state.output_folder = get_output_folder_from_input(uploaded_file.name)
            logger.info(f"New file uploaded with hash: {file_hash}")
            logger.info(f"Output folder set to: {st.session_state.output_folder}")
        
        # Only run analysis if not already completed for this file
        if not st.session_state.analysis_completed:
            try:
                # Get content from uploaded file
                file_content = uploaded_file.getvalue().decode()
                project_name = get_project_name_from_upload(uploaded_file)
                output_folder = f"output/{project_name}"
                
                logger.info(f"Processing uploaded file: {uploaded_file.name}")
                logger.info(f"Project name: {project_name}")
                logger.info(f"Output folder: {output_folder}")
                
                # Check existing files
                file_status = check_existing_files(output_folder)
                steps_to_run = get_steps_to_run(file_status)
                
                # If all steps are completed, skip to results
                if not any(steps_to_run.values()):
                    logger.info("All analysis steps already completed, skipping to results")
                    
                    # Add option to force re-run all steps
                    st.success("🎉 All analysis steps are already completed!")
                    col1, col2 = st.columns([2, 1])
                    with col2:
                        force_rerun = st.button("🔄 Force Re-run All", help="Re-execute all analysis steps even if files exist")
                    
                    if force_rerun:
                        st.info("🔄 Force re-running all analysis steps...")
                        # Override steps_to_run to force execution of all steps
                        steps_to_run = {
                            "Hotspot Analysis": True,
                            "ArXiv Paper Download": True,
                            "PDF Processing": True,
                            "Detailed Sustainability Report": True,
                            "Deep Hotspot Analysis": True,
                            # "Visualization Generation": True
                        }
                    else:
                        # Set session state and skip to results
                        st.session_state.analysis_completed = True
                        st.session_state.analysis_results = {"output_folder": output_folder}
                
                # Only run workflow if there are steps to execute
                if any(steps_to_run.values()):
                    # Initialize components only if needed
                    workflow = initialize_new_workflow()
                    logger.info("LCA workflow initialized successfully")
                
                    # Log which steps will be executed
                    steps_to_execute = [step for step, will_run in steps_to_run.items() if will_run]
                    steps_to_skip = [step for step, will_run in steps_to_run.items() if not will_run]
                    
                    if steps_to_execute:
                        logger.info(f"Steps to execute: {', '.join(steps_to_execute)}")
                    if steps_to_skip:
                        logger.info(f"Steps to skip (already completed): {', '.join(steps_to_skip)}")
                    
                    # Create step display
                    steps = {
                        "Hotspot Analysis": False,
                        "ArXiv Paper Download": False,
                        "PDF Processing": False,
                        "Detailed Sustainability Report": False,
                        "Deep Hotspot Analysis": False,
                        # "Visualization Generation": False
                    }
                    
                    # Check if we should run the complete workflow or partial workflow
                    if steps_to_run["Hotspot Analysis"]:
                        # Run complete workflow from input file
                        st.markdown("### Running Complete LCA Workflow")
                        st.markdown("This will execute 6 steps: Hotspot Analysis → ArXiv Download → PDF Processing → Detailed Sustainability Report → Deep Hotspot Analysis")
                        
                        with st.container():
                            # Save the input file content temporarily with proper naming
                            input_filename = uploaded_file.name if uploaded_file else "automotive_sample_input.txt"
                            temp_input_file = f"temp/{input_filename}"
                            Path("temp").mkdir(parents=True, exist_ok=True)
                            with open(temp_input_file, 'w', encoding='utf-8') as f:
                                f.write(file_content)
                            
                            # Execute complete workflow
                            workflow_outputs = workflow.execute_complete_workflow(temp_input_file)
                            
                            # Clean up temp file
                            Path(temp_input_file).unlink(missing_ok=True)
                            
                            # Update step status
                            for step in steps.keys():
                                steps[step] = True
                            
                            st.success("✓ Complete LCA workflow executed successfully!")
                            
                            # Display workflow summary
                            st.markdown("### Workflow Summary")
                            for output_type, file_path in workflow_outputs.items():
                                if Path(file_path).exists():
                                    st.markdown(f"✓ **{output_type.replace('_', ' ').title()}**: {file_path}")
                                else:
                                    st.markdown(f"⚠️ **{output_type.replace('_', ' ').title()}**: {file_path} (not found)")
                    else:
                        # Run partial workflow from existing hotspot analysis
                        st.markdown("### Running Partial LCA Workflow")
                        st.markdown("Hotspot analysis already exists. Running remaining steps...")
                        
                        # Load existing hotspot analysis
                        try:
                            hotspot_file = f"{output_folder}/hotspot_lca_analysis.json"
                            if Path(hotspot_file).exists():
                                workflow_outputs = workflow.execute_workflow_from_hotspot_analysis(hotspot_file)
                                
                                # Update step status
                                steps["Hotspot Analysis"] = True
                                for step in ["ArXiv Paper Download", "PDF Processing", "Detailed Sustainability Report", "Deep Hotspot Analysis"]: # , "Visualization Generation"]:
                                    steps[step] = True
                                
                                st.success("✓ Partial LCA workflow executed successfully!")
                            else:
                                st.error("Hotspot analysis file not found. Please run complete workflow.")
                                return
                        except Exception as e:
                            st.error(f"Error running partial workflow: {str(e)}")
                            logger.error(f"Error in partial workflow: {str(e)}")
                            return
                    
                    # Mark analysis as completed after successful execution
                    st.session_state.analysis_completed = True
                    st.session_state.analysis_results = {"output_folder": output_folder}
                    
                    # Display success message
                    st.success("Analysis completed successfully!")
                    logger.info("Analysis completed successfully")
                
            except Exception as e:
                error_msg = f"An error occurred during analysis: {str(e)}"
                st.error(error_msg)
                logger.error(error_msg)
                # Reset session state on error so user can try again
                st.session_state.analysis_completed = False
                st.session_state.analysis_results = None
        
        # Display results if analysis is completed
        if st.session_state.analysis_completed:
            # Get output folder from session state
            output_folder = st.session_state.analysis_results.get("output_folder", st.session_state.output_folder)
            
            # Check if files still exist (in case they were deleted)
            file_status = check_existing_files(output_folder)
            missing_files = [step for step, exists in file_status.items() if not exists]
            
            if missing_files:
                st.warning(f"⚠️ Some analysis files are missing: {', '.join(missing_files)}")
                st.info("Please upload your file again to regenerate missing analysis.")
                # Reset session state to force re-analysis
                st.session_state.analysis_completed = False
                st.session_state.analysis_results = None
            else:
                # Create tabs for different reports
                tab1, tab2, tab3 = st.tabs(["📊 Hotspot Analysis", "🌱 Sustainable Solutions", "📈 Visualization"])
                
                with tab1:
                    # Load and display hotspot analysis report
                    try:
                        with open(f"{output_folder}/hotspot_lca_analysis.json", 'r') as f:
                            hotspot_data = json.load(f)
                        format_hotspot_report(hotspot_data)
                    except FileNotFoundError:
                        st.error("Hotspot analysis file not found. Please run the analysis again.")
                
                with tab2:
                    # Load and display sustainable solutions report
                    try:
                        with open(f"{output_folder}/sustainable_solutions_report.txt", 'r', encoding='utf-8') as f:
                            solutions_report = f.read()
                        format_solutions_report(solutions_report)
                    except FileNotFoundError:
                        st.error("Sustainable solutions report file not found. Please run the analysis again.")
                
                with tab3:
                    # Create sub-tabs for different visualizations
                    viz_tab1, viz_tab2 = st.tabs(["📊 Hotspot Analysis Visualizations", "🌱 Sustainable Solutions Visualizations"])
                    
                    with viz_tab1:
                        st.subheader("Hotspot Analysis Visualizations")
                        # Load visualizations from files
                        lca_viz, _ = load_visualizations(output_folder)
                        
                        # Display visualizations
                        for name, html_content in lca_viz.items():
                            st.markdown(f"### {name.replace('_', ' ').title()}")
                            st.components.v1.html(
                                f'''
                                <div style="
                                    width: 100%;
                                    height: 800px;
                                    overflow: auto;
                                    border: 1px solid #ddd;
                                    border-radius: 5px;
                                    padding: 10px;
                                    margin: 10px 0;
                                ">
                                    {html_content}
                                </div>
                                ''',
                                height=800,
                                scrolling=True
                            )
                    
                    with viz_tab2:
                        st.subheader("Sustainable Solutions Visualizations")
                        # Load visualizations from files
                        _, solutions_viz = load_visualizations(output_folder)
                        
                        # Display visualizations
                        for name, html_content in solutions_viz.items():
                            st.markdown(f"### {name.replace('_', ' ').title()}")
                            st.components.v1.html(
                                f'''
                                <div style="
                                    width: 100%;
                                    height: 800px;
                                    overflow: auto;
                                    border: 1px solid #ddd;
                                    border-radius: 5px;
                                    padding: 10px;
                                    margin: 10px 0;
                                ">
                                    {html_content}
                                </div>
                                ''',
                                height=800,
                                scrolling=True
                            )
    else:
        # Reset session state when no file is uploaded, but preserve output folder for default
        if st.session_state.analysis_completed:
            st.session_state.analysis_completed = False
            st.session_state.current_file_hash = None
            st.session_state.analysis_results = None
            # Keep default output folder for potential analysis with default data
            st.session_state.output_folder = "output/automotive_sample"

def main_workflow():
    """Main execution function for the new LCA workflow (command-line usage)."""
    try:
        # Initialize workflow
        workflow = NewLCAWorkflow()
        
        # Example execution with default input file
        input_file = "automotive_sample_input.txt"
        
        if not Path(input_file).exists():
            logger.error(f"Input file '{input_file}' not found!")
            return
        
        # Execute complete workflow
        outputs = workflow.execute_complete_workflow(input_file)
        
        logger.info("=" * 80)
        logger.info("NEW LCA WORKFLOW COMPLETED SUCCESSFULLY!")
        logger.info("=" * 80)
        
        print("\nWorkflow Results:")
        print("=" * 50)
        for output_type, file_path in outputs.items():
            if Path(file_path).exists():
                print(f"[SUCCESS] {output_type.upper()}: {file_path}")
            else:
                print(f"[MISSING] {output_type.upper()}: {file_path} (not found)")
        
        print("\nNext Steps:")
        print("1. Review the hotspot analysis results")
        print("2. Examine the downloaded research papers")
        print("3. Read the sustainability solutions report")
        print("4. Implement the recommended solutions")
        
    except Exception as e:
        logger.error(f"Critical error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--workflow":
        # Run the workflow directly from command line
        main_workflow()
    else:
        # Run the Streamlit app
        main() 
