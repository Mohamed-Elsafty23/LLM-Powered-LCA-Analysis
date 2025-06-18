import streamlit as st
import json
import time
from pathlib import Path
import logging
import warnings
from component_analyzer import ComponentAnalyzer
from lca_implementation import LLMBasedLCAAnalyzer
from sustainable_solutions_generator import SustainableSolutionsGenerator
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

def create_impact_chart(lca_results):
    """Create an interactive chart for LCA impacts."""
    # Extract impact data
    impact_data = []
    for phase, data in lca_results.items():
        if isinstance(data, dict) and 'total_impact' in data:
            impact_data.append({
                'Phase': phase.replace('_', ' ').title(),
                'Impact': data['total_impact']
            })
    
    if impact_data:
        df = pd.DataFrame(impact_data)
        fig = px.bar(df, x='Phase', y='Impact', 
                    title='Environmental Impact by Life Cycle Phase',
                    labels={'Impact': 'Environmental Impact', 'Phase': 'Life Cycle Phase'})
        return fig
    return None

def initialize_components():
    """Initialize all required components for the analysis."""
    try:
        # Get API keys from either source
        primary_key, secondary_key, base_url = get_api_keys()
        api_keys = [primary_key, secondary_key]
        
        # Initialize component analyzer
        component_analyzer = ComponentAnalyzer(
            api_keys=api_keys,
            base_url=base_url
        )
        
        # Initialize LCA analyzer
        lca_analyzer = LLMBasedLCAAnalyzer(
            api_keys=api_keys,
            base_url=base_url
        )
        
        # Initialize sustainable solutions generator
        api_configs = [
            {
                "api_key": primary_key,
                "base_url": base_url,
                "model": "llama-3.3-70b-instruct"
            },
            {
                "api_key": secondary_key,
                "base_url": base_url,
                "model": "llama-3.3-70b-instruct"
            }
        ]
        
        solutions_generator = SustainableSolutionsGenerator(
            vector_db_path="vector_db",
            api_configs=api_configs
        )
        
        return component_analyzer, lca_analyzer, solutions_generator
    except Exception as e:
        logger.error(f"Error initializing components: {str(e)}")
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

def format_lca_report(report):
    """Format the LCA report for better display using the approach from test_lca.py."""
    def format_key(key):
        """Format key for display by replacing underscores with spaces and capitalizing."""
        return key.replace('_', ' ').title()

    def display_value(value):
        """Display a value in an appropriate format."""
        if isinstance(value, (int, float)):
            return f"{value:,.2f}"
        elif isinstance(value, str):
            return value
        elif isinstance(value, dict):
            return None  # Will be handled by display_section
        elif isinstance(value, list):
            return None  # Will be handled by display_section
        return str(value)

    def display_section(data, level=1):
        """Display a section of data with proper formatting."""
        if isinstance(data, dict):
            for key, value in data.items():
                formatted_key = format_key(key)
                
                if isinstance(value, dict):
                    st.markdown(f"{'#' * level} {formatted_key}")
                    display_section(value, level + 1)
                elif isinstance(value, list):
                    st.markdown(f"{'#' * level} {formatted_key}")
                    for item in value:
                        if isinstance(item, dict):
                            display_section(item, level + 1)
                        else:
                            st.markdown(f"- {item}")
                else:
                    displayed_value = display_value(value)
                    if displayed_value is not None:
                        st.markdown(f"**{formatted_key}:** {displayed_value}")

    def display_metrics(data):
        """Display metrics in a visually appealing way."""
        if isinstance(data, dict):
            # Flatten nested dictionaries
            flat_data = {}
            def flatten_dict(d, prefix=''):
                for key, value in d.items():
                    new_key = f"{prefix}{key}" if prefix else key
                    if isinstance(value, dict):
                        flatten_dict(value, f"{new_key}_")
                    else:
                        flat_data[new_key] = value
            
            flatten_dict(data)
            
            # Create columns for the flattened data
            cols = st.columns(len(flat_data))
            for col, (key, value) in zip(cols, flat_data.items()):
                with col:
                    # Format the key for display
                    display_key = key.replace('_', ' ').title()
                    # Ensure value is a simple type
                    if isinstance(value, (int, float)):
                        st.metric(label=display_key, value=f"{value:,.2f}")
                    else:
                        st.metric(label=display_key, value=str(value))

    # Get the main report data
    report_data = report.get("lca_report", report)
    
    # Create tabs dynamically based on the top-level keys
    tabs = st.tabs([format_key(key) for key in report_data.keys()])
    
    # Display content in tabs
    for tab, (section_name, section_data) in zip(tabs, report_data.items()):
        with tab:
            st.header(format_key(section_name))
            
            # Special handling for executive summary if it exists
            if section_name == "executive_summary" and isinstance(section_data, dict):
                # Display overview if it exists
                if "overview" in section_data:
                    st.markdown("### Overview")
                    st.write(section_data["overview"])
                
                # Display key findings if they exist
                if "key_findings" in section_data:
                    st.markdown("### Key Findings")
                    key_findings = section_data["key_findings"]
                    if isinstance(key_findings, dict):
                        for key, value in key_findings.items():
                            st.markdown(f"**{format_key(key)}:** {value}")
                    elif isinstance(key_findings, list):
                        for item in key_findings:
                            if isinstance(item, str):
                                st.markdown(f"• {item}")
                            elif isinstance(item, dict):
                                for key, value in item.items():
                                    st.markdown(f"**{format_key(key)}:** {value}")
                    else:
                        st.write(key_findings)
                
                # Display percentage breakdown if it exists
                if "percentage_breakdown" in section_data:
                    st.markdown("### Percentage Breakdown")
                    display_metrics(section_data["percentage_breakdown"])
                
                # Display conclusion if it exists
                if "conclusion" in section_data:
                    st.markdown("### Conclusion")
                    st.write(section_data["conclusion"])
            else:
                # Generic display for other sections
                display_section(section_data)

def format_solutions_report(report_text):
    """Format the sustainable solutions report using the approach from test_sus.py."""
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
    """Load the latest visualizations for both LCA and sustainable solutions from project-specific folders."""
    lca_viz = {}
    solutions_viz = {}
    
    if output_folder:
        # Load LCA visualizations from project-specific folder
        lca_dir = Path(output_folder) / "visualizations" / "lca"
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
        # Load LCA visualizations from global folder
        lca_dir = Path("visualizations/lca")
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
    Check which analysis files already exist in the output folder.
    
    Args:
        output_folder: Path to the project output folder
        
    Returns:
        Dict: Status of each analysis step
    """
    output_path = Path(output_folder)
    
    # Define required files for each step
    required_files = {
        "component_analysis": output_path / "component_analysis.json",
        "lca_analysis": output_path / "llm_based_lca_analysis.json", 
        "sustainable_solutions": output_path / "sustainable_solutions_report.txt",
        "retrieved_papers": output_path / "retrieved_papers.json",
        "lca_visualizations": output_path / "visualizations" / "lca",
        "solutions_visualizations": output_path / "visualizations" / "sustainable_solutions"
    }
    
    # Check which files exist and are not empty
    file_status = {}
    for step, file_path in required_files.items():
        if step.endswith("_visualizations"):
            # For visualization folders, check if they exist and contain files
            exists = file_path.exists() and any(file_path.iterdir()) if file_path.exists() else False
        else:
            # For regular files, check if they exist and are not empty
            exists = file_path.exists() and file_path.stat().st_size > 0 if file_path.exists() else False
        file_status[step] = exists
    
    return file_status

def get_steps_to_run(file_status: Dict[str, bool]) -> Dict[str, bool]:
    """
    Determine which steps need to be run based on existing files.
    
    Args:
        file_status: Dictionary of file existence status
        
    Returns:
        Dict: Which steps should be executed
    """
    # Check if component analysis and LCA analysis files exist
    component_analysis_exists = file_status["component_analysis"]
    lca_analysis_exists = file_status["lca_analysis"]
    
    # If both component analysis and LCA analysis exist, skip to sustainable solutions
    if component_analysis_exists and lca_analysis_exists:
        steps_to_run = {
            "Component Analysis": False,
            "LCA Analysis": False,
            "Sustainable Solutions": not file_status["sustainable_solutions"],
            "Visualization Generation": not (file_status["lca_visualizations"] and file_status["solutions_visualizations"])
        }
    else:
        # If either file is missing, run all steps
        steps_to_run = {
            "Component Analysis": not component_analysis_exists,
            "LCA Analysis": not lca_analysis_exists,
            "Sustainable Solutions": not file_status["sustainable_solutions"],
            "Visualization Generation": not (file_status["lca_visualizations"] and file_status["solutions_visualizations"])
        }
    
    return steps_to_run

def main():
    # Set page config with custom theme
    st.set_page_config(
        page_title="LCA Analysis Tool",
        page_icon="🌍",
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
    
    st.title("🌍 LCA Analysis Tool")
    st.markdown("""
    This tool performs Life Cycle Assessment (LCA) analysis using LLM-based components.
    Upload a text file containing component data to begin the analysis.
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
                    
                    if not force_rerun:
                        st.session_state.analysis_completed = True
                        st.session_state.analysis_results = {"output_folder": output_folder}
                        # Skip to results display
                        st.rerun()
                        return
                    else:
                        st.info("🔄 Force re-running all analysis steps...")
                        # Override steps_to_run to force execution of all steps
                        steps_to_run = {
                            "Component Analysis": True,
                            "LCA Analysis": True,
                            "Sustainable Solutions": True,
                            "Visualization Generation": True
                        }
                
                # Initialize components only if needed
                if any(steps_to_run.values()):
                    component_analyzer, lca_analyzer, solutions_generator = initialize_components()
                    logger.info("Components initialized successfully")
                
                # Log which steps will be executed
                steps_to_execute = [step for step, will_run in steps_to_run.items() if will_run]
                steps_to_skip = [step for step, will_run in steps_to_run.items() if not will_run]
                
                if steps_to_execute:
                    logger.info(f"Steps to execute: {', '.join(steps_to_execute)}")
                if steps_to_skip:
                    logger.info(f"Steps to skip (already completed): {', '.join(steps_to_skip)}")
                
                # Create step display
                steps = {
                    "Component Analysis": False,
                    "LCA Analysis": False,
                    "Sustainable Solutions": False,
                    "Visualization Generation": False
                }
                
                # Step 1: Component Analysis
                if steps_to_run["Component Analysis"]:
                    with st.container():
                        st.markdown("### Step 1/4: Component Analysis")
                        st.markdown("Analyzing components from the input file...")
                        component_results = component_analyzer.analyze_ecu_components_from_content(file_content, project_name)
                        component_analyzer.save_analysis(component_results, f"{output_folder}/component_analysis.json")
                        steps["Component Analysis"] = True
                        st.success("✓ Component analysis completed")
                else:
                    st.markdown("### Step 1/4: Component Analysis")
                    st.success("✓ Component analysis already exists - skipping")
                    # Load existing component results for later steps
                    try:
                        with open(f"{output_folder}/component_analysis.json", 'r') as f:
                            component_results = json.load(f)
                        steps["Component Analysis"] = True
                    except Exception as e:
                        st.error(f"Error loading existing component analysis: {str(e)}")
                        logger.error(f"Error loading component analysis: {str(e)}")
                        return
                
                # Step 2: LCA Analysis
                if steps_to_run["LCA Analysis"]:
                    with st.container():
                        st.markdown("### Step 2/4: LCA Analysis")
                        st.markdown("Performing LCA analysis for each life cycle phase...")
                        lca_results = {}
                        phases = ['production', 'distribution', 'use', 'end_of_life']
                        
                        # Set output folder for LCA analyzer
                        lca_analyzer.output_folder = output_folder
                        
                        for phase in phases:
                            st.markdown(f"Analyzing {phase} phase...")
                            if phase == 'production':
                                lca_results[phase] = lca_analyzer.analyze_production_phase(component_results)
                            elif phase == 'distribution':
                                lca_results[phase] = lca_analyzer.analyze_distribution_phase(component_results)
                            elif phase == 'use':
                                lca_results[phase] = lca_analyzer.analyze_use_phase(component_results)
                            elif phase == 'end_of_life':
                                lca_results[phase] = lca_analyzer.analyze_end_of_life_phase(component_results)
                        
                        final_report = lca_analyzer.generate_comprehensive_report(lca_results, component_results)
                        with open(f"{output_folder}/llm_based_lca_analysis.json", 'w') as f:
                            json.dump({"lca_report": final_report}, f, indent=2)
                        steps["LCA Analysis"] = True
                        st.success("✓ LCA analysis completed")
                else:
                    st.markdown("### Step 2/4: LCA Analysis")
                    st.success("✓ LCA analysis already exists - skipping")
                    steps["LCA Analysis"] = True
                
                # Step 3: Generate Sustainable Solutions
                if steps_to_run["Sustainable Solutions"]:
                    with st.container():
                        st.markdown("### Step 3/4: Sustainable Solutions")
                        st.markdown("Generating sustainable solutions based on LCA results...")
                        
                        # Generate sustainable solutions and get retrieved papers
                        retrieved_papers = solutions_generator.generate_sustainable_solutions(
                            lca_report_path=f"{output_folder}/llm_based_lca_analysis.json",
                            output_path=f"{output_folder}/sustainable_solutions_report.txt"
                        )
                        
                        steps["Sustainable Solutions"] = True
                        st.success("✓ Sustainable solutions generated")
                else:
                    st.markdown("### Step 3/4: Sustainable Solutions")
                    st.success("✓ Sustainable solutions already exist - skipping")
                    steps["Sustainable Solutions"] = True
                
                # Step 4: Generate Visualizations
                if steps_to_run["Visualization Generation"]:
                    with st.container():
                        st.markdown("### Step 4/4: Visualization Generation")
                        st.markdown("Creating visualizations for LCA and sustainable solutions...")
                        
                        # Create LCA visualizations
                        try:
                            lca_visualizations, lca_saved_files = create_lca_visualizations(f"{output_folder}/llm_based_lca_analysis.json")
                            logger.info("LCA visualizations created successfully")
                        except Exception as e:
                            logger.error(f"Error creating LCA visualizations: {str(e)}")
                            lca_visualizations = {}
                            lca_saved_files = []
                        
                        # Create sustainable solutions visualizations
                        try:
                            solutions_visualizations, solutions_saved_files = create_solutions_visualizations(f"{output_folder}/sustainable_solutions_report.txt")
                            logger.info("Sustainable solutions visualizations created successfully")
                        except Exception as e:
                            logger.error(f"Error creating sustainable solutions visualizations: {str(e)}")
                            solutions_visualizations = {}
                            solutions_saved_files = []
                        
                        steps["Visualization Generation"] = True
                        st.success("✓ Visualizations generated")
                else:
                    st.markdown("### Step 4/4: Visualization Generation")
                    st.success("✓ Visualizations already exist - skipping")
                    steps["Visualization Generation"] = True
                
                # Mark analysis as completed
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
                st.rerun()
                return
            
            # Create tabs for different reports
            tab1, tab2, tab3 = st.tabs(["📊 LCA Report", "🌱 Sustainable Solutions", "📈 Visualization"])
            
            with tab1:
                # Load and display LCA report
                try:
                    with open(f"{output_folder}/llm_based_lca_analysis.json", 'r') as f:
                        lca_data = json.load(f)
                    format_lca_report(lca_data)
                except FileNotFoundError:
                    st.error("LCA report file not found. Please run the analysis again.")
            
            with tab2:
                # Load and display sustainable solutions report
                try:
                    with open(f"{output_folder}/sustainable_solutions_report.txt", 'r') as f:
                        solutions_report = f.read()
                    format_solutions_report(solutions_report)
                except FileNotFoundError:
                    st.error("Sustainable solutions report file not found. Please run the analysis again.")
            
            with tab3:
                # Create sub-tabs for different visualizations
                viz_tab1, viz_tab2 = st.tabs(["📊 LCA Visualizations", "🌱 Sustainable Solutions Visualizations"])
                
                with viz_tab1:
                    st.subheader("LCA Analysis Visualizations")
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

if __name__ == "__main__":
    main() 