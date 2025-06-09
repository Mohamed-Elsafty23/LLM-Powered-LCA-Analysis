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

# Suppress Faiss GPU warning
warnings.filterwarnings('ignore', message='.*Failed to load GPU Faiss.*')

# Create a queue for log messages
log_queue = queue.Queue()

# Custom StreamHandler to capture logs
class StreamlitHandler(logging.Handler):
    def emit(self, record):
        try:
            msg = self.format(record)
            log_queue.put(msg)
        except Exception:
            self.handleError(record)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/app.log'),
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
            api_key=primary_key,
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

def save_uploaded_file(uploaded_file):
    """Save the uploaded file to a temporary location."""
    temp_file = Path("temp_input.txt")
    temp_file.write_text(uploaded_file.getvalue().decode())
    return str(temp_file)

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
            cols = st.columns(len(data))
            for col, (key, value) in zip(cols, data.items()):
                with col:
                    st.metric(
                        label=key.replace('_', ' ').title(),
                        value=value
                    )

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
                    for key, value in section_data["key_findings"].items():
                        st.markdown(f"**{format_key(key)}:** {value}")
                
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

def load_visualizations():
    """Load the latest visualizations for both LCA and sustainable solutions."""
    lca_viz = {}
    solutions_viz = {}
    
    # Load LCA visualizations from newest timestamped folder
    lca_dir = Path("visualizations/lca")
    if lca_dir.exists():
        # Get all timestamped directories and sort by name (newest first)
        timestamp_dirs = [d for d in lca_dir.iterdir() if d.is_dir() and all(part.isdigit() for part in d.name.split('_'))]
        if timestamp_dirs:
            newest_dir = sorted(timestamp_dirs, key=lambda x: x.name, reverse=True)[0]
            for html_file in newest_dir.glob("*.html"):
                with open(html_file, 'r', encoding='utf-8') as f:
                    lca_viz[html_file.stem] = f.read()
    
    # Load sustainable solutions visualizations from newest timestamped folder
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
            logger.info(f"New file uploaded with hash: {file_hash}")
        
        # Only run analysis if not already completed for this file
        if not st.session_state.analysis_completed:
            try:
                # Save uploaded file
                input_file = save_uploaded_file(uploaded_file)
                logger.info(f"File uploaded and saved: {input_file}")
                
                # Initialize components
                component_analyzer, lca_analyzer, solutions_generator = initialize_components()
                logger.info("Components initialized successfully")
                
                # Create step display
                steps = {
                    "Component Analysis": False,
                    "LCA Analysis": False,
                    "Sustainable Solutions": False,
                    "Visualization Generation": False
                }
                
                # Step 1: Component Analysis
                with st.container():
                    st.markdown("### Step 1/4: Component Analysis")
                    st.markdown("Analyzing components from the input file...")
                    component_results = component_analyzer.analyze_ecu_components(input_file)
                    component_analyzer.save_analysis(component_results, "output/component_analysis.json")
                    steps["Component Analysis"] = True
                    st.success("✓ Component analysis completed")
                
                # Step 2: LCA Analysis
                with st.container():
                    st.markdown("### Step 2/4: LCA Analysis")
                    st.markdown("Performing LCA analysis for each life cycle phase...")
                    lca_results = {}
                    phases = ['production', 'distribution', 'use', 'end_of_life']
                    
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
                    with open("output/llm_based_lca_analysis.json", 'w') as f:
                        json.dump({"lca_report": final_report}, f, indent=2)
                    steps["LCA Analysis"] = True
                    st.success("✓ LCA analysis completed")
                
                # Step 3: Generate Sustainable Solutions
                with st.container():
                    st.markdown("### Step 3/4: Sustainable Solutions")
                    st.markdown("Generating sustainable solutions based on LCA results...")
                    
                    # Generate sustainable solutions and get retrieved papers
                    retrieved_papers = solutions_generator.generate_sustainable_solutions(
                        lca_report_path="output/llm_based_lca_analysis.json",
                        output_path="output/sustainable_solutions_report.txt"
                    )
                    
                    # Store retrieved papers in output folder if they exist
                    if retrieved_papers:
                        try:
                            with open("output/retrieved_papers.json", 'w') as f:
                                json.dump(retrieved_papers, f, indent=2)
                            logger.info(f"Retrieved papers saved to output/retrieved_papers.json")
                        except Exception as e:
                            logger.error(f"Error saving retrieved papers: {str(e)}")
                    
                    steps["Sustainable Solutions"] = True
                    st.success("✓ Sustainable solutions generated")
                
                # Step 4: Generate Visualizations
                with st.container():
                    st.markdown("### Step 4/4: Visualization Generation")
                    st.markdown("Creating visualizations for LCA and sustainable solutions...")
                    
                    # Create LCA visualizations
                    try:
                        lca_visualizations, lca_saved_files = create_lca_visualizations("output/llm_based_lca_analysis.json")
                        logger.info("LCA visualizations created successfully")
                    except Exception as e:
                        logger.error(f"Error creating LCA visualizations: {str(e)}")
                        lca_visualizations = {}
                        lca_saved_files = []
                    
                    # Create sustainable solutions visualizations
                    try:
                        solutions_visualizations, solutions_saved_files = create_solutions_visualizations("output/sustainable_solutions_report.txt")
                        logger.info("Sustainable solutions visualizations created successfully")
                    except Exception as e:
                        logger.error(f"Error creating sustainable solutions visualizations: {str(e)}")
                        solutions_visualizations = {}
                        solutions_saved_files = []
                    
                    steps["Visualization Generation"] = True
                    st.success("✓ Visualizations generated")
                
                # Mark analysis as completed
                st.session_state.analysis_completed = True
                st.session_state.analysis_results = "completed"
                
                # Display success message
                st.success("Analysis completed successfully!")
                logger.info("Analysis completed successfully")
                
                # Clean up temporary file
                try:
                    Path(input_file).unlink(missing_ok=True)
                    logger.info("Temporary file cleaned up")
                except Exception as e:
                    logger.error(f"Error cleaning up temporary file: {str(e)}")
                
            except Exception as e:
                error_msg = f"An error occurred during analysis: {str(e)}"
                st.error(error_msg)
                logger.error(error_msg)
                # Reset session state on error so user can try again
                st.session_state.analysis_completed = False
                st.session_state.analysis_results = None
        
        # Display results if analysis is completed
        if st.session_state.analysis_completed:
            # Create tabs for different reports
            tab1, tab2, tab3 = st.tabs(["📊 LCA Report", "🌱 Sustainable Solutions", "📈 Visualization"])
            
            with tab1:
                # Load and display LCA report
                try:
                    with open("output/llm_based_lca_analysis.json", 'r') as f:
                        lca_data = json.load(f)
                    format_lca_report(lca_data)
                except FileNotFoundError:
                    st.error("LCA report file not found. Please run the analysis again.")
            
            with tab2:
                # Load and display sustainable solutions report
                try:
                    with open("output/sustainable_solutions_report.txt", 'r') as f:
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
                    lca_viz, _ = load_visualizations()
                    
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
                    _, solutions_viz = load_visualizations()
                    
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
        # Reset session state when no file is uploaded
        if st.session_state.analysis_completed:
            st.session_state.analysis_completed = False
            st.session_state.current_file_hash = None
            st.session_state.analysis_results = None

if __name__ == "__main__":
    main() 