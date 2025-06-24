import re
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from openai import OpenAI
import os
from dotenv import load_dotenv
import json
from pathlib import Path
import datetime
import time
import logging
from tenacity import retry, stop_after_attempt, wait_exponential
from config import PRIMARY_API_KEY, SECONDARY_API_KEY, BASE_URL

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SolutionsVisualizationManager:
    def __init__(self, api_keys=None, base_url=None):
        """Initialize the visualization manager with multiple API keys."""
        if api_keys is None:
            api_keys = [PRIMARY_API_KEY, SECONDARY_API_KEY]
        elif isinstance(api_keys, str):
            api_keys = [api_keys]
            
        if base_url is None:
            base_url = BASE_URL
            
        self.api_keys = [key for key in api_keys if key]  # Filter out None/empty keys
        self.base_url = base_url
        self.current_client_index = 0
        
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
            
        logger.info(f"Initialized SolutionsVisualizationManager with {len(self.clients)} API clients")

    def _get_next_client(self):
        """Get the next available client in rotation."""
        client = self.clients[self.current_client_index]
        self.current_client_index = (self.current_client_index + 1) % len(self.clients)
        return client

    def _make_api_request(self, messages, model="qwen2.5-coder-32b-instruct", **kwargs):
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
                    model=model,
                    **kwargs
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
        return self._make_api_request_with_retry(messages, model, **kwargs)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=4, max=20),
        reraise=True
    )
    def _make_api_request_with_retry(self, messages, model="qwen2.5-coder-32b-instruct", **kwargs):
        """Make API request with retry logic and client rotation."""
        client = self._get_next_client()
        client_index = (self.current_client_index - 1) % len(self.clients)
        
        try:
            logger.debug(f"Retry attempt with client {client_index + 1}/{len(self.clients)}")
            response = client.chat.completions.create(
                messages=messages,
                model=model,
                **kwargs
            )
            return response
        except Exception as e:
            logger.warning(f"Retry failed with client {client_index + 1}: {str(e)}")
            raise

# Create global instance
_viz_manager = None

def get_llm_client():
    """Initialize and return OpenAI client (legacy function for backward compatibility)."""
    global _viz_manager
    if _viz_manager is None:
        _viz_manager = SolutionsVisualizationManager()
    return _viz_manager.clients[0]  # Return first client for backward compatibility

def get_viz_manager():
    """Get or create the visualization manager."""
    global _viz_manager
    if _viz_manager is None:
        _viz_manager = SolutionsVisualizationManager()
    return _viz_manager

def analyze_solutions_report(text):
    """Use LLM to analyze the solutions report and suggest visualizations."""
    viz_manager = get_viz_manager()
    
    prompt = f"""
    Analyze this sustainable solutions report and suggest appropriate visualizations based ONLY on the data present in the report text below.
    
    CRITICAL INSTRUCTIONS - PREVENTING HALLUCINATION:
    - Only suggest visualizations for data points that are explicitly mentioned in the report text below
    - Do NOT suggest visualizations for data that might typically be in sustainability reports but is missing from this input
    - Do NOT assume the presence of standard sustainability metrics if they are not explicitly mentioned in the report
    - Base suggestions strictly on the actual content and structure of the provided report text
    
    For each visualization:
    1. Identify the relevant data points that are actually mentioned in the report text
    2. Suggest the best chart type based on the available data 
    3. Provide the data structure needed based on what is explicitly available in the text
    
    Report Text:
    {text}
    
    Return a JSON object with this structure:
    {{
        "visualizations": [
            {{
                "name": "string",
                "type": "string",
                "data_points": ["string"],
                "chart_type": "string",
                "description": "string"
            }}
        ]
    }}
    
    IMPORTANT: Your response must be valid JSON. Do not include any additional text or explanation.
    Only suggest visualizations for data that is explicitly mentioned in the report text above.
    """
    
    response = viz_manager._make_api_request(
        messages=[
            {"role": "system", "content": "You are a data visualization expert. Return only valid JSON."},
            {"role": "user", "content": prompt}
        ],
        model="qwen2.5-coder-32b-instruct",
        temperature=0.7,
        response_format={"type": "json_object"}
    )
    
    try:
        return json.loads(response.choices[0].message.content)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON response: {e}")
        print("Raw response:", response.choices[0].message.content)
        return {"visualizations": []}

def extract_data_for_visualization(text, visualization_spec):
    """Extract data for a specific visualization based on LLM suggestions."""
    viz_manager = get_viz_manager()
    
    prompt = f"""
    Extract the necessary data for this visualization from the solutions report provided below.
    
    CRITICAL INSTRUCTIONS - PREVENTING HALLUCINATION:
    - Extract ONLY data that is explicitly mentioned in the report text below
    - Do NOT add or assume any data that is not explicitly provided in the report
    - Do NOT fill in missing data with typical sustainability values
    - If the required data for the visualization is not present in the report text, return an empty object
    - Base extraction strictly on the actual content of the provided report text
    
    Visualization Spec:
    {json.dumps(visualization_spec, indent=2)}
    
    Report Text:
    {text}
    
    Return a JSON object with the extracted data in a format suitable for the specified chart type.
    Only include data that is explicitly mentioned in the report text above.
    
    IMPORTANT: Your response must be valid JSON. Do not include any additional text or explanation.
    """
    
    response = viz_manager._make_api_request(
        messages=[
            {"role": "system", "content": "You are a data extraction expert. Return only valid JSON."},
            {"role": "user", "content": prompt}
        ],
        model="qwen2.5-coder-32b-instruct",
        temperature=0.7,
        response_format={"type": "json_object"}
    )
    
    try:
        return json.loads(response.choices[0].message.content)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON response: {e}")
        print("Raw response:", response.choices[0].message.content)
        return {}

def create_visualization(extracted_data, visualization_spec):
    """Create a visualization based on the extracted data and specification."""
    viz_manager = get_viz_manager()
    
    # Determine chart type
    chart_type = visualization_spec.get('chart_type', 'bar').lower()
    if 'bar' in chart_type:
        chart_type = 'bar'
    elif 'pie' in chart_type:
        chart_type = 'pie'
    elif 'line' in chart_type:
        chart_type = 'line'
    else:
        chart_type = 'bar'  # default to bar chart
    
    prompt = f"""
    Create a Plotly visualization using this data and specification.
    
    Extracted Data:
    {json.dumps(extracted_data, indent=2)}
    
    Visualization Spec:
    {json.dumps(visualization_spec, indent=2)}
    
    Return ONLY the Python code that creates a Plotly figure using the data.
    The code should:
    1. Import necessary libraries
    2. Create the data structure
    3. Create the figure using plotly.express or plotly.graph_objects
    4. Assign the figure to a variable named 'fig'
    5. Choose an appropriate color scheme based on:
       - The data type (sequential for continuous data, categorical for discrete data)
       - The context of the data (e.g., environmental, economic, social)
       - Accessibility and distinguishability
       - Professional appearance
    6. Apply modern styling with:
       - White background
       - Grid lines (if applicable)
       - Proper margins
       - Hover templates (only for supported chart types)
       - Formatted numbers
    
    CRITICAL PLOTLY CHART TYPE RULES:
    - For Indicator/Gauge charts (go.Indicator): DO NOT use hovertemplate, update_traces with hovertemplate, or hover-related parameters
    - For Scatter, Bar, Line charts (px or go): hovertemplate is supported
    - For Pie charts: hovertemplate is supported
    - Only apply hover formatting to chart types that support it
    
    Example format for non-gauge charts:
    import plotly.express as px
    import pandas as pd

    # Create data
    data = {{'Category': ['A', 'B'], 'Value': [10, 20]}}
    df = pd.DataFrame(data)

    # Create figure
    fig = px.{chart_type}(df, x='Category', y='Value', title='{visualization_spec.get('name', 'Visualization')}')
    
    # Update layout
    fig.update_layout(
        title_x=0.5,
        title_font_size=20,
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(t=50, l=50, r=50, b=50),
        xaxis=dict(showgrid=True, gridcolor='#f0f0f0'),
        yaxis=dict(showgrid=True, gridcolor='#f0f0f0')
    )
    
    # Update traces (ONLY for charts that support hovertemplate)
    fig.update_traces(
        hovertemplate='%{{x}}: %{{y:,.2f}}<extra></extra>'
    )
    
    Example format for gauge/indicator charts:
    import plotly.graph_objects as go

    # Create data
    value = 15

    # Create figure
    fig = go.Figure(go.Indicator(
        domain = {{'x': [0, 1], 'y': [0, 1]}},
        value = value,
        mode = "gauge+number",
        title = {{'text': "Title"}},
        gauge = {{
            'axis': {{'range': [None, 25]}},
            'bar': {{'color': "#1f77b4"}},
            'steps': [...],
            'threshold': {{...}}
        }}
    ))

    # Update layout (DO NOT use update_traces with hovertemplate for Indicator charts)
    fig.update_layout(
        title_x=0.5,
        title_font_size=20,
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(t=100, l=50, r=50, b=50)
    )
    """
    
    response = viz_manager._make_api_request(
        messages=[
            {"role": "system", "content": "You are a Plotly visualization expert. Return only Python code without any markdown formatting or comments."},
            {"role": "user", "content": prompt}
        ],
        model="qwen2.5-coder-32b-instruct",
        temperature=0.7
    )
    
    try:
        # Get the code and clean it
        code = response.choices[0].message.content
        # Remove markdown formatting if present
        code = code.replace('```python', '').replace('```', '').strip()
        
        # Create a new namespace for execution with necessary imports
        namespace = {
            'px': px,
            'go': go,
            'pd': pd,
            'plotly': __import__('plotly'),
            'json': json
        }
        
        # Execute the code
        exec(code, namespace)
        
        # Get the figure from the namespace
        fig = namespace.get('fig')
        if fig is None:
            raise ValueError("No figure object found in the generated code")
        
        return fig
        
    except Exception as e:
        print(f"Error creating visualization: {e}")
        print("Generated code:", code)
        return None

def ensure_visualization_dir(output_folder=None):
    """Ensure the visualization directory exists in the project-specific output folder."""
    if output_folder:
        # Use project-specific folder structure
        viz_dir = Path(output_folder) / "visualizations" / "sustainable_solutions"
    else:
        # Fallback to default structure
        viz_dir = Path("visualizations") / "sustainable_solutions"
    
    # Create directory if it doesn't exist
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    # Create timestamped subdirectory for this run
    timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
    current_run_dir = viz_dir / timestamp
    current_run_dir.mkdir(exist_ok=True)
    
    # Clean up old directories (keep only 3 most recent)
    timestamp_dirs = [d for d in viz_dir.iterdir() if d.is_dir() and all(part.isdigit() for part in d.name.split('_'))]
    timestamp_dirs.sort(key=lambda x: x.name, reverse=True)
    for old_dir in timestamp_dirs[3:]:
        import shutil
        shutil.rmtree(old_dir)
    
    return current_run_dir

def save_visualization(fig, name, viz_dir):
    """Save a visualization to HTML and PNG files."""
    base_name = name
    
    # Save as HTML (interactive)
    html_path = viz_dir / f"{base_name}.html"
    fig.write_html(str(html_path))
    
    # Save as PNG (static)
    png_path = viz_dir / f"{base_name}.png"
    fig.write_image(str(png_path))
    
    return str(html_path), str(png_path)

def get_latest_visualizations(output_folder=None):
    """Get the latest visualizations from the specified output folder."""
    if output_folder:
        # Use project-specific folder structure
        viz_base_dir = Path(output_folder) / "visualizations" / "sustainable_solutions"
    else:
        # Fallback to default structure
        viz_base_dir = Path("visualizations") / "sustainable_solutions"
    
    if not viz_base_dir.exists():
        return {}
    
    # Get the most recent timestamped directory
    timestamp_dirs = [d for d in viz_base_dir.iterdir() if d.is_dir() and all(part.isdigit() for part in d.name.split('_'))]
    if not timestamp_dirs:
        return {}
    
    latest_dir = sorted(timestamp_dirs, key=lambda x: x.name, reverse=True)[0]
    
    visualizations = {}
    for html_file in latest_dir.glob("*.html"):
        try:
            with open(html_file, 'r', encoding='utf-8') as f:
                html_content = f.read()
            visualizations[html_file.stem] = html_content
        except Exception as e:
            print(f"Error loading visualization {html_file.name}: {e}")
    
    return visualizations

def create_all_visualizations(file_path="output/sustainable_solutions_report.txt"):
    """Create and return all visualizations based on LLM analysis."""
    try:
        # Extract output folder from file path
        file_path_obj = Path(file_path)
        if len(file_path_obj.parts) >= 2 and file_path_obj.parts[0] == "output":
            # If file is in output/project_name/ structure
            output_folder = file_path_obj.parent
        else:
            # Fallback to default
            output_folder = Path("output/automotive_sample")
        
        # Ensure visualization directory exists
        viz_dir = ensure_visualization_dir(str(output_folder))
        
        # Load solutions report
        with open(file_path, 'r') as f:
            text = f.read()
        
        # Get visualization suggestions from LLM
        visualization_specs = analyze_solutions_report(text)
        
        # Create visualizations
        visualizations = {}
        saved_files = []
        
        for spec in visualization_specs.get('visualizations', []):
            try:
                # Extract data for this visualization
                extracted_data = extract_data_for_visualization(text, spec)
                
                # Create the visualization
                fig = create_visualization(extracted_data, spec)
                
                if fig is not None:
                    # Store the visualization
                    visualizations[spec['name']] = fig
                    
                    # Save the visualization
                    html_path, png_path = save_visualization(fig, spec['name'], viz_dir)
                    saved_files.append({
                        'name': spec['name'],
                        'html_path': html_path,
                        'png_path': png_path
                    })
            except Exception as e:
                print(f"Error processing visualization {spec.get('name', 'unknown')}: {e}")
                continue
        
        # Save metadata about the visualizations
        metadata = {
            'timestamp': datetime.datetime.now().isoformat(),
            'source_file': file_path,
            'output_folder': str(output_folder),
            'visualizations': saved_files
        }
        
        with open(viz_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return visualizations, saved_files
    except Exception as e:
        print(f"Error in create_all_visualizations: {e}")
        return {}, []

if __name__ == "__main__":
    # Example usage
    visualizations, saved_files = create_all_visualizations()
    print("\nSaved visualizations:")
    for file_info in saved_files:
        print(f"\n{file_info['name']}:")
        print(f"  HTML: {file_info['html_path']}")
        print(f"  PNG: {file_info['png_path']}") 