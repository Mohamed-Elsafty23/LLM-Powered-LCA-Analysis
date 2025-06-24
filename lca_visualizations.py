import json
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from openai import OpenAI
import os
from dotenv import load_dotenv
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

class HotspotLCAVisualizationManager:
    def __init__(self, api_keys=None, base_url=None):
        """Initialize the hotspot LCA visualization manager with multiple API keys."""
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
            
        logger.info(f"Initialized HotspotLCAVisualizationManager with {len(self.clients)} API clients")

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
        _viz_manager = HotspotLCAVisualizationManager()
    return _viz_manager.clients[0]  # Return first client for backward compatibility

def get_viz_manager():
    """Get or create the visualization manager."""
    global _viz_manager
    if _viz_manager is None:
        _viz_manager = HotspotLCAVisualizationManager()
    return _viz_manager

def analyze_hotspot_data(hotspot_data):
    """Use LLM to analyze hotspot LCA data and suggest visualizations."""
    viz_manager = get_viz_manager()
    
    prompt = f"""
    Analyze this hotspot LCA data and suggest appropriate visualizations based ONLY on the data present in the input.
    
    CRITICAL INSTRUCTIONS - PREVENTING HALLUCINATION:
    - Only suggest visualizations for data fields that are explicitly present in the hotspot data below
    - Do NOT suggest visualizations for data that might typically be in LCA reports but is missing from this input
    - Do NOT assume the presence of standard LCA categories if they are not in the provided data
    - Base suggestions strictly on the actual structure and content of the provided hotspot data
    
    For each visualization:
    1. Identify the relevant data fields that are actually present in the input data
    2. Suggest the best chart type based on the available data structure
    3. Provide the data structure needed based on what is actually available
    
    Hotspot LCA Data:
    {json.dumps(hotspot_data, indent=2)}
    
    Return a JSON object with this structure:
    {{
        "visualizations": [
            {{
                "name": "string",
                "type": "string", 
                "data_fields": ["string"],
                "chart_type": "string",
                "description": "string"
            }}
        ]
    }}
    
    IMPORTANT: Your response must be valid JSON. Do not include any additional text or explanation.
    Only suggest visualizations for data that is explicitly present in the input above.
    Focus on hotspot-specific visualizations like hotspot rankings, environmental significance comparisons, etc.
    """
    
    response = viz_manager._make_api_request(
        messages=[
            {"role": "system", "content": "You are a data visualization expert specializing in environmental hotspot analysis. Return only valid JSON."},
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

def extract_data_for_visualization(hotspot_data, visualization_spec):
    """Extract data for a specific visualization based on LLM suggestions."""
    viz_manager = get_viz_manager()
    
    prompt = f"""
    Extract the necessary data for this visualization from the hotspot LCA data provided below.
    
    CRITICAL INSTRUCTIONS - PREVENTING HALLUCINATION:
    - Extract ONLY data that is explicitly present in the hotspot LCA data below
    - Do NOT add or assume any data that is not explicitly provided
    - Do NOT fill in missing data with typical LCA values
    - If the required data for the visualization is not present, return an empty object
    - Base extraction strictly on the actual content of the provided hotspot LCA data
    
    Visualization Spec:
    {json.dumps(visualization_spec, indent=2)}
    
    Hotspot LCA Data:
    {json.dumps(hotspot_data, indent=2)}
    
    Return a JSON object with the extracted data in a format suitable for the specified chart type.
    Only include data that is explicitly present in the hotspot LCA data above.
    Focus on hotspot-specific data like environmental significance levels, hotspot rankings, etc.
    
    IMPORTANT: Your response must be valid JSON. Do not include any additional text or explanation.
    """
    
    response = viz_manager._make_api_request(
        messages=[
            {"role": "system", "content": "You are a data extraction expert specializing in environmental hotspot data. Return only valid JSON."},
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

def create_visualization(data, viz_type, name):
    """Create a visualization using LLM."""
    viz_manager = get_viz_manager()
    code = None  # Initialize code variable
    try:
        
        # Map visualization type to chart type
        chart_type = viz_type.lower()
        if 'bar' in chart_type:
            chart_type = 'bar'
        elif 'pie' in chart_type:
            chart_type = 'pie'
        elif 'line' in chart_type:
            chart_type = 'line'
        elif 'scatter' in chart_type:
            chart_type = 'scatter'
        else:
            chart_type = 'bar'  # default to bar chart
        
        # Prepare the prompt for the LLM
        prompt = f"""Create a {chart_type} chart visualization for the following hotspot LCA data:

Data: {json.dumps(data, indent=2)}

REQUIREMENTS:
1. Return ONLY executable Python code that creates a Plotly visualization
2. DO NOT return function definitions or JSON
3. The code must be directly executable and create a Plotly figure
4. The code must:
   - Import necessary libraries (plotly.express or plotly.graph_objects)
   - Create a pandas DataFrame or dictionary with the data
   - Create a Plotly figure using the data
   - Assign the figure to a variable named 'fig'
5. For hotspot visualizations:
   - Use appropriate colors for environmental significance (red for high, yellow for medium, green for low)
   - Include hotspot rankings and environmental impact categories
   - Show life cycle phases where hotspots occur
6. For bar charts:
   - Use px.bar() with proper x and y values
   - Include title and labels
   - Add hover template with value formatting
7. For pie charts:
   - Use px.pie() with proper values and names
   - Include title and labels
   - Add hover template with percentage formatting
8. For scatter plots:
   - Use px.scatter() with proper x and y values
   - Include title and labels
   - Add hover template with value formatting

9. For all charts:
   - Use a clean, modern style
   - Add proper margins and padding
   - Use a white background
   - Add grid lines for better readability (if applicable)
   - Format numbers appropriately
   - Add hover information (only for supported chart types)
   - Choose colors that are:
     * Appropriate for environmental hotspot data
     * Accessible and distinguishable
     * Professional and visually appealing
     * Contextually relevant (e.g., red/yellow/green for significance levels)

CRITICAL PLOTLY CHART TYPE RULES:
- For Indicator/Gauge charts (go.Indicator): DO NOT use hovertemplate, update_traces with hovertemplate, or hover-related parameters
- For Scatter, Bar, Line charts (px or go): hovertemplate is supported
- For Pie charts: hovertemplate is supported
- Only apply hover formatting to chart types that support it

Example of correct executable code for standard charts:
import plotly.express as px
import pandas as pd

# Create data
data = {{'Hotspot': ['A', 'B'], 'Significance': ['high', 'low']}}
df = pd.DataFrame(data)

# Create figure
fig = px.{chart_type}(df, x='Hotspot', y='Significance', title='{name}')
fig.update_layout(
    title_x=0.5,
    title_font_size=20,
    plot_bgcolor='white',
    paper_bgcolor='white',
    margin=dict(t=50, l=50, r=50, b=50),
    xaxis=dict(showgrid=True, gridcolor='#f0f0f0'),
    yaxis=dict(showgrid=True, gridcolor='#f0f0f0')
)
# ONLY add update_traces for charts that support hovertemplate
fig.update_traces(
    hovertemplate='%{{x}}: %{{y}}<extra></extra>'
)

Example for gauge/indicator charts:
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

Return ONLY the executable Python code without any function definitions, markdown formatting, comments, or explanations."""

        # Get response from LLM
        response = viz_manager._make_api_request(
            messages=[
                {"role": "system", "content": "You are a data visualization expert specializing in environmental hotspot visualizations. Return only executable Python code that creates a Plotly visualization. Do not return function definitions or JSON."},
                {"role": "user", "content": prompt}
            ],
            model="llama-3.3-70b-instruct",
            temperature=0.1
        )
        
        # Extract and clean the code
        code = response.choices[0].message.content.strip()
        # Remove any markdown formatting if present
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
        print(f"Error creating visualization: {str(e)}")
        if code:
            print(f"Generated code: {code}")
        raise

def ensure_visualization_dir(output_folder=None):
    """Ensure the visualization directory exists in the project-specific output folder."""
    if output_folder:
        # Use project-specific folder structure
        viz_dir = Path(output_folder) / "visualizations" / "hotspot_lca"
    else:
        # Fallback to default structure
        viz_dir = Path("visualizations") / "hotspot_lca"
    
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
        viz_base_dir = Path(output_folder) / "visualizations" / "hotspot_lca"
    else:
        # Fallback to default structure
        viz_base_dir = Path("visualizations") / "hotspot_lca"
    
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

def create_all_visualizations(file_path="output/hotspot_lca_analysis.json"):
    """Create and return all visualizations based on hotspot LCA analysis."""
    try:
        # Extract output folder from file path
        file_path_obj = Path(file_path)
        if len(file_path_obj.parts) >= 2 and file_path_obj.parts[0] == "output":
            # If file is in output/project_name/ structure
            output_folder = file_path_obj.parent
        else:
            # Fallback to default
            output_folder = Path("output/automotive_sample_input")
        
        # Ensure visualization directory exists
        viz_dir = ensure_visualization_dir(str(output_folder))
        
        # Load hotspot LCA data
        with open(file_path, 'r') as f:
            data = json.load(f)
        hotspot_data = data.get('hotspot_analysis', {})
        
        # Get visualization suggestions from LLM
        visualization_specs = analyze_hotspot_data(hotspot_data)
        
        # Create visualizations
        visualizations = {}
        saved_files = []
        
        for spec in visualization_specs.get('visualizations', []):
            try:
                # Extract data for this visualization
                extracted_data = extract_data_for_visualization(hotspot_data, spec)
                
                # Create the visualization
                fig = create_visualization(extracted_data, spec['type'], spec['name'])
                
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
    print("\nSaved hotspot LCA visualizations:")
    for file_info in saved_files:
        print(f"\n{file_info['name']}:")
        print(f"  HTML: {file_info['html_path']}")
        print(f"  PNG: {file_info['png_path']}") 