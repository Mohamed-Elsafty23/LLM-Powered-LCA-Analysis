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

# Load environment variables
load_dotenv()

def get_llm_client():
    """Initialize and return OpenAI client."""
    return OpenAI(
        api_key=os.getenv("PRIMARY_API_KEY"),
        base_url=os.getenv("BASE_URL")
    )

def analyze_solutions_report(text):
    """Use LLM to analyze the solutions report and suggest visualizations."""
    client = get_llm_client()
    
    prompt = f"""
    Analyze this sustainable solutions report and suggest appropriate visualizations. For each visualization:
    1. Identify the relevant data points
    2. Suggest the best chart type
    3. Provide the data structure needed
    
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
    """
    
    response = client.chat.completions.create(
        model="qwen2.5-coder-32b-instruct",
        messages=[
            {"role": "system", "content": "You are a data visualization expert. Return only valid JSON."},
            {"role": "user", "content": prompt}
        ],
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
    client = get_llm_client()
    
    prompt = f"""
    Extract the necessary data for this visualization from the solutions report.
    
    Visualization Spec:
    {json.dumps(visualization_spec, indent=2)}
    
    Report Text:
    {text}
    
    Return a JSON object with the extracted data in a format suitable for the specified chart type.
    
    IMPORTANT: Your response must be valid JSON. Do not include any additional text or explanation.
    """
    
    response = client.chat.completions.create(
        model="qwen2.5-coder-32b-instruct",
        messages=[
            {"role": "system", "content": "You are a data extraction expert. Return only valid JSON."},
            {"role": "user", "content": prompt}
        ],
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
    client = get_llm_client()
    
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
       - Grid lines
       - Proper margins
       - Hover templates
       - Formatted numbers
    
    Example format:
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
        showlegend=True,
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(t=50, l=50, r=50, b=50),
        xaxis=dict(showgrid=True, gridcolor='#f0f0f0'),
        yaxis=dict(showgrid=True, gridcolor='#f0f0f0')
    )
    
    # Update traces
    fig.update_traces(
        hovertemplate='%{{x}}: %{{y:,.2f}}<extra></extra>'
    )
    """
    
    response = client.chat.completions.create(
        model="qwen2.5-coder-32b-instruct",
        messages=[
            {"role": "system", "content": "You are a Plotly visualization expert. Return only Python code without any markdown formatting or comments."},
            {"role": "user", "content": prompt}
        ],
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

def ensure_visualization_dir():
    """Ensure the visualization directory exists and maintain only three most recent versions."""
    base_dir = Path("visualizations")
    solutions_dir = base_dir / "sustainable_solutions"
    
    # Create base directory if it doesn't exist
    solutions_dir.mkdir(parents=True, exist_ok=True)
    
    # Create timestamped directory for this run (add microseconds for uniqueness)
    timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
    current_run_dir = solutions_dir / timestamp
    current_run_dir.mkdir(exist_ok=True)
    
    # Re-list all timestamped directories after creating the new one
    timestamp_dirs = [d for d in solutions_dir.iterdir() if d.is_dir() and all(part.isdigit() for part in d.name.split('_'))]
    # Sort directories by name (newest first)
    timestamp_dirs.sort(key=lambda x: x.name, reverse=True)
    # Remove old directories if more than 3 exist
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

def get_latest_visualizations():
    """Get the latest visualizations from the current directory."""
    current_dir = Path("visualizations/sustainable_solutions")
    if not current_dir.exists():
        return {}
    
    visualizations = {}
    for html_file in current_dir.glob("*.html"):
        try:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=[0], y=[0], mode='markers'))
            fig.update_layout(title=html_file.stem)
            visualizations[html_file.stem] = fig
        except Exception as e:
            print(f"Error loading visualization {html_file.name}: {e}")
    
    return visualizations

def create_all_visualizations(file_path="output/sustainable_solutions_report.txt"):
    """Create and return all visualizations based on LLM analysis."""
    try:
        # Ensure visualization directory exists
        viz_dir = ensure_visualization_dir()
        
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