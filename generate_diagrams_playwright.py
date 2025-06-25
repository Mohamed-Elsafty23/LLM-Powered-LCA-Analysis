"""
Advanced script to generate PNG images from Mermaid diagrams using Playwright
This script creates HTML files with mermaid.js and captures screenshots
"""

import re
import asyncio
from pathlib import Path
from playwright.async_api import async_playwright

def extract_mermaid_code(md_file_path):
    """Extract mermaid code from markdown file"""
    with open(md_file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find mermaid code block
    pattern = r'```mermaid\n(.*?)\n```'
    match = re.search(pattern, content, re.DOTALL)
    
    if match:
        return match.group(1).strip()
    else:
        print(f"No mermaid code found in {md_file_path}")
        return None

def create_html_template(mermaid_code, title):
    """Create HTML template with mermaid diagram"""
    html_template = f"""
<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
    <script src="https://cdn.jsdelivr.net/npm/mermaid@10.9.0/dist/mermaid.min.js"></script>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #ffffff;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
        }}
        .container {{
            text-align: center;
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }}
        h1 {{
            color: #333;
            margin-bottom: 20px;
            font-size: 24px;
        }}
        .mermaid {{
            margin: 20px auto;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{title}</h1>
        <div class="mermaid">
{mermaid_code}
        </div>
    </div>
    
    <script>
        mermaid.initialize({{ 
            startOnLoad: true, 
            theme: 'default',
            themeVariables: {{
                primaryColor: '#e1f5fe',
                primaryTextColor: '#333',
                primaryBorderColor: '#1976d2',
                lineColor: '#333',
                secondaryColor: '#f3e5f5',
                tertiaryColor: '#fff3e0'
            }}
        }});
    </script>
</body>
</html>
"""
    return html_template

async def generate_diagram_image(mermaid_code, output_path, title):
    """Generate PNG image from mermaid code using Playwright"""
    try:
        # Create HTML file
        html_content = create_html_template(mermaid_code, title)
        html_file = f"temp_{output_path.replace('.png', '.html')}"
        
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        # Launch browser and capture screenshot
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            
            # Load the HTML file
            await page.goto(f"file:///{Path.cwd()}/{html_file}")
            
            # Wait for mermaid to render
            await page.wait_for_timeout(3000)
            
            # Wait for the mermaid diagram to be rendered
            await page.wait_for_selector('.mermaid svg', timeout=10000)
            
            # Take screenshot of the container
            container = page.locator('.container')
            await container.screenshot(path=output_path, type='png')
            
            await browser.close()
        
        # Clean up HTML file
        Path(html_file).unlink()
        
        print(f"✅ Generated: {output_path}")
        return True
        
    except Exception as e:
        print(f"❌ Error generating {output_path}: {str(e)}")
        # Clean up HTML file if it exists
        html_file = f"temp_{output_path.replace('.png', '.html')}"
        if Path(html_file).exists():
            Path(html_file).unlink()
        return False

async def main():
    """Main function to generate all diagram images"""
    
    # Define input files and output names
    diagrams = [
        {
            'input': 'workflow_diagram_1_vector_database.md',
            'output': 'workflow_1_vector_database.png',
            'title': 'First Approach - Prepared Literature Papers Workflow'
        },
        {
            'input': 'workflow_diagram_2_hotspot_driven.md', 
            'output': 'workflow_2_hotspot_driven.png',
            'title': 'Second Approach - API-based Paper Download and Web Search Workflow (Current Implementation)'
        }
    ]
    
    print("🎨 Generating workflow diagram images with Playwright...")
    print("=" * 60)
    
    success_count = 0
    
    for diagram in diagrams:
        input_file = diagram['input']
        output_file = diagram['output']
        title = diagram['title']
        
        print(f"\n📊 Processing: {title}")
        print(f"   Input:  {input_file}")
        print(f"   Output: {output_file}")
        
        # Check if input file exists
        if not Path(input_file).exists():
            print(f"❌ Input file not found: {input_file}")
            continue
        
        # Extract mermaid code
        mermaid_code = extract_mermaid_code(input_file)
        
        if mermaid_code:
            # Generate image
            if await generate_diagram_image(mermaid_code, output_file, title):
                success_count += 1
        
    print("\n" + "=" * 60)
    print(f"🎉 Successfully generated {success_count}/{len(diagrams)} diagrams!")
    
    if success_count > 0:
        print("\n📁 Generated files:")
        for diagram in diagrams:
            output_file = diagram['output']
            if Path(output_file).exists():
                file_size = Path(output_file).stat().st_size / 1024  # KB
                print(f"   ✅ {output_file} ({file_size:.1f} KB)")

if __name__ == "__main__":
    asyncio.run(main()) 