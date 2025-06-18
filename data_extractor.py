from openai import OpenAI
import json
from typing import Dict, Any, Optional
from pathlib import Path
import logging
from datetime import datetime
import time
import asyncio
from openai import AsyncOpenAI
from config import PRIMARY_API_KEY, SECONDARY_API_KEY, BASE_URL

class DataExtractor:
    def __init__(self, api_key: str = PRIMARY_API_KEY, base_url: str = BASE_URL):
        self.extraction_client = AsyncOpenAI(
            api_key=PRIMARY_API_KEY,
            base_url=base_url
        )
        self.cleanup_client = AsyncOpenAI(
            api_key=SECONDARY_API_KEY,  # Different API key for cleanup
            base_url=base_url
        )
        self.model = 'llama-3.3-70b-instruct' #'meta-llama-3.1-8b-instruct'
        self.output_file = "output/extracted_data.json"
        self.system_role = """You are an expert academic paper analyzer with expertise in technical, environmental, and sustainability aspects. Your task is to extract ALL relevant information from academic papers, with special attention to:
1. Capturing all numerical data, parameters, and calculations with their context
2. Identifying environmental and sustainability aspects in detail
3. Documenting methodological details and assumptions thoroughly
4. Including units, uncertainty values, and confidence intervals
5. Preserving context and relationships between different pieces of information
6. Technical specifications and performance metrics with full context
7. Economic and market-related information with supporting data
8. Any sustainability or environmental considerations with detailed analysis
9. Any other significant information that could be relevant for analysis

Present all information in clear, detailed sentences rather than paragraphs. Focus on precision and clarity in descriptions. Return the data in a clean JSON format with consistent structure. Include ALL information that could be relevant for technical analysis, environmental assessment, or sustainability studies, even if it seems minor or indirect. Use consistent naming conventions and omit sections where all fields are empty.
            
IMPORTANT: 
- Never use "not specified" or similar placeholder text
- Only include fields where actual information is present in the paper
- If a section has no available information, omit the entire section
- For nested objects, only include fields that have actual data
- Do not create empty or placeholder fields"""

        self.combined_prompt = """Analyze this academic paper and extract ALL relevant information, with special attention to Life Cycle Assessment (LCA) and sustainability aspects. Extract:

1. Paper metadata:
   - title (full title with subtitle if any)
   - DOI (if available)
   - citation (in standard format)
   - abstract (full text)

2. Research information:
   - objectives (detailed description of research goals)
   - methodology (detailed steps and procedures)
   - key findings (with supporting data and context)
   - limitations (with explanations)
   - research questions (detailed description)
   - hypotheses (with context and rationale)
   - assumptions (with justification)
   - scope (detailed description)

3. Technical parameters:
   - measurements (with units, uncertainty, and context)
   - specifications (detailed descriptions)
   - performance metrics (with values and conditions)
   - experimental conditions (detailed setup)
   - materials (with properties and characteristics)
   - processes (with parameters and conditions)
   - equipment (with specifications and usage)
   - quality parameters (with standards and requirements)

4. Environmental aspects:
   - life cycle phases (detailed descriptions)
   - environmental impacts (with values and units)
   - resource consumption (with quantities and context)
   - emissions data (with types and amounts)
   - environmental indicators (with measurements)
   - impact categories (with assessment methods)
   - environmental standards (with compliance details)
   - mitigation measures (with implementation details)

5. Economic aspects:
   - costs (with types and values)
   - economic indicators (with calculations)
   - market analysis (with trends and data)
   - financial implications (with detailed analysis)
   - investment requirements (with breakdown)
   - operational costs (with categories)
   - revenue streams (with projections)
   - economic feasibility (detailed analysis)

6. Sustainability considerations:
   - environmental impacts (with detailed assessments)
   - social aspects (with stakeholder impacts)
   - circular economy elements (with implementation details)
   - sustainability metrics (with values and context)
   - sustainable practices (with examples)
   - sustainability goals (with targets)
   - stakeholder engagement (with strategies)
   - sustainability challenges (with solutions)

7. Additional information:
   - tables (with content and context)
   - figures (with detailed descriptions)
   - equations (with explanations)
   - important notes (with context)
   - recommendations (with rationale)
   - future work (with priorities)
   - case studies (with outcomes)
   - best practices (with examples)

8. Any other significant information:
   - regulatory compliance (with requirements)
   - industry standards (with specifications)
   - innovation aspects (with details)
   - risk assessment (with mitigation)
   - quality assurance (with procedures)
   - implementation details (with steps)
   - maintenance requirements (with schedule)
   - safety considerations (with protocols)

IMPORTANT:
- Extract ALL numerical values, measurements, and units with their context
- Include detailed descriptions and context for each finding
- Do not omit any information that could be relevant for technical or sustainability analysis
- Format the response as a clean JSON object
- Ensure all values are properly escaped and formatted as valid JSON strings
- Use consistent naming conventions (use underscores instead of hyphens)
- For any section where all fields are empty, omit that section entirely
- Include units, uncertainty values, and confidence intervals where available
- Preserve relationships between different pieces of information
- Include any information that doesn't fit the above categories in section 8
- Present information in detailed sentences rather than paragraphs
- Focus on clarity and precision in descriptions
- DO NOT use "not specified" or similar placeholder text - if information is not available, omit the field entirely
- Only include fields where actual information is present in the paper
- If a section has no available information, omit the entire section
- For nested objects, only include fields that have actual data
- Do not create empty or placeholder fields
- Extract ALL important information from tables, figures, and supplementary materials
- Include detailed context and explanations for all extracted data
- Capture relationships and dependencies between different pieces of information
- Extract both qualitative and quantitative information with full context
- Include all relevant technical specifications and parameters
- Document all environmental and sustainability metrics with their context
- Extract all economic and financial data with supporting information
- Include all methodological details and experimental procedures
- Capture all assumptions and their justifications
- Document all limitations and their implications
- Extract all recommendations and their rationale
- Include all future work suggestions and their priorities

Return a comprehensive JSON object containing all extracted information."""
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging configuration."""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        logging.basicConfig(
            filename=log_dir / f"extraction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def split_text(self, text: str, max_chars: int = 8000) -> list:
        """Split text into chunks of maximum size while preserving paragraph boundaries."""
        if len(text) <= max_chars:
            return [text]
            
        chunks = []
        current_chunk = ""
        paragraphs = text.split('\n\n')
        
        # Estimate tokens (rough approximation: 1 token ≈ 4 characters)
        max_tokens = 100000  # Leave room for system prompt and other overhead
        max_chars = min(max_chars, max_tokens * 4)
        
        for paragraph in paragraphs:
            # If a single paragraph is too long, split it into sentences
            if len(paragraph) > max_chars:
                sentences = paragraph.split('. ')
                for sentence in sentences:
                    if len(current_chunk) + len(sentence) + 2 <= max_chars:
                        current_chunk += sentence + '. '
                    else:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = sentence + '. '
            else:
                if len(current_chunk) + len(paragraph) + 2 <= max_chars:
                    current_chunk += paragraph + '\n\n'
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = paragraph + '\n\n'
                
        if current_chunk:
            chunks.append(current_chunk.strip())
            
        return chunks

    def make_llm_request(self, messages: list, is_cleanup: bool = False, max_retries: int = 3, initial_delay: float = 2.0) -> Optional[str]:
        """Make LLM request with retry logic and exponential backoff."""
        client = self.cleanup_client if is_cleanup else self.extraction_client
        
        # Estimate total tokens in messages
        total_chars = sum(len(msg.get('content', '')) for msg in messages)
        estimated_tokens = total_chars // 4  # Rough approximation
        
        if estimated_tokens > 100000:  # Leave room for system prompt and other overhead
            print(f"Warning: Estimated token count ({estimated_tokens}) exceeds recommended limit. Reducing chunk size...")
            return None
        
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    messages=messages,
                    model=self.model,
                    response_format={"type": "json_object"}
                )
                return response.choices[0].message.content
            except Exception as e:
                delay = initial_delay * (2 ** attempt)  # Exponential backoff
                if attempt < max_retries - 1:
                    print(f"Request failed (attempt {attempt + 1}/{max_retries}): {str(e)}")
                    print(f"Retrying in {delay} seconds...")
                    logging.warning(f"Request failed (attempt {attempt + 1}/{max_retries}): {str(e)}")
                    time.sleep(delay)
                else:
                    print(f"All {max_retries} attempts failed. Last error: {str(e)}")
                    logging.error(f"All {max_retries} attempts failed. Last error: {str(e)}")
                    return None

    def extract_data(self, text: str, filename: str) -> Optional[Dict[str, Any]]:
        """Extract data from text using LLM."""
        try:
            print(f"\nProcessing file: {filename}")
            print(f"Text length: {len(text)} characters")
            
            # Remove references section with multiple possible headers
            reference_headers = [
                "\nReferences",
                "\n19References",
                "REFERENCES\n",
                "references\n",
                "\nREFERENCES",
                "\nReferences\n",
                "\nREFERENCES\n"
            ]
            
            for header in reference_headers:
                if header in text:
                    text = text.split(header)[0]
                    print(f"Removed references section starting with: {header}")
                    break
            
            print(f"Text length after removing references: {len(text)} characters")
            
            # Split text into manageable chunks
            text_chunks = self.split_text(text)
            print(f"Split text into {len(text_chunks)} chunks")
            
            if not text_chunks:
                print("Error: Text splitting resulted in no chunks")
                return None
                
            all_extracted_data = {}
            
            for i, chunk in enumerate(text_chunks, 1):
                print(f"\nProcessing chunk {i}/{len(text_chunks)}")
                print(f"Chunk size: {len(chunk)} characters")
                
                # Initial extraction request
                extraction_messages = [
                    {"role": "system", "content": self.system_role},
                    {"role": "user", "content": f"{self.combined_prompt}\n\nText: {chunk}"}
                ]
                
                raw_response = self.make_llm_request(extraction_messages, is_cleanup=False)
                if not raw_response:
                    print(f"Failed to get response for chunk {i}")
                    continue
                
                # Clean the response
                import re
                cleaned_response = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', raw_response)
                cleaned_response = re.sub(r'\s+', ' ', cleaned_response)
                cleaned_response = cleaned_response.strip()
                
                try:
                    chunk_data = json.loads(cleaned_response)
                    print(f"Successfully parsed JSON for chunk {i}")
                    
                    # Merge chunk data with existing data
                    for key, value in chunk_data.items():
                        if key not in all_extracted_data:
                            all_extracted_data[key] = value
                        elif isinstance(value, list):
                            all_extracted_data[key].extend(value)
                        elif isinstance(value, dict):
                            all_extracted_data[key].update(value)
                        else:
                            # For other types, keep the most detailed value
                            if len(str(value)) > len(str(all_extracted_data[key])):
                                all_extracted_data[key] = value                
                except json.JSONDecodeError as je:
                    print(f"JSON parsing error in chunk {i}: {str(je)}")
                    logging.error(f"JSON parsing error in chunk {i}: {str(je)}")
                    logging.error(f"Raw response: {raw_response}")
                    continue
            
            if all_extracted_data:
                # Clean up the data by removing fields with useless or negative information
                cleanup_prompt = """Review the following JSON data and remove any fields that contain:
1. Negative statements about missing information
2. Placeholder text or empty information
3. Statements about what was not done or not provided
4. Any other useless or non-informative content

Keep only fields that contain actual, useful information. Return the cleaned JSON object."""

                cleanup_messages = [
                    {"role": "system", "content": "You are a data cleaning expert. Your task is to remove any fields containing useless or negative information while preserving all valuable data."},
                    {"role": "user", "content": f"{cleanup_prompt}\n\nData to clean: {json.dumps(all_extracted_data, indent=2)}"}
                ]

                cleaned_response = self.make_llm_request(cleanup_messages, is_cleanup=True)
                if cleaned_response:
                    try:
                        cleaned_data = json.loads(cleaned_response)
                        all_extracted_data = cleaned_data
                        print("Successfully cleaned the extracted data")
                    except json.JSONDecodeError as je:
                        print(f"JSON parsing error in cleanup: {str(je)}")
                        logging.error(f"JSON parsing error in cleanup: {str(je)}")
                        # Continue with original data if cleanup fails
                        pass

                # Remove file_info field if it exists
                all_extracted_data.pop('file_info', None)
                return all_extracted_data
            else:
                print("No valid data extracted from any chunk")
                return None
                
        except Exception as e:
            print(f"Error in extraction: {str(e)}")
            logging.error(f"Extraction error: {str(e)}")
            return None

    def process_papers(self, papers_file: str):
        """Process all papers from the processed papers file."""
        try:
            print(f"\nLoading papers from: {papers_file}")
            with open(papers_file, 'r', encoding='utf-8') as f:
                papers = json.load(f)
            
            total_files = len(papers)
            print(f"Found {total_files} papers to process")
            
            # Create output directory
            output_path = Path(self.output_file)
            output_path.parent.mkdir(exist_ok=True)
            
            # Load existing data if available
            if output_path.exists():
                with open(output_path, 'r', encoding='utf-8') as f:
                    all_data = json.load(f)
            else:
                all_data = {}
            
            # Process each paper
            for i, (paper_id, paper_data) in enumerate(papers.items(), 1):
                # Check if paper was already processed
                if paper_id in all_data:
                    print(f"\nSkipping already processed paper {i}/{total_files}: {paper_id}")
                    continue
                
                print(f"\nProcessing paper {i}/{total_files}: {paper_id}")
                
                # Extract data
                extracted_data = self.extract_data(
                    paper_data["full_text"],
                    paper_data.get('metadata', {}).get('filename', 'unknown')
                )
                
                if extracted_data:
                    # Save data
                    all_data[paper_id] = extracted_data
                    with open(output_path, 'w', encoding='utf-8') as f:
                        json.dump(all_data, f, ensure_ascii=False, indent=2)
                    print(f"Saved data for paper {paper_id}")
                
                # Log progress
                progress = (i / total_files) * 100
                print(f"Progress: {progress:.2f}% ({i}/{total_files} files)")
            
            print("\nProcessing complete!")
            
        except Exception as e:
            print(f"Error processing papers: {str(e)}")
            logging.error(f"Error processing papers: {str(e)}")
    
    async def make_llm_request(self, messages: list, is_cleanup: bool = False, max_retries: int = 3, initial_delay: float = 2.0) -> Optional[str]:
        """Make LLM request with retry logic and exponential backoff."""
        client = self.cleanup_client if is_cleanup else self.extraction_client
        
        for attempt in range(max_retries):
            try:
                response = await client.chat.completions.create(
                    messages=messages,
                    model=self.model,
                    response_format={"type": "json_object"}
                )
                return response.choices[0].message.content
            except Exception as e:
                delay = initial_delay * (2 ** attempt)  # Exponential backoff
                if attempt < max_retries - 1:
                    print(f"Request failed (attempt {attempt + 1}/{max_retries}): {str(e)}")
                    print(f"Retrying in {delay} seconds...")
                    logging.warning(f"Request failed (attempt {attempt + 1}/{max_retries}): {str(e)}")
                    await asyncio.sleep(delay)
                else:
                    print(f"All {max_retries} attempts failed. Last error: {str(e)}")
                    logging.error(f"All {max_retries} attempts failed. Last error: {str(e)}")
                    return None

    async def process_chunk(self, chunk: str, chunk_num: int, total_chunks: int) -> Optional[Dict[str, Any]]:
        """Process a single chunk of text."""
        print(f"\nProcessing chunk {chunk_num}/{total_chunks}")
        
        # Initial extraction request
        extraction_messages = [
            {"role": "system", "content": self.system_role},
            {"role": "user", "content": f"{self.combined_prompt}\n\nText: {chunk}"}
        ]
        
        raw_response = await self.make_llm_request(extraction_messages, is_cleanup=False)
        if not raw_response:
            print(f"Failed to get response for chunk {chunk_num}")
            return None
        
        # Clean the response
        import re
        cleaned_response = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', raw_response)
        cleaned_response = re.sub(r'\s+', ' ', cleaned_response)
        cleaned_response = cleaned_response.strip()
        
        try:
            chunk_data = json.loads(cleaned_response)
            print(f"Successfully parsed JSON for chunk {chunk_num}")
            return chunk_data
        except json.JSONDecodeError as je:
            print(f"JSON parsing error in chunk {chunk_num}: {str(je)}")
            logging.error(f"JSON parsing error in chunk {chunk_num}: {str(je)}")
            logging.error(f"Raw response: {raw_response}")
            return None

    async def cleanup_data(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Clean up the extracted data."""
        cleanup_prompt = """Review the following JSON data and remove any fields that contain:
1. Negative statements about missing information
2. Placeholder text or empty information
3. Statements about what was not done or not provided
4. Any other useless or non-informative content

Keep only fields that contain actual, useful information. Return the cleaned JSON object."""

        cleanup_messages = [
            {"role": "system", "content": "You are a data cleaning expert. Your task is to remove any fields containing useless or negative information while preserving all valuable data."},
            {"role": "user", "content": f"{cleanup_prompt}\n\nData to clean: {json.dumps(data, indent=2)}"}
        ]

        cleaned_response = await self.make_llm_request(cleanup_messages, is_cleanup=True)
        if cleaned_response:
            try:
                cleaned_data = json.loads(cleaned_response)
                print("Successfully cleaned the extracted data")
                return cleaned_data
            except json.JSONDecodeError as je:
                print(f"JSON parsing error in cleanup: {str(je)}")
                logging.error(f"JSON parsing error in cleanup: {str(je)}")
                return data
        return data

    async def extract_data_async(self, text: str, filename: str) -> Optional[Dict[str, Any]]:
        """Extract data from text using LLM with parallel processing."""
        try:
            print(f"\nProcessing file: {filename}")
            print(f"Text length: {len(text)} characters")
            
            # Remove references section
            reference_headers = [
                "\nReferences", "\n19References", "REFERENCES\n", "references\n",
                "\nREFERENCES", "\nReferences\n", "\nREFERENCES\n"
            ]
            
            for header in reference_headers:
                if header in text:
                    text = text.split(header)[0]
                    print(f"Removed references section starting with: {header}")
                    break
            
            print(f"Text length after removing references: {len(text)} characters")
            
            # Split text into chunks
            text_chunks = self.split_text(text)
            print(f"Split text into {len(text_chunks)} chunks")
            
            all_extracted_data = {}
            cleanup_tasks = []
            
            # Process chunks with parallel cleanup
            for i, chunk in enumerate(text_chunks, 1):
                # Process current chunk
                chunk_data = await self.process_chunk(chunk, i, len(text_chunks))
                if chunk_data:
                    # Merge chunk data
                    for key, value in chunk_data.items():
                        if key not in all_extracted_data:
                            all_extracted_data[key] = value
                        elif isinstance(value, list):
                            all_extracted_data[key].extend(value)
                        elif isinstance(value, dict):
                            all_extracted_data[key].update(value)
                        else:
                            if len(str(value)) > len(str(all_extracted_data[key])):
                                all_extracted_data[key] = value
                
                # Start cleanup of previous chunk's data if available
                if i > 1 and all_extracted_data:
                    cleanup_task = asyncio.create_task(self.cleanup_data(all_extracted_data.copy()))
                    cleanup_tasks.append(cleanup_task)
            
            # Wait for all cleanup tasks to complete
            if cleanup_tasks:
                cleaned_results = await asyncio.gather(*cleanup_tasks)
                # Use the last cleaned result
                if cleaned_results:
                    all_extracted_data = cleaned_results[-1]
            
            if all_extracted_data:
                # Remove file_info field if it exists
                all_extracted_data.pop('file_info', None)
                return all_extracted_data
            else:
                print("No valid data extracted from any chunk")
                return None
                
        except Exception as e:
            print(f"Error in extraction: {str(e)}")
            logging.error(f"Extraction error: {str(e)}")
            return None

    def extract_data(self, text: str, filename: str) -> Optional[Dict[str, Any]]:
        """Synchronous wrapper for async extraction."""
        return asyncio.run(self.extract_data_async(text, filename))

if __name__ == "__main__":
    # Create output directory if it doesn't exist
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    # Initialize and run data extractor
    extractor = DataExtractor()  # Will use API keys from config
    extractor.process_papers("output/processed_papers.json")
    print("Data extraction complete! Check the 'output' directory for results.")

