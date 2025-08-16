import os
import argparse
import logging
from typing import Dict, List, Optional, Generator, Tuple
from openai import OpenAI
from config import Config, InfoSource
from search_engine import SearchEngine
import voyageai

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RAGSystem:
    """Main RAG system class"""
    
    def __init__(self, shared_data=None):
        self.config = Config()
        
        # Initialize clients
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if gemini_api_key:
            self.gemini_client = OpenAI(
                api_key=gemini_api_key, 
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
            )
        else:
            self.gemini_client = None

        openai_api_key = os.getenv("OPENAI_API_KEY")
        if openai_api_key:
            self.openai_client = OpenAI(api_key=openai_api_key)
        else:
            self.openai_client = None
            
        self.voyage_client = voyageai.Client(api_key=os.getenv("VOYAGE_API_KEY"))
        self.search_engine = SearchEngine(self.voyage_client)
        
    def _validate_inputs(self, query_text: str, similarity_k: int, info_source: str):
        """Validate input parameters"""
        if not query_text or not query_text.strip():
            raise ValueError("Query text cannot be empty")
        
        if similarity_k <= 0:
            raise ValueError("similarity_k must be a positive integer")
        
        try:
            InfoSource(info_source.lower())
        except ValueError:
            valid_sources = [s.value for s in InfoSource]
            raise ValueError(f"Invalid info_source '{info_source}'. Must be one of: {valid_sources}")

    def _clean_section_id(self, section_id: str) -> str:
        """Clean section ID for display - NHS format: condition__section__part"""
        if not section_id or section_id == 'Unknown section':
            return section_id
        
        # Handle NHS format: "adhd-adults__Overview__Part_1"
        if '__' in section_id:
            parts = section_id.split('__')
            if len(parts) >= 2:
                # Get condition and section, ignore part number
                condition = parts[0].replace('-', ' ').replace('_', ' ').title()
                section = parts[1].replace('_', ' ').title()
                return f"{condition} - {section}"
        
        # Fallback: just clean up underscores and dashes
        clean_section = section_id.replace('_', ' ').replace('-', ' ').title()
        return clean_section
    
    def _get_context_text(self, results: List[Dict]) -> str:
        """Generate context text from search results"""
        context_text_sections = []

        for doc in results:
            section_id = doc['metadata'].get('original_id', 'Unknown section')
            url = doc['metadata'].get('url', '')
            document_text = doc['metadata'].get('document', '')
            
            # Clean up section_id for display
            clean_section_id = self._clean_section_id(section_id)
            
            # Create formatted section without showing URL explicitly
            # The URL will be available in the document_text if it was part of the original content
            formatted_section = (
                f"Source Information: [Section: {clean_section_id}]\n"
                f"Context: {document_text}"
                f"{f' Available at: {url}' if url else ''}"  # Include URL for LLM to use
            )
            context_text_sections.append(formatted_section)
        
        return "\n\n---\n\n".join(context_text_sections)
        
    def _create_system_prompt(self, context_text: str, context_description: str, 
                            not_found_message: str, query_text: str) -> List[Dict]:
        """Create system prompt for LLM"""
        return [
            {
                "role": "system",
                "content": (
                    f"You are a medical AI assistant tasked with answering clinical questions strictly based on the provided {context_description} context. Follow the requirements below to ensure accurate, consistent, and professional responses.\n\n"
                    "# Response Rules\n\n"
                    "1. **Context Restriction**:\n"
                    "   - Only use information given in the provided NHS health information context.\n"
                    "   - Do not generate or speculate with information not explicitly found in the given context.\n\n"
                    "2. **Answer Format**:\n"
                    "   - Provide a clear and concise response based solely on the context.\n"
                    "   - When including a list, use standard markdown bullet points (`*` or `-`).\n"
                    "   - If a list follows introductory text, insert a line break before the first bullet point.\n"
                    "   - Each bullet point must be on its own line.\n\n"
                    "3. **Preserve Tables**:\n"
                    "   - If relevant markdown tables appear in the context, reproduce them in your answer.\n"
                    "   - Maintain the original structure, formatting, and content of any included tables.\n\n"
                    "4. **Links and URLs**:\n"
                    "   - Include any URLs or web links from the context directly in your response when relevant.\n"
                    "   - Integrate links naturally within sentences, using markdown syntax for clickable text links.\n"
                    "   - DO NOT generate or invent any URLs not explicitly present in the context.\n\n"
                    "5. **Markdown Link Formatting**:\n"
                    "   - In responses, only the descriptive text in brackets should be visible and clickable (e.g., `[NHS ADHD information](https://www.nhs.uk/conditions/attention-deficit-hyperactivity-disorder-adhd/)`).\n"
                    "   - Readers should never see raw URLs in the text.\n"
                    "   - Use descriptive link text like 'NHS ADHD information' or 'NHS depression guide' rather than generic terms.\n\n"
                    "6. **If No Relevant Information**:\n"
                    "   - If the context contains no relevant information, state clearly:\n"
                    f"      *\"{not_found_message}\"*\n\n"
                    "# Output Format\n\n"
                    "- All responses should be in plain text, using markdown formatting for lists and links as required.\n"
                    "- Do not use code blocks.\n"
                    "- Answers should be concise, accurate, and formatted according to the rules above.\n\n"
                    "# Examples\n\n"
                    "**Example 1: Integration of markdown link in context**\n"
                    "Question: \"What are the symptoms of ADHD?\"\n"
                    "Context snippet: ...see the NHS information on ADHD symptoms...\n"
                    "Output:\n"
                    "According to the [NHS ADHD information](https://www.nhs.uk/conditions/attention-deficit-hyperactivity-disorder-adhd/), symptoms include...\n\n"
                    "**Example 2: Multiple condition references**\n"
                    "According to NHS guidance:\n"
                    "* Initial symptoms may include difficulty concentrating.\n"
                    "* For detailed information, see the [NHS ADHD guide](https://www.nhs.uk/conditions/adhd/).\n\n"
                    "**Example 3: No relevant context**\n"
                    f"{not_found_message}\n\n"
                    "# Notes\n\n"
                    "- Never output information beyond what is provided in the supplied context.\n"
                    "- Always use markdown for lists and links.\n"
                    "- Make sure all markdown tables from context are preserved in your answer if relevant.\n"
                    "- Present links only as clickable text, not as bare URLs.\n"
                    "- Use descriptive link text that indicates the specific NHS condition or topic.\n\n"
                    "**REMINDER:**\n"
                    "Strictly adhere to all formatting and content rules above for every response."
                ),
            },
            {
                "role": "assistant", 
                "content": (
                    f"Here is the context from {context_description} that you should use to answer the following question:\n\n{context_text}\n\n"
                ),
            },
            {
                "role": "user",
                "content": query_text,
            },
        ]
    


    def get_sources_from_results(self, results: List[Dict], info_source: str) -> List[Dict]:
        """Extract formatted sources from search results"""
        sources = []
        for doc in results:
            metadata = doc.get('metadata', {})
            section_id = metadata.get('original_id', 'Unknown section')
            source = metadata.get('source', 'Unknown')
            url = metadata.get('url', '')
            
            # Clean section ID for display
            clean_section_id = self._clean_section_id(section_id)
            
            source_info = {
                'metadata': {
                    'source': source,
                    'original_id': section_id,
                    'url': url,
                    'clean_section': clean_section_id
                }
            }
            sources.append(source_info)
        return sources
    
    def query_rag_stream(self, query_text: str, llm_model: str, similarity_k: int = 25, info_source: str = "NHS", 
                        filename_filter: Optional[str] = None) -> Generator[Tuple[str, List[Dict]], None, None]:
        """Query RAG system with streaming response"""
        try:
            self._validate_inputs(query_text, similarity_k, info_source)
            source_config = self.config.get_source_config(info_source)
            
            # Use the correct namespace from your test
            namespace = "nhs_guidelines_voyage_3_large"
            
            # Get similar documents using only similarity search
            results = self.search_engine.similarity_search(
                query_text, 
                namespace=namespace,
                top_k=similarity_k
            )
            
            if not results:
                yield "I couldn't find any relevant information to answer your question.", []
                return
            
            # Generate context and system prompt
            context_text = self._get_context_text(results)
            system_messages = self._create_system_prompt(
                context_text, 
                source_config.context_description,
                source_config.not_found_message, 
                query_text
            )
            
            # Get sources for response
            sources_data = self.get_sources_from_results(results, info_source)
            
            # Stream LLM response
            yield from self._stream_llm_response(system_messages, query_text, llm_model, sources_data)
            
        except Exception as e:
            logger.error(f"Error in query_rag_stream: {e}")
            yield f"An error occurred while processing your query: {str(e)}", []
    
    def _stream_llm_response(self, system_messages: List[Dict], query_text: str, 
                           llm_model: str, sources_data: List[Dict]) -> Generator[Tuple[str, List[Dict]], None, None]:
        """Stream LLM response"""
        try:
            if "gemini" in llm_model.lower() and self.gemini_client:
                stream = self.gemini_client.chat.completions.create(
                    model=llm_model,
                    messages=system_messages,
                    temperature=0,
                    stream=True
                )
                
                for chunk in stream:
                    if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        yield content, sources_data
                    
            else:
                error_msg = f"Unsupported LLM model or client not available: {llm_model}"
                logger.error(error_msg)
                yield error_msg, []
                return

        except Exception as e:
            logger.error(f"Error in LLM completion: {e}")
            yield f"Error generating response: {str(e)}", []



def main():
    """Main function for CLI usage"""
    parser = argparse.ArgumentParser(description="RAG System Query Interface")
    parser.add_argument("--query_text", type=str, default="What are the symptoms of ADHD in adults?", 
                       help="The query text.")
    parser.add_argument("--llm_model", type=str, default="gemini-2.0-flash", 
                       help="The LLM model to use.")
    parser.add_argument("--similarity_k", type=int, default=5, 
                       help="Number of results to retrieve in similarity search.")
    parser.add_argument("--info_source", type=str, default="NHS", 
                       choices=["nhs", "NHS"],
                       help="Information source to query.")
    
    args = parser.parse_args()
    
    try:
        print("Initializing RAG system...")
        rag_system = RAGSystem()
        
        print(f"\n=== Query: {args.query_text} ===")
        print(f"Source: {args.info_source}")
        print(f"LLM Model: {args.llm_model}")
        print("\n=== LLM Response ===\n")
        
        response_text, sources_data = "", []
        
        for chunk, sources in rag_system.query_rag_stream(
            query_text=args.query_text,
            llm_model=args.llm_model,
            similarity_k=args.similarity_k,
            info_source=args.info_source
        ):
            print(chunk, end="", flush=True)
            response_text += chunk
            sources_data = sources
        
        print("\n\n=== Sources Data ===\n")
        for i, source in enumerate(sources_data, 1):
            metadata = source.get('metadata', {})
            print(f"Source {i}:")
            print(f"  Clean Section: {metadata.get('clean_section', 'Unknown')}")
            print(f"  URL: {metadata.get('url', 'No URL')}")
            print()
        
    except Exception as e:
        logger.error(f"Error in main: {e}")
        print(f"Error: {e}")


if __name__ == "__main__":
    main()