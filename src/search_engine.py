import numpy as np
import pandas as pd
import voyageai
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import logging
import os 
from pinecone import Pinecone

pinecone_api_key = os.getenv("PINECONE_API_KEY")

class SearchEngine:
    """Handles similarity search"""
    
    def __init__(self, voyage_client: voyageai.Client):
        self.vo = voyage_client
        self.logger = logging.getLogger(__name__)
        self.pc = Pinecone(api_key=pinecone_api_key)
        self.index = self.pc.Index("nhs-conditions")
    
    def similarity_search(self, query_text: str, namespace: str, top_k: int = 25) -> List[dict]:
        """Perform similarity search using Pinecone"""
        try:
            # Embed the query using the same model 
            query_embedding = self.vo.contextualized_embed(
                inputs=[[query_text]], 
                model="voyage-context-3", 
                input_type="query", 
                output_dimension=2048
            ).results[0].embeddings[0]
            
            # Search Pinecone
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                namespace=namespace,
                include_metadata=True
            )
            
            matches = results['matches']
            self.logger.info(f"Pinecone search found {len(matches)} results")
            return matches
        
        except Exception as e:
            self.logger.error(f"Error in Pinecone similarity search: {e}")
            return []
