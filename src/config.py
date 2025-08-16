import os
from enum import Enum
from typing import Dict, NamedTuple
from dataclasses import dataclass

class InfoSource(Enum):
    NHS = "nhs"

@dataclass
class SourceConfig:
    context_description: str
    not_found_message: str

class Config:
    """Configuration settings for the RAG system"""
    
    # Default similarity search parameters
    DEFAULT_SIMILARITY_K = 5
    
    SOURCE_CONFIGS = {
        InfoSource.NHS: SourceConfig(
            context_description="NHS health conditions and medical information",
            not_found_message="no relevant NHS health information is available to answer this question"
        )
    }
    
    @classmethod
    def get_source_config(cls, source: str) -> SourceConfig:
        """Get configuration for a source"""
        try:
            source_enum = InfoSource(source.lower())
            return cls.SOURCE_CONFIGS[source_enum]
        except ValueError:
            raise ValueError(f"Unknown source: {source}. Valid sources: {[s.value for s in InfoSource]}")


