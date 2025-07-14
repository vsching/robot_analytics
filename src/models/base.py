"""Base model classes for the Trading Strategy Analyzer."""

from datetime import datetime
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from abc import ABC, abstractmethod


@dataclass
class BaseModel(ABC):
    """Base class for all data models."""
    
    id: Optional[int] = None
    created_at: Optional[datetime] = None
    
    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary."""
        pass
    
    @classmethod
    @abstractmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BaseModel':
        """Create model from dictionary."""
        pass
    
    def __post_init__(self):
        """Initialize timestamps if not provided."""
        if self.created_at is None:
            self.created_at = datetime.now()


@dataclass
class TimestampedModel(BaseModel):
    """Base class for models with created_at and updated_at timestamps."""
    
    updated_at: Optional[datetime] = None
    
    def __post_init__(self):
        """Initialize timestamps if not provided."""
        super().__post_init__()
        if self.updated_at is None:
            self.updated_at = self.created_at