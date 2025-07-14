"""Base repository class for CRUD operations."""

from abc import ABC, abstractmethod
from typing import TypeVar, Generic, List, Optional, Dict, Any, Type
import logging
from datetime import datetime

from .connection import get_db_manager


logger = logging.getLogger(__name__)

T = TypeVar('T')


class BaseRepository(Generic[T], ABC):
    """Base repository class providing common CRUD operations."""
    
    def __init__(self, table_name: str, model_class: Type[T]):
        """
        Initialize base repository.
        
        Args:
            table_name: Name of the database table
            model_class: Model class for type conversion
        """
        self.table_name = table_name
        self.model_class = model_class
        self.db = get_db_manager()
    
    @abstractmethod
    def _row_to_model(self, row: tuple, columns: List[str]) -> T:
        """Convert database row to model instance."""
        pass
    
    @abstractmethod
    def _model_to_params(self, model: T) -> Dict[str, Any]:
        """Convert model instance to database parameters."""
        pass
    
    def create(self, model: T) -> T:
        """
        Create a new record.
        
        Args:
            model: Model instance to create
            
        Returns:
            T: Created model with ID
        """
        params = self._model_to_params(model)
        
        # Remove id if present (auto-increment)
        params.pop('id', None)
        
        columns = list(params.keys())
        placeholders = ', '.join(['?' for _ in columns])
        column_names = ', '.join(columns)
        
        query = f"""
            INSERT INTO {self.table_name} ({column_names})
            VALUES ({placeholders})
        """
        
        try:
            cursor = self.db.execute(query, tuple(params.values()))
            model.id = cursor.lastrowid
            logger.info(f"Created {self.model_class.__name__} with ID: {model.id}")
            return model
        except Exception as e:
            logger.error(f"Failed to create {self.model_class.__name__}: {e}")
            raise
    
    def get_by_id(self, id: int) -> Optional[T]:
        """
        Get a record by ID.
        
        Args:
            id: Record ID
            
        Returns:
            Optional[T]: Model instance or None
        """
        query = f"SELECT * FROM {self.table_name} WHERE id = ?"
        
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query, (id,))
                row = cursor.fetchone()
                
                if row:
                    columns = [desc[0] for desc in cursor.description]
                    return self._row_to_model(row, columns)
                return None
        except Exception as e:
            logger.error(f"Failed to get {self.model_class.__name__} by ID {id}: {e}")
            raise
    
    def get_all(self, limit: Optional[int] = None, offset: int = 0) -> List[T]:
        """
        Get all records with optional pagination.
        
        Args:
            limit: Maximum number of records to return
            offset: Number of records to skip
            
        Returns:
            List[T]: List of model instances
        """
        query = f"SELECT * FROM {self.table_name}"
        params = []
        
        if limit:
            query += " LIMIT ? OFFSET ?"
            params = [limit, offset]
        
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                if params:
                    cursor.execute(query, params)
                else:
                    cursor.execute(query)
                
                rows = cursor.fetchall()
                columns = [desc[0] for desc in cursor.description]
                
                return [self._row_to_model(row, columns) for row in rows]
        except Exception as e:
            logger.error(f"Failed to get all {self.model_class.__name__}: {e}")
            raise
    
    def update(self, model: T) -> bool:
        """
        Update an existing record.
        
        Args:
            model: Model instance with updated values
            
        Returns:
            bool: True if updated successfully
        """
        if not model.id:
            raise ValueError("Model must have an ID to update")
        
        params = self._model_to_params(model)
        id = params.pop('id')
        
        # Update timestamp if model has it
        if 'updated_at' in params:
            params['updated_at'] = datetime.now()
        
        set_clause = ', '.join([f"{k} = ?" for k in params.keys()])
        query = f"""
            UPDATE {self.table_name}
            SET {set_clause}
            WHERE id = ?
        """
        
        try:
            cursor = self.db.execute(query, tuple(params.values()) + (id,))
            success = cursor.rowcount > 0
            
            if success:
                logger.info(f"Updated {self.model_class.__name__} ID: {id}")
            else:
                logger.warning(f"No {self.model_class.__name__} found with ID: {id}")
            
            return success
        except Exception as e:
            logger.error(f"Failed to update {self.model_class.__name__} ID {id}: {e}")
            raise
    
    def delete(self, id: int) -> bool:
        """
        Delete a record by ID.
        
        Args:
            id: Record ID to delete
            
        Returns:
            bool: True if deleted successfully
        """
        query = f"DELETE FROM {self.table_name} WHERE id = ?"
        
        try:
            cursor = self.db.execute(query, (id,))
            success = cursor.rowcount > 0
            
            if success:
                logger.info(f"Deleted {self.model_class.__name__} ID: {id}")
            else:
                logger.warning(f"No {self.model_class.__name__} found with ID: {id}")
            
            return success
        except Exception as e:
            logger.error(f"Failed to delete {self.model_class.__name__} ID {id}: {e}")
            raise
    
    def count(self, where_clause: Optional[str] = None, params: Optional[tuple] = None) -> int:
        """
        Count records with optional filter.
        
        Args:
            where_clause: Optional WHERE clause
            params: Optional parameters for WHERE clause
            
        Returns:
            int: Number of records
        """
        query = f"SELECT COUNT(*) FROM {self.table_name}"
        
        if where_clause:
            query += f" WHERE {where_clause}"
        
        try:
            result = self.db.fetchone(query, params)
            return result[0] if result else 0
        except Exception as e:
            logger.error(f"Failed to count {self.model_class.__name__}: {e}")
            raise
    
    def exists(self, id: int) -> bool:
        """
        Check if a record exists by ID.
        
        Args:
            id: Record ID
            
        Returns:
            bool: True if exists
        """
        query = f"SELECT 1 FROM {self.table_name} WHERE id = ? LIMIT 1"
        
        try:
            result = self.db.fetchone(query, (id,))
            return result is not None
        except Exception as e:
            logger.error(f"Failed to check existence of {self.model_class.__name__} ID {id}: {e}")
            raise
    
    def find_by(self, **kwargs) -> List[T]:
        """
        Find records by field values.
        
        Args:
            **kwargs: Field-value pairs to filter by
            
        Returns:
            List[T]: List of matching model instances
        """
        if not kwargs:
            return self.get_all()
        
        where_clauses = [f"{k} = ?" for k in kwargs.keys()]
        where_clause = " AND ".join(where_clauses)
        query = f"SELECT * FROM {self.table_name} WHERE {where_clause}"
        
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query, tuple(kwargs.values()))
                
                rows = cursor.fetchall()
                columns = [desc[0] for desc in cursor.description]
                
                return [self._row_to_model(row, columns) for row in rows]
        except Exception as e:
            logger.error(f"Failed to find {self.model_class.__name__} by {kwargs}: {e}")
            raise