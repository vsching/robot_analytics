"""Database connection manager with pooling support."""

import os
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Generator, Any
from queue import Queue, Empty
import logging

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.pool import QueuePool, NullPool


logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages database connections with pooling support."""
    
    def __init__(self, 
                 db_path: Optional[str] = None,
                 pool_size: int = 5,
                 max_overflow: int = 10,
                 pool_timeout: int = 30,
                 echo: bool = False):
        """
        Initialize database manager.
        
        Args:
            db_path: Path to SQLite database file
            pool_size: Number of connections to maintain in pool
            max_overflow: Maximum overflow connections allowed
            pool_timeout: Timeout for getting connection from pool
            echo: Whether to echo SQL statements
        """
        self.db_path = db_path or os.getenv('DATABASE_URL', 'sqlite:///trading_analyzer.db')
        self.pool_size = pool_size
        self.max_overflow = max_overflow
        self.pool_timeout = pool_timeout
        self.echo = echo
        
        # Remove sqlite:/// prefix if present
        if self.db_path.startswith('sqlite:///'):
            self.db_path = self.db_path[10:]
        
        # Ensure database directory exists
        db_dir = Path(self.db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)
        
        # Thread-local storage for connections
        self._local = threading.local()
        
        # Initialize SQLAlchemy engine
        self._engine = self._create_engine()
        
        # Initialize database schema
        self._initialize_database()
    
    def _create_engine(self) -> Engine:
        """Create SQLAlchemy engine with connection pooling."""
        # SQLite doesn't benefit from connection pooling in the same way
        # but we'll use QueuePool for consistency
        engine = create_engine(
            f'sqlite:///{self.db_path}',
            poolclass=QueuePool,
            pool_size=self.pool_size,
            max_overflow=self.max_overflow,
            pool_timeout=self.pool_timeout,
            echo=self.echo,
            connect_args={
                'check_same_thread': False,  # Allow multiple threads
                'timeout': 30.0  # Connection timeout
            }
        )
        
        # Enable foreign keys for SQLite
        @event.listens_for(engine, "connect")
        def set_sqlite_pragma(dbapi_connection, connection_record):
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute("PRAGMA journal_mode=WAL")  # Write-Ahead Logging
            cursor.close()
        
        return engine
    
    def _initialize_database(self) -> None:
        """Initialize database with schema."""
        schema_path = Path(__file__).parent / 'schema.sql'
        
        if schema_path.exists():
            with open(schema_path, 'r') as f:
                schema_sql = f.read()
            
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.executescript(schema_sql)
                conn.commit()
                logger.info("Database schema initialized successfully")
        else:
            logger.warning(f"Schema file not found at {schema_path}")
    
    @property
    def engine(self) -> Engine:
        """Get SQLAlchemy engine."""
        return self._engine
    
    @contextmanager
    def get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """
        Get a database connection from the pool.
        
        Yields:
            sqlite3.Connection: Database connection
        """
        conn = None
        try:
            # Get raw connection from SQLAlchemy pool
            conn = self._engine.raw_connection()
            yield conn.connection  # Get the underlying sqlite3 connection
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Database connection error: {e}")
            raise
        finally:
            if conn:
                conn.close()
    
    @contextmanager
    def transaction(self) -> Generator[sqlite3.Connection, None, None]:
        """
        Execute operations within a transaction.
        
        Yields:
            sqlite3.Connection: Database connection with transaction
        """
        with self.get_connection() as conn:
            try:
                yield conn
                conn.commit()
            except Exception as e:
                conn.rollback()
                logger.error(f"Transaction rollback due to: {e}")
                raise
    
    def execute(self, query: str, params: Optional[tuple] = None) -> sqlite3.Cursor:
        """
        Execute a single query.
        
        Args:
            query: SQL query to execute
            params: Query parameters
            
        Returns:
            sqlite3.Cursor: Query result cursor
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            conn.commit()
            return cursor
    
    def executemany(self, query: str, params_list: list) -> sqlite3.Cursor:
        """
        Execute a query with multiple parameter sets.
        
        Args:
            query: SQL query to execute
            params_list: List of parameter tuples
            
        Returns:
            sqlite3.Cursor: Query result cursor
        """
        with self.transaction() as conn:
            cursor = conn.cursor()
            cursor.executemany(query, params_list)
            return cursor
    
    def fetchone(self, query: str, params: Optional[tuple] = None) -> Optional[tuple]:
        """
        Execute query and fetch one result.
        
        Args:
            query: SQL query to execute
            params: Query parameters
            
        Returns:
            Optional[tuple]: First result row or None
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            return cursor.fetchone()
    
    def fetchall(self, query: str, params: Optional[tuple] = None) -> list:
        """
        Execute query and fetch all results.
        
        Args:
            query: SQL query to execute
            params: Query parameters
            
        Returns:
            list: All result rows
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            return cursor.fetchall()
    
    def close(self) -> None:
        """Close all connections and dispose of the engine."""
        self._engine.dispose()
        logger.info("Database connections closed")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


# Global database manager instance
_db_manager: Optional[DatabaseManager] = None


def get_db_manager() -> DatabaseManager:
    """
    Get or create the global database manager instance.
    
    Returns:
        DatabaseManager: Global database manager
    """
    global _db_manager
    
    if _db_manager is None:
        _db_manager = DatabaseManager()
    
    return _db_manager


def close_db_manager() -> None:
    """Close the global database manager."""
    global _db_manager
    
    if _db_manager is not None:
        _db_manager.close()
        _db_manager = None