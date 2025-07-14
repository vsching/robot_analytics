"""CSV data transformation and standardization pipeline."""

import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, date, time
from decimal import Decimal
import pandas as pd
import numpy as np

from .csv_formats import Platform, CSVFormat, get_format
from ..models import Trade


logger = logging.getLogger(__name__)


class CSVTransformer:
    """Transforms CSV data from various platforms into standardized format."""
    
    def __init__(self):
        self.standard_columns = {
            'trade_date': 'datetime64[ns]',
            'symbol': 'str',
            'side': 'str',
            'quantity': 'float64',
            'entry_price': 'float64',
            'exit_price': 'float64',
            'pnl': 'float64',
            'commission': 'float64',
            'entry_time': 'object',  # Will be time object
            'exit_time': 'object',   # Will be time object
            'duration_hours': 'float64'
        }
        
        self.side_mapping = {
            # Buy variations
            'buy': 'buy',
            'b': 'buy',
            'long': 'long',
            'l': 'long',
            '1': 'buy',
            'bought': 'buy',
            # Sell variations
            'sell': 'sell',
            's': 'sell',
            'short': 'short',
            'sh': 'short',
            '-1': 'sell',
            'sold': 'sell',
            # MetaTrader specific
            '0': 'buy',  # MT4/MT5 buy
            '1': 'sell',  # MT4/MT5 sell
            'buy limit': 'buy',
            'sell limit': 'sell',
            'buy stop': 'buy',
            'sell stop': 'sell'
        }
    
    def transform(self, df: pd.DataFrame, platform: Platform, 
                 format_spec: Optional[CSVFormat] = None) -> pd.DataFrame:
        """
        Transform platform-specific CSV data to standardized format.
        
        Args:
            df: Input DataFrame
            platform: Detected platform
            format_spec: Optional format specification
            
        Returns:
            Transformed DataFrame with standardized columns
        """
        if format_spec is None:
            format_spec = get_format(platform)
        
        logger.info(f"Transforming {len(df)} rows from {platform.value} format")
        
        # Create a copy to avoid modifying original
        transformed_df = df.copy()
        
        # Step 1: Rename columns based on mapping
        transformed_df = self._rename_columns(transformed_df, format_spec)
        
        # Step 2: Transform data types
        transformed_df = self._transform_data_types(transformed_df, format_spec)
        
        # Step 3: Standardize values
        transformed_df = self._standardize_values(transformed_df)
        
        # Step 4: Calculate derived fields
        transformed_df = self._calculate_derived_fields(transformed_df)
        
        # Step 5: Select and order standard columns
        transformed_df = self._select_standard_columns(transformed_df)
        
        # Step 6: Final cleanup
        transformed_df = self._final_cleanup(transformed_df)
        
        logger.info(f"Transformation complete. Output has {len(transformed_df)} rows")
        
        return transformed_df
    
    def _rename_columns(self, df: pd.DataFrame, format_spec: CSVFormat) -> pd.DataFrame:
        """Rename columns based on format mapping."""
        # First apply direct mappings from format spec
        rename_map = {}
        for old_col, new_col in format_spec.column_mappings.items():
            if old_col in df.columns:
                rename_map[old_col] = new_col
        
        # Apply generic mappings for columns not in format spec
        generic_mappings = {
            'datetime': 'trade_date',
            'date/time': 'trade_date',
            'trade date': 'trade_date',
            'ticker': 'symbol',
            'instrument': 'symbol',
            'direction': 'side',
            'action': 'side',
            'market pos.': 'side',
            'buy/sell': 'side',
            'type': 'side',
            'qty': 'quantity',
            'size': 'quantity',
            'volume': 'quantity',
            'lots': 'quantity',
            'price': 'entry_price',
            'open price': 'entry_price',
            'close price': 'exit_price',
            'profit': 'pnl',
            'profit/loss': 'pnl',
            'realized pnl': 'pnl',
            'p&l': 'pnl',
            'fees': 'commission',
            'costs': 'commission',
            'open time': 'entry_time',
            'close time': 'exit_time'
        }
        
        for col in df.columns:
            col_lower = col.lower().strip()
            if col not in rename_map and col_lower in generic_mappings:
                rename_map[col] = generic_mappings[col_lower]
        
        if rename_map:
            df = df.rename(columns=rename_map)
            logger.debug(f"Renamed columns: {rename_map}")
        
        return df
    
    def _transform_data_types(self, df: pd.DataFrame, format_spec: CSVFormat) -> pd.DataFrame:
        """Transform data types to standard formats."""
        # Transform dates
        if 'trade_date' in df.columns:
            df['trade_date'] = self._parse_dates(df['trade_date'], format_spec.date_formats)
        
        # Transform times
        for time_col in ['entry_time', 'exit_time']:
            if time_col in df.columns:
                df[time_col] = self._parse_times(df[time_col], df.get('trade_date'))
        
        # Transform numeric columns
        numeric_columns = ['quantity', 'entry_price', 'exit_price', 'pnl', 
                          'commission', 'duration_hours']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Ensure symbol is string
        if 'symbol' in df.columns:
            df['symbol'] = df['symbol'].astype(str).str.strip()
        
        return df
    
    def _parse_dates(self, date_series: pd.Series, date_formats: List[str]) -> pd.Series:
        """Parse dates with multiple format attempts."""
        # First try each specified format
        for date_format in date_formats:
            try:
                return pd.to_datetime(date_series, format=date_format, errors='coerce')
            except:
                continue
        
        # Fallback to pandas auto-detection
        return pd.to_datetime(date_series, errors='coerce')
    
    def _parse_times(self, time_series: pd.Series, date_series: Optional[pd.Series] = None) -> pd.Series:
        """Parse time values, handling various formats."""
        def parse_time_value(val):
            if pd.isna(val):
                return None
            
            # If it's already a time object
            if isinstance(val, time):
                return val
            
            # If it's a datetime, extract time
            if isinstance(val, (datetime, pd.Timestamp)):
                return val.time()
            
            # Try to parse string
            val_str = str(val).strip()
            
            # Common time formats
            time_formats = ['%H:%M:%S', '%H:%M', '%I:%M:%S %p', '%I:%M %p']
            
            for fmt in time_formats:
                try:
                    dt = datetime.strptime(val_str, fmt)
                    return dt.time()
                except:
                    continue
            
            return None
        
        return time_series.apply(parse_time_value)
    
    def _standardize_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize values like side/direction."""
        # Standardize side values
        if 'side' in df.columns:
            df['side'] = df['side'].astype(str).str.lower().str.strip()
            df['side'] = df['side'].map(lambda x: self.side_mapping.get(x, x))
        
        # Standardize symbol format
        if 'symbol' in df.columns:
            # Remove common suffixes/prefixes
            df['symbol'] = df['symbol'].str.replace(r'\.(FX|CFD|CASH|SPOT)$', '', regex=True)
            df['symbol'] = df['symbol'].str.upper()
        
        # Handle negative quantities for sell orders
        if 'quantity' in df.columns and 'side' in df.columns:
            # Make all quantities positive
            df['quantity'] = df['quantity'].abs()
        
        return df
    
    def _calculate_derived_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate fields that can be derived from other fields."""
        # Calculate P&L if not present
        if 'pnl' not in df.columns or df['pnl'].isna().all():
            if all(col in df.columns for col in ['entry_price', 'exit_price', 'quantity', 'side']):
                logger.info("Calculating P&L from entry/exit prices")
                
                def calculate_pnl(row):
                    if pd.isna(row['entry_price']) or pd.isna(row['exit_price']):
                        return np.nan
                    
                    price_diff = row['exit_price'] - row['entry_price']
                    
                    if row['side'] in ['sell', 'short']:
                        price_diff = -price_diff
                    
                    return price_diff * row['quantity']
                
                df['pnl'] = df.apply(calculate_pnl, axis=1)
        
        # Calculate duration if we have entry and exit times
        if all(col in df.columns for col in ['entry_time', 'exit_time']) and 'duration_hours' not in df.columns:
            logger.info("Calculating trade duration")
            
            def calculate_duration(row):
                if pd.isna(row['entry_time']) or pd.isna(row['exit_time']):
                    return np.nan
                
                # For simplicity, assume same day trades
                # In real scenario, would need to consider date changes
                try:
                    entry_dt = datetime.combine(date.today(), row['entry_time'])
                    exit_dt = datetime.combine(date.today(), row['exit_time'])
                    
                    # Handle overnight trades
                    if exit_dt < entry_dt:
                        exit_dt = datetime.combine(date.today(), row['exit_time']) + pd.Timedelta(days=1)
                    
                    duration = (exit_dt - entry_dt).total_seconds() / 3600
                    return duration
                except:
                    return np.nan
            
            df['duration_hours'] = df.apply(calculate_duration, axis=1)
        
        # Set default commission to 0 if not present
        if 'commission' not in df.columns:
            df['commission'] = 0.0
        
        return df
    
    def _select_standard_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select and order standard columns."""
        # Define the standard column order
        standard_order = [
            'trade_date', 'symbol', 'side', 'quantity',
            'entry_price', 'exit_price', 'pnl', 'commission',
            'entry_time', 'exit_time', 'duration_hours'
        ]
        
        # Select columns that exist
        selected_columns = [col for col in standard_order if col in df.columns]
        
        # Add any missing required columns with null values
        for col in standard_order:
            if col not in selected_columns:
                df[col] = np.nan
                selected_columns.append(col)
        
        return df[selected_columns]
    
    def _final_cleanup(self, df: pd.DataFrame) -> pd.DataFrame:
        """Final cleanup and validation."""
        # Remove rows with no trade date
        if 'trade_date' in df.columns:
            df = df[df['trade_date'].notna()]
        
        # Remove rows with no symbol
        if 'symbol' in df.columns:
            df = df[df['symbol'].notna()]
            df = df[df['symbol'] != '']
        
        # Sort by trade date
        if 'trade_date' in df.columns:
            df = df.sort_values('trade_date')
        
        # Reset index
        df = df.reset_index(drop=True)
        
        return df
    
    def to_trades(self, df: pd.DataFrame, strategy_id: int) -> List[Trade]:
        """
        Convert standardized DataFrame to Trade model objects.
        
        Args:
            df: Standardized DataFrame
            strategy_id: Strategy ID to assign to trades
            
        Returns:
            List of Trade objects
        """
        trades = []
        
        for _, row in df.iterrows():
            try:
                trade = Trade(
                    strategy_id=strategy_id,
                    trade_date=row['trade_date'].date() if pd.notna(row['trade_date']) else date.today(),
                    symbol=row['symbol'] if pd.notna(row['symbol']) else 'UNKNOWN',
                    side=row['side'] if pd.notna(row['side']) else 'buy',
                    entry_price=Decimal(str(row['entry_price'])) if pd.notna(row['entry_price']) else None,
                    exit_price=Decimal(str(row['exit_price'])) if pd.notna(row['exit_price']) else None,
                    quantity=Decimal(str(row['quantity'])) if pd.notna(row['quantity']) else None,
                    pnl=Decimal(str(row['pnl'])) if pd.notna(row['pnl']) else None,
                    commission=Decimal(str(row['commission'])) if pd.notna(row['commission']) else Decimal('0'),
                    entry_time=row['entry_time'] if pd.notna(row['entry_time']) else None,
                    exit_time=row['exit_time'] if pd.notna(row['exit_time']) else None,
                    duration_hours=Decimal(str(row['duration_hours'])) if pd.notna(row['duration_hours']) else None
                )
                trades.append(trade)
            except Exception as e:
                logger.error(f"Failed to convert row to Trade: {e}")
                continue
        
        return trades