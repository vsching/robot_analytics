"""CSV format definitions for various trading platforms."""

from typing import Dict, List, Optional, Set
from dataclasses import dataclass, field
from enum import Enum


class Platform(Enum):
    """Supported trading platforms."""
    TRADINGVIEW = "TradingView"
    METATRADER4 = "MetaTrader4"
    METATRADER5 = "MetaTrader5"
    CTRADER = "cTrader"
    NINJATRADER = "NinjaTrader"
    INTERACTIVE_BROKERS = "InteractiveBrokers"
    GENERIC = "Generic"
    UNKNOWN = "Unknown"


@dataclass
class CSVFormat:
    """Defines the format specification for a trading platform CSV."""
    
    platform: Platform
    name: str
    description: str
    required_columns: Set[str] = field(default_factory=set)
    optional_columns: Set[str] = field(default_factory=set)
    column_mappings: Dict[str, str] = field(default_factory=dict)  # platform_col -> standard_col
    date_columns: List[str] = field(default_factory=list)
    date_formats: List[str] = field(default_factory=list)
    delimiter: str = ","
    encoding: str = "utf-8"
    has_header: bool = True
    
    def get_all_columns(self) -> Set[str]:
        """Get all possible columns for this format."""
        return self.required_columns.union(self.optional_columns)
    
    def map_column(self, platform_column: str) -> str:
        """Map platform-specific column name to standard name."""
        return self.column_mappings.get(platform_column, platform_column)


# Define format specifications for each platform
FORMATS = {
    Platform.TRADINGVIEW: CSVFormat(
        platform=Platform.TRADINGVIEW,
        name="TradingView",
        description="TradingView strategy tester export format",
        required_columns={"Date/Time", "Symbol", "Side", "Qty", "Price"},
        optional_columns={"Commission", "Comment", "Signal"},
        column_mappings={
            "Date/Time": "trade_date",
            "Symbol": "symbol",
            "Side": "side",
            "Qty": "quantity",
            "Price": "entry_price",
            "Commission": "commission"
        },
        date_columns=["Date/Time"],
        date_formats=["%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"]
    ),
    
    Platform.METATRADER4: CSVFormat(
        platform=Platform.METATRADER4,
        name="MetaTrader 4",
        description="MetaTrader 4 history export format",
        required_columns={"Ticket", "Open Time", "Type", "Size", "Symbol", "Price"},
        optional_columns={"S/L", "T/P", "Close Time", "Price", "Commission", "Swap", "Profit"},
        column_mappings={
            "Ticket": "trade_id",
            "Open Time": "entry_time",
            "Close Time": "exit_time",
            "Type": "side",
            "Size": "quantity",
            "Symbol": "symbol",
            "Price": "entry_price",
            "Close Price": "exit_price",
            "Commission": "commission",
            "Profit": "pnl"
        },
        date_columns=["Open Time", "Close Time"],
        date_formats=["%Y.%m.%d %H:%M:%S", "%Y.%m.%d %H:%M", "%d.%m.%Y %H:%M:%S"]
    ),
    
    Platform.METATRADER5: CSVFormat(
        platform=Platform.METATRADER5,
        name="MetaTrader 5",
        description="MetaTrader 5 history export format",
        required_columns={"Time", "Deal", "Symbol", "Type", "Direction", "Volume", "Price"},
        optional_columns={"Order", "Commission", "Swap", "Profit", "Balance", "Comment"},
        column_mappings={
            "Deal": "trade_id",
            "Time": "trade_date",
            "Symbol": "symbol",
            "Direction": "side",
            "Volume": "quantity",
            "Price": "entry_price",
            "Commission": "commission",
            "Profit": "pnl"
        },
        date_columns=["Time"],
        date_formats=["%Y.%m.%d %H:%M:%S", "%Y-%m-%d %H:%M:%S"]
    ),
    
    Platform.CTRADER: CSVFormat(
        platform=Platform.CTRADER,
        name="cTrader",
        description="cTrader history export format",
        required_columns={"Date", "Symbol", "Action", "Volume", "Price"},
        optional_columns={"SL", "TP", "Commission", "Swap", "Profit"},
        column_mappings={
            "Date": "trade_date",
            "Symbol": "symbol",
            "Action": "side",
            "Volume": "quantity",
            "Price": "entry_price",
            "Commission": "commission",
            "Profit": "pnl"
        },
        date_columns=["Date"],
        date_formats=["%d/%m/%Y %H:%M:%S", "%Y-%m-%d %H:%M:%S"]
    ),
    
    Platform.NINJATRADER: CSVFormat(
        platform=Platform.NINJATRADER,
        name="NinjaTrader",
        description="NinjaTrader trade performance export format",
        required_columns={"Entry time", "Exit time", "Instrument", "Market pos.", "Qty", "Entry price", "Exit price"},
        optional_columns={"Profit", "Cum. profit", "Commission", "MAE", "MFE"},
        column_mappings={
            "Entry time": "entry_time",
            "Exit time": "exit_time",
            "Instrument": "symbol",
            "Market pos.": "side",
            "Qty": "quantity",
            "Entry price": "entry_price",
            "Exit price": "exit_price",
            "Profit": "pnl",
            "Commission": "commission"
        },
        date_columns=["Entry time", "Exit time"],
        date_formats=["%m/%d/%Y %I:%M:%S %p", "%Y-%m-%d %H:%M:%S"]
    ),
    
    Platform.INTERACTIVE_BROKERS: CSVFormat(
        platform=Platform.INTERACTIVE_BROKERS,
        name="Interactive Brokers",
        description="Interactive Brokers Flex Query export format",
        required_columns={"TradeDate", "Symbol", "Buy/Sell", "Quantity", "TradePrice"},
        optional_columns={"Commission", "RealizedPnL", "Codes", "OrderTime", "TradeTime"},
        column_mappings={
            "TradeDate": "trade_date",
            "Symbol": "symbol",
            "Buy/Sell": "side",
            "Quantity": "quantity",
            "TradePrice": "entry_price",
            "Commission": "commission",
            "RealizedPnL": "pnl",
            "TradeTime": "entry_time"
        },
        date_columns=["TradeDate", "OrderTime", "TradeTime"],
        date_formats=["%Y%m%d", "%Y%m%d;%H%M%S", "%Y-%m-%d %H:%M:%S"]
    ),
    
    Platform.GENERIC: CSVFormat(
        platform=Platform.GENERIC,
        name="Generic",
        description="Generic CSV format with flexible column names",
        required_columns={"date", "symbol", "side", "quantity", "price"},
        optional_columns={"pnl", "commission", "entry_price", "exit_price", "entry_time", "exit_time"},
        column_mappings={
            "date": "trade_date",
            "datetime": "trade_date",
            "trade_date": "trade_date",
            "symbol": "symbol",
            "ticker": "symbol",
            "side": "side",
            "direction": "side",
            "quantity": "quantity",
            "qty": "quantity",
            "size": "quantity",
            "volume": "quantity",
            "price": "entry_price",
            "entry_price": "entry_price",
            "exit_price": "exit_price",
            "pnl": "pnl",
            "profit": "pnl",
            "profit_loss": "pnl",
            "commission": "commission",
            "fees": "commission"
        },
        date_columns=["date", "datetime", "trade_date", "entry_time", "exit_time"],
        date_formats=[
            "%Y-%m-%d",
            "%Y-%m-%d %H:%M:%S",
            "%Y/%m/%d",
            "%Y/%m/%d %H:%M:%S",
            "%d/%m/%Y",
            "%d/%m/%Y %H:%M:%S",
            "%m/%d/%Y",
            "%m/%d/%Y %H:%M:%S",
            "%d-%m-%Y",
            "%d-%m-%Y %H:%M:%S"
        ]
    )
}


def get_format(platform: Platform) -> CSVFormat:
    """Get format specification for a platform."""
    return FORMATS.get(platform, FORMATS[Platform.GENERIC])