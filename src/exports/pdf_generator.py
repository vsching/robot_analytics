"""
PDF Report Generator

Professional PDF reports with charts, metrics, and customizable templates.
"""

import logging
from datetime import datetime, date
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import io
import base64

# PDF and plotting imports
try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.backends.backend_pdf import PdfPages
    import seaborn as sns
    import numpy as np
    import pandas as pd
    
    # ReportLab for advanced PDF generation
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib.colors import Color, black, blue, red, green
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image
    from reportlab.platypus.flowables import HRFlowable
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    from reportlab.graphics.shapes import Drawing
    from reportlab.graphics.charts.linecharts import HorizontalLineChart
    from reportlab.graphics.charts.barcharts import VerticalBarChart
    
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

from ..models import Trade, Strategy
from ..analytics import PerformanceMetrics, ConfluenceMetrics, SignalOverlap
from ..services.strategy_manager import StrategyManager
from ..analytics import AnalyticsEngine
from .export_manager import ReportTemplate


logger = logging.getLogger(__name__)


class PDFReportGenerator:
    """Generates professional PDF reports for trading strategy analysis."""
    
    def __init__(self, 
                 strategy_manager: StrategyManager,
                 analytics_engine: AnalyticsEngine,
                 export_dir: str = "exports/pdf"):
        """
        Initialize PDF report generator.
        
        Args:
            strategy_manager: Strategy manager instance
            analytics_engine: Analytics engine instance
            export_dir: Directory for PDF exports
        """
        if not PDF_AVAILABLE:
            raise ImportError("Required packages for PDF generation are not available")
        
        self.strategy_manager = strategy_manager
        self.analytics_engine = analytics_engine
        self.export_dir = Path(export_dir)
        self.export_dir.mkdir(parents=True, exist_ok=True)
        
        # Set matplotlib style
        plt.style.use('default')
        sns.set_palette("husl")
    
    def generate_strategy_report(self, 
                               strategy_id: int,
                               template: Optional[ReportTemplate] = None,
                               include_confluence: bool = False,
                               confluence_data: Optional[Tuple[List[SignalOverlap], ConfluenceMetrics]] = None) -> str:
        """
        Generate comprehensive PDF report for a strategy.
        
        Args:
            strategy_id: Strategy ID to generate report for
            template: Report template configuration
            include_confluence: Whether to include confluence analysis
            confluence_data: Tuple of (overlaps, metrics) for confluence
            
        Returns:
            Path to generated PDF file
        """
        template = template or ReportTemplate()
        
        # Get strategy and data
        strategy = self.strategy_manager.get_strategy(strategy_id)
        if not strategy:
            raise ValueError(f"Strategy {strategy_id} not found")
        
        trades = self.analytics_engine.get_trades_for_strategy(strategy_id)
        metrics = self.analytics_engine.calculate_metrics_for_strategy(strategy_id)
        
        if not trades or not metrics:
            raise ValueError(f"No data found for strategy {strategy.name}")
        
        # Generate filename
        filename = f"strategy_report_{strategy.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        filepath = self.export_dir / filename
        
        # Create PDF document
        doc = SimpleDocTemplate(
            str(filepath),
            pagesize=A4,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=18
        )
        
        # Build story (content)
        story = []
        styles = getSampleStyleSheet()
        
        # Add title page
        self._add_title_page(story, styles, template, strategy)
        
        # Add executive summary
        if template.include_summary:
            self._add_executive_summary(story, styles, strategy, metrics, trades)
        
        # Add detailed metrics
        if template.include_detailed_metrics:
            self._add_detailed_metrics(story, styles, metrics)
        
        # Add charts
        if template.include_charts:
            self._add_performance_charts(story, styles, trades, strategy.name)
        
        # Add confluence analysis
        if include_confluence and confluence_data and template.include_confluence_analysis:
            overlaps, confluence_metrics = confluence_data
            self._add_confluence_analysis(story, styles, overlaps, confluence_metrics)
        
        # Add trade details
        self._add_trade_details(story, styles, trades)
        
        # Build PDF
        doc.build(story)
        
        logger.info(f"Generated PDF report for {strategy.name}: {filepath}")
        return str(filepath)
    
    def generate_multi_strategy_report(self, 
                                     strategy_ids: List[int],
                                     template: Optional[ReportTemplate] = None,
                                     include_comparison: bool = True) -> str:
        """
        Generate comparative PDF report for multiple strategies.
        
        Args:
            strategy_ids: List of strategy IDs
            template: Report template configuration
            include_comparison: Whether to include comparison analysis
            
        Returns:
            Path to generated PDF file
        """
        template = template or ReportTemplate(title="Multi-Strategy Analysis Report")
        
        # Get strategies and data
        strategies = []
        all_metrics = {}
        all_trades = {}
        
        for strategy_id in strategy_ids:
            strategy = self.strategy_manager.get_strategy(strategy_id)
            if strategy:
                strategies.append(strategy)
                trades = self.analytics_engine.get_trades_for_strategy(strategy_id)
                metrics = self.analytics_engine.calculate_metrics_for_strategy(strategy_id)
                
                if trades and metrics:
                    all_trades[strategy_id] = trades
                    all_metrics[strategy_id] = metrics
        
        if not strategies:
            raise ValueError("No valid strategies found")
        
        # Generate filename
        filename = f"multi_strategy_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        filepath = self.export_dir / filename
        
        # Create PDF document
        doc = SimpleDocTemplate(
            str(filepath),
            pagesize=A4,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=18
        )
        
        story = []
        styles = getSampleStyleSheet()
        
        # Add title page
        self._add_multi_strategy_title_page(story, styles, template, strategies)
        
        # Add comparison overview
        if include_comparison:
            self._add_strategy_comparison(story, styles, strategies, all_metrics)
        
        # Add individual strategy sections
        for strategy in strategies:
            if strategy.id in all_metrics:
                self._add_individual_strategy_section(
                    story, styles, strategy, 
                    all_metrics[strategy.id], 
                    all_trades[strategy.id]
                )
        
        # Build PDF
        doc.build(story)
        
        logger.info(f"Generated multi-strategy PDF report: {filepath}")
        return str(filepath)
    
    def _add_title_page(self, story: List, styles: Dict, template: ReportTemplate, strategy: Strategy):
        """Add title page to report."""
        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Title'],
            fontSize=24,
            spaceAfter=30,
            alignment=TA_CENTER
        )
        
        story.append(Paragraph(template.title, title_style))
        
        # Subtitle
        if template.subtitle:
            subtitle_style = ParagraphStyle(
                'CustomSubtitle',
                parent=styles['Heading2'],
                fontSize=16,
                spaceAfter=20,
                alignment=TA_CENTER
            )
            story.append(Paragraph(template.subtitle, subtitle_style))
        
        # Strategy name
        strategy_style = ParagraphStyle(
            'StrategyName',
            parent=styles['Heading1'],
            fontSize=20,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=blue
        )
        story.append(Paragraph(f"Strategy: {strategy.name}", strategy_style))
        
        # Generation info
        story.append(Spacer(1, 0.5*inch))
        
        info_data = [
            ['Report Generated:', datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
            ['Strategy Description:', strategy.description or 'No description provided'],
            ['Generated by:', template.author]
        ]
        
        info_table = Table(info_data, colWidths=[2*inch, 4*inch])
        info_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ]))
        
        story.append(info_table)
        story.append(PageBreak())
    
    def _add_executive_summary(self, story: List, styles: Dict, strategy: Strategy, 
                             metrics: PerformanceMetrics, trades: List[Trade]):
        """Add executive summary section."""
        story.append(Paragraph("Executive Summary", styles['Heading1']))
        story.append(Spacer(1, 12))
        
        # Key metrics table
        summary_data = [
            ['Metric', 'Value'],
            ['Total P&L', f"${float(metrics.pnl_summary.total_pnl):,.2f}"],
            ['Total Trades', str(metrics.trade_statistics.total_trades)],
            ['Win Rate', f"{metrics.trade_statistics.win_rate:.1f}%"],
            ['Profit Factor', f"{metrics.trade_statistics.profit_factor:.2f}"],
            ['Average Trade', f"${float(metrics.pnl_summary.average_pnl):,.2f}"],
            ['Best Trade', f"${float(metrics.pnl_summary.max_pnl):,.2f}"],
            ['Worst Trade', f"${float(metrics.pnl_summary.min_pnl):,.2f}"]
        ]
        
        if metrics.advanced_statistics:
            summary_data.extend([
                ['Sharpe Ratio', f"{metrics.advanced_statistics.sharpe_ratio:.2f}" if metrics.advanced_statistics.sharpe_ratio else "N/A"],
                ['Max Drawdown', f"{metrics.advanced_statistics.max_drawdown:.1%}"],
                ['Sortino Ratio', f"{metrics.advanced_statistics.sortino_ratio:.2f}" if metrics.advanced_statistics.sortino_ratio else "N/A"]
            ])
        
        summary_table = Table(summary_data, colWidths=[2.5*inch, 2*inch])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), Color(0.8, 0.8, 0.8)),
            ('TEXTCOLOR', (0, 0), (-1, 0), black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('GRID', (0, 0), (-1, -1), 1, black)
        ]))
        
        story.append(summary_table)
        story.append(Spacer(1, 20))
        
        # Performance assessment
        story.append(Paragraph("Performance Assessment", styles['Heading2']))
        
        # Generate assessment text based on metrics
        assessment = self._generate_performance_assessment(metrics)
        story.append(Paragraph(assessment, styles['Normal']))
        story.append(PageBreak())
    
    def _add_detailed_metrics(self, story: List, styles: Dict, metrics: PerformanceMetrics):
        """Add detailed metrics section."""
        story.append(Paragraph("Detailed Performance Metrics", styles['Heading1']))
        story.append(Spacer(1, 12))
        
        # Trade Statistics
        story.append(Paragraph("Trade Statistics", styles['Heading2']))
        
        trade_data = [
            ['Metric', 'Value'],
            ['Total Trades', str(metrics.trade_statistics.total_trades)],
            ['Winning Trades', str(metrics.trade_statistics.winning_trades)],
            ['Losing Trades', str(metrics.trade_statistics.losing_trades)],
            ['Win Rate', f"{metrics.trade_statistics.win_rate:.1f}%"],
            ['Average Win', f"${float(metrics.trade_statistics.average_win):,.2f}"],
            ['Average Loss', f"${float(metrics.trade_statistics.average_loss):,.2f}"],
            ['Largest Win', f"${float(metrics.pnl_summary.max_pnl):,.2f}"],
            ['Largest Loss', f"${float(metrics.pnl_summary.min_pnl):,.2f}"],
            ['Profit Factor', f"{metrics.trade_statistics.profit_factor:.2f}"],
            ['Max Consecutive Wins', str(metrics.trade_statistics.max_consecutive_wins)],
            ['Max Consecutive Losses', str(metrics.trade_statistics.max_consecutive_losses)]
        ]
        
        trade_table = Table(trade_data, colWidths=[2.5*inch, 2*inch])
        trade_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), Color(0.8, 0.8, 0.8)),
            ('TEXTCOLOR', (0, 0), (-1, 0), black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('GRID', (0, 0), (-1, -1), 1, black)
        ]))
        
        story.append(trade_table)
        story.append(Spacer(1, 20))
        
        # Advanced Statistics
        if metrics.advanced_statistics:
            story.append(Paragraph("Risk-Adjusted Metrics", styles['Heading2']))
            
            advanced_data = [
                ['Metric', 'Value', 'Description'],
                ['Sharpe Ratio', f"{metrics.advanced_statistics.sharpe_ratio:.2f}" if metrics.advanced_statistics.sharpe_ratio else "N/A", 'Risk-adjusted return measure'],
                ['Sortino Ratio', f"{metrics.advanced_statistics.sortino_ratio:.2f}" if metrics.advanced_statistics.sortino_ratio else "N/A", 'Downside deviation adjusted return'],
                ['Calmar Ratio', f"{metrics.advanced_statistics.calmar_ratio:.2f}" if metrics.advanced_statistics.calmar_ratio else "N/A", 'Return vs maximum drawdown'],
                ['Maximum Drawdown', f"{metrics.advanced_statistics.max_drawdown:.1%}", 'Largest peak-to-trough decline'],
                ['Drawdown Duration', f"{metrics.advanced_statistics.max_drawdown_duration or 0} days", 'Longest drawdown period'],
                ['Recovery Duration', f"{metrics.advanced_statistics.recovery_duration or 0} days", 'Time to recover from max drawdown'],
                ['Value at Risk (95%)', f"{metrics.advanced_statistics.value_at_risk_95:.2%}" if metrics.advanced_statistics.value_at_risk_95 else "N/A", '95% confidence loss threshold'],
                ['Conditional VaR (95%)', f"{metrics.advanced_statistics.conditional_var_95:.2%}" if metrics.advanced_statistics.conditional_var_95 else "N/A", 'Expected loss beyond VaR']
            ]
            
            advanced_table = Table(advanced_data, colWidths=[2*inch, 1.5*inch, 2.5*inch])
            advanced_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), Color(0.8, 0.8, 0.8)),
                ('TEXTCOLOR', (0, 0), (-1, 0), black),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 8),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                ('GRID', (0, 0), (-1, -1), 1, black)
            ]))
            
            story.append(advanced_table)
        
        story.append(PageBreak())
    
    def _add_performance_charts(self, story: List, styles: Dict, trades: List[Trade], strategy_name: str):
        """Add performance charts section."""
        story.append(Paragraph("Performance Charts", styles['Heading1']))
        story.append(Spacer(1, 12))
        
        # Create charts and save as images
        chart_paths = []
        
        try:
            # P&L Chart
            pnl_path = self._create_pnl_chart(trades, strategy_name)
            if pnl_path:
                chart_paths.append(('Daily P&L Performance', pnl_path))
            
            # Cumulative Returns Chart
            cumulative_path = self._create_cumulative_chart(trades, strategy_name)
            if cumulative_path:
                chart_paths.append(('Cumulative Returns', cumulative_path))
            
            # Drawdown Chart
            drawdown_path = self._create_drawdown_chart(trades, strategy_name)
            if drawdown_path:
                chart_paths.append(('Drawdown Analysis', drawdown_path))
            
            # Add charts to PDF
            for chart_title, chart_path in chart_paths:
                story.append(Paragraph(chart_title, styles['Heading2']))
                story.append(Spacer(1, 6))
                
                # Add image
                img = Image(chart_path, width=6*inch, height=4*inch)
                story.append(img)
                story.append(Spacer(1, 20))
                
                # Clean up temporary file
                Path(chart_path).unlink(missing_ok=True)
        
        except Exception as e:
            logger.error(f"Error creating charts: {str(e)}")
            story.append(Paragraph("Charts could not be generated due to technical issues.", styles['Normal']))
        
        story.append(PageBreak())
    
    def _create_pnl_chart(self, trades: List[Trade], strategy_name: str) -> Optional[str]:
        """Create P&L chart and return path to image."""
        try:
            # Aggregate daily P&L
            daily_pnl = {}
            for trade in trades:
                if trade.pnl:
                    date_key = trade.trade_date
                    if date_key in daily_pnl:
                        daily_pnl[date_key] += float(trade.pnl)
                    else:
                        daily_pnl[date_key] = float(trade.pnl)
            
            if not daily_pnl:
                return None
            
            # Create chart
            plt.figure(figsize=(10, 6))
            dates = sorted(daily_pnl.keys())
            pnls = [daily_pnl[date] for date in dates]
            
            colors = ['green' if pnl >= 0 else 'red' for pnl in pnls]
            plt.bar(dates, pnls, color=colors, alpha=0.7)
            
            plt.title(f'Daily P&L - {strategy_name}', fontsize=14, fontweight='bold')
            plt.xlabel('Date')
            plt.ylabel('P&L ($)')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            
            # Format dates on x-axis
            if len(dates) > 10:
                plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
                plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            
            plt.tight_layout()
            
            # Save to temporary file
            temp_path = self.export_dir / f"temp_pnl_{datetime.now().timestamp()}.png"
            plt.savefig(temp_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return str(temp_path)
        
        except Exception as e:
            logger.error(f"Error creating P&L chart: {str(e)}")
            plt.close()
            return None
    
    def _create_cumulative_chart(self, trades: List[Trade], strategy_name: str) -> Optional[str]:
        """Create cumulative returns chart and return path to image."""
        try:
            # Sort trades by date
            sorted_trades = sorted(trades, key=lambda t: t.trade_date)
            
            dates = []
            cumulative_pnl = []
            current_pnl = 0
            
            for trade in sorted_trades:
                if trade.pnl:
                    dates.append(trade.trade_date)
                    current_pnl += float(trade.pnl)
                    cumulative_pnl.append(current_pnl)
            
            if not dates:
                return None
            
            # Create chart
            plt.figure(figsize=(10, 6))
            plt.plot(dates, cumulative_pnl, linewidth=2, color='blue')
            plt.fill_between(dates, cumulative_pnl, alpha=0.3, color='blue')
            
            plt.title(f'Cumulative Returns - {strategy_name}', fontsize=14, fontweight='bold')
            plt.xlabel('Date')
            plt.ylabel('Cumulative P&L ($)')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            
            # Format dates on x-axis
            if len(dates) > 10:
                plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
                plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            
            plt.tight_layout()
            
            # Save to temporary file
            temp_path = self.export_dir / f"temp_cumulative_{datetime.now().timestamp()}.png"
            plt.savefig(temp_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return str(temp_path)
        
        except Exception as e:
            logger.error(f"Error creating cumulative chart: {str(e)}")
            plt.close()
            return None
    
    def _create_drawdown_chart(self, trades: List[Trade], strategy_name: str) -> Optional[str]:
        """Create drawdown chart and return path to image."""
        try:
            # Calculate drawdown series
            sorted_trades = sorted(trades, key=lambda t: t.trade_date)
            
            dates = []
            cumulative_pnl = []
            peak = 0
            drawdown = []
            current_pnl = 0
            
            for trade in sorted_trades:
                if trade.pnl:
                    dates.append(trade.trade_date)
                    current_pnl += float(trade.pnl)
                    cumulative_pnl.append(current_pnl)
                    
                    # Update peak
                    if current_pnl > peak:
                        peak = current_pnl
                    
                    # Calculate drawdown
                    dd = (current_pnl - peak) / (peak if peak != 0 else 1)
                    drawdown.append(dd)
            
            if not dates:
                return None
            
            # Create chart
            plt.figure(figsize=(10, 6))
            plt.fill_between(dates, drawdown, 0, alpha=0.7, color='red', label='Drawdown')
            plt.plot(dates, drawdown, linewidth=1, color='darkred')
            
            plt.title(f'Drawdown Analysis - {strategy_name}', fontsize=14, fontweight='bold')
            plt.xlabel('Date')
            plt.ylabel('Drawdown (%)')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            
            # Format y-axis as percentage
            plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))
            
            # Format dates on x-axis
            if len(dates) > 10:
                plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
                plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            
            plt.tight_layout()
            
            # Save to temporary file
            temp_path = self.export_dir / f"temp_drawdown_{datetime.now().timestamp()}.png"
            plt.savefig(temp_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return str(temp_path)
        
        except Exception as e:
            logger.error(f"Error creating drawdown chart: {str(e)}")
            plt.close()
            return None
    
    def _add_confluence_analysis(self, story: List, styles: Dict, 
                               overlaps: List[SignalOverlap], 
                               confluence_metrics: ConfluenceMetrics):
        """Add confluence analysis section."""
        story.append(Paragraph("Signal Confluence Analysis", styles['Heading1']))
        story.append(Spacer(1, 12))
        
        # Overview
        overview_text = f"""
        Confluence analysis examines when multiple strategies generate signals simultaneously.
        A total of {confluence_metrics.total_overlaps} confluence events were detected.
        """
        story.append(Paragraph(overview_text, styles['Normal']))
        story.append(Spacer(1, 12))
        
        # Confluence metrics table
        confluence_data = [
            ['Metric', 'Confluence Trades', 'Individual Trades', 'Advantage'],
            ['Total Events', str(confluence_metrics.total_overlaps), '-', '-'],
            ['Win Rate', f"{confluence_metrics.overlap_win_rate:.1%}", f"{confluence_metrics.individual_win_rate:.1%}", f"{confluence_metrics.overlap_win_rate - confluence_metrics.individual_win_rate:+.1%}"],
            ['Avg P&L', f"${confluence_metrics.overlap_avg_pnl:.2f}", f"${confluence_metrics.individual_avg_pnl:.2f}", f"${confluence_metrics.overlap_avg_pnl - confluence_metrics.individual_avg_pnl:+.2f}"],
            ['Performance Advantage', '-', '-', f"{confluence_metrics.confluence_advantage:+.1f}%"]
        ]
        
        confluence_table = Table(confluence_data, colWidths=[2*inch, 1.5*inch, 1.5*inch, 1.5*inch])
        confluence_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), Color(0.8, 0.8, 0.8)),
            ('TEXTCOLOR', (0, 0), (-1, 0), black),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('GRID', (0, 0), (-1, -1), 1, black)
        ]))
        
        story.append(confluence_table)
        story.append(Spacer(1, 20))
        
        # Best combinations
        if confluence_metrics.best_confluence_strategies:
            story.append(Paragraph("Top Performing Strategy Combinations", styles['Heading2']))
            
            combo_data = [['Rank', 'Strategy Combination', 'Avg P&L']]
            for i, (strategies, avg_pnl) in enumerate(confluence_metrics.best_confluence_strategies[:5], 1):
                combo_name = " + ".join(strategies)
                combo_data.append([str(i), combo_name, f"${avg_pnl:.2f}"])
            
            combo_table = Table(combo_data, colWidths=[0.5*inch, 4*inch, 1.5*inch])
            combo_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), Color(0.8, 0.8, 0.8)),
                ('TEXTCOLOR', (0, 0), (-1, 0), black),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                ('GRID', (0, 0), (-1, -1), 1, black)
            ]))
            
            story.append(combo_table)
        
        story.append(PageBreak())
    
    def _add_trade_details(self, story: List, styles: Dict, trades: List[Trade]):
        """Add trade details section."""
        story.append(Paragraph("Trade Details", styles['Heading1']))
        story.append(Spacer(1, 12))
        
        # Limit to recent trades for space
        recent_trades = sorted(trades, key=lambda t: t.trade_date, reverse=True)[:50]
        
        trade_data = [['Date', 'Symbol', 'Side', 'Entry', 'Exit', 'Qty', 'P&L']]
        
        for trade in recent_trades:
            trade_data.append([
                trade.trade_date.strftime('%Y-%m-%d'),
                trade.symbol or '',
                trade.side or '',
                f"${float(trade.entry_price or 0):.2f}",
                f"${float(trade.exit_price or 0):.2f}",
                str(float(trade.quantity or 0)),
                f"${float(trade.pnl or 0):.2f}"
            ])
        
        trade_table = Table(trade_data, colWidths=[1*inch, 1*inch, 0.7*inch, 1*inch, 1*inch, 0.8*inch, 1*inch])
        trade_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), Color(0.8, 0.8, 0.8)),
            ('TEXTCOLOR', (0, 0), (-1, 0), black),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ('GRID', (0, 0), (-1, -1), 1, black)
        ]))
        
        story.append(trade_table)
        
        if len(trades) > 50:
            story.append(Spacer(1, 12))
            story.append(Paragraph(f"Note: Showing most recent 50 trades out of {len(trades)} total trades.", styles['Normal']))
    
    def _generate_performance_assessment(self, metrics: PerformanceMetrics) -> str:
        """Generate performance assessment text."""
        assessment_parts = []
        
        # Overall performance
        total_pnl = float(metrics.pnl_summary.total_pnl)
        if total_pnl > 0:
            assessment_parts.append(f"The strategy generated positive returns of ${total_pnl:,.2f}")
        else:
            assessment_parts.append(f"The strategy experienced losses of ${abs(total_pnl):,.2f}")
        
        # Win rate assessment
        win_rate = metrics.trade_statistics.win_rate
        if win_rate >= 60:
            assessment_parts.append(f"with a strong win rate of {win_rate:.1f}%")
        elif win_rate >= 50:
            assessment_parts.append(f"with a moderate win rate of {win_rate:.1f}%")
        else:
            assessment_parts.append(f"with a challenging win rate of {win_rate:.1f}%")
        
        # Profit factor assessment
        profit_factor = metrics.trade_statistics.profit_factor
        if profit_factor > 1.5:
            assessment_parts.append("The profit factor indicates strong profitability per dollar risked.")
        elif profit_factor > 1.0:
            assessment_parts.append("The profit factor shows modest profitability.")
        else:
            assessment_parts.append("The profit factor indicates losses exceeded gains.")
        
        # Risk assessment
        if metrics.advanced_statistics:
            sharpe = metrics.advanced_statistics.sharpe_ratio
            if sharpe and sharpe > 1.0:
                assessment_parts.append("Risk-adjusted returns appear favorable based on the Sharpe ratio.")
            elif sharpe and sharpe > 0:
                assessment_parts.append("Risk-adjusted returns show modest performance.")
        
        return " ".join(assessment_parts)
    
    def _add_multi_strategy_title_page(self, story: List, styles: Dict, 
                                     template: ReportTemplate, strategies: List[Strategy]):
        """Add title page for multi-strategy report."""
        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Title'],
            fontSize=24,
            spaceAfter=30,
            alignment=TA_CENTER
        )
        
        story.append(Paragraph(template.title, title_style))
        
        # Strategy list
        strategy_style = ParagraphStyle(
            'StrategyList',
            parent=styles['Heading2'],
            fontSize=14,
            spaceAfter=20,
            alignment=TA_CENTER
        )
        
        strategy_names = ", ".join([s.name for s in strategies])
        story.append(Paragraph(f"Strategies: {strategy_names}", strategy_style))
        
        # Generation info
        story.append(Spacer(1, 0.5*inch))
        
        info_data = [
            ['Report Generated:', datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
            ['Number of Strategies:', str(len(strategies))],
            ['Generated by:', template.author]
        ]
        
        info_table = Table(info_data, colWidths=[2*inch, 4*inch])
        info_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ]))
        
        story.append(info_table)
        story.append(PageBreak())
    
    def _add_strategy_comparison(self, story: List, styles: Dict, 
                               strategies: List[Strategy], 
                               all_metrics: Dict[int, PerformanceMetrics]):
        """Add strategy comparison section."""
        story.append(Paragraph("Strategy Comparison", styles['Heading1']))
        story.append(Spacer(1, 12))
        
        # Comparison table
        comparison_data = [
            ['Strategy', 'Total P&L', 'Trades', 'Win Rate', 'Profit Factor', 'Sharpe Ratio']
        ]
        
        for strategy in strategies:
            if strategy.id in all_metrics:
                metrics = all_metrics[strategy.id]
                comparison_data.append([
                    strategy.name,
                    f"${float(metrics.pnl_summary.total_pnl):,.2f}",
                    str(metrics.trade_statistics.total_trades),
                    f"{metrics.trade_statistics.win_rate:.1f}%",
                    f"{metrics.trade_statistics.profit_factor:.2f}",
                    f"{metrics.advanced_statistics.sharpe_ratio:.2f}" if metrics.advanced_statistics and metrics.advanced_statistics.sharpe_ratio else "N/A"
                ])
        
        comparison_table = Table(comparison_data, colWidths=[2*inch, 1*inch, 0.8*inch, 0.8*inch, 1*inch, 1*inch])
        comparison_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), Color(0.8, 0.8, 0.8)),
            ('TEXTCOLOR', (0, 0), (-1, 0), black),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('GRID', (0, 0), (-1, -1), 1, black)
        ]))
        
        story.append(comparison_table)
        story.append(PageBreak())
    
    def _add_individual_strategy_section(self, story: List, styles: Dict, 
                                       strategy: Strategy, 
                                       metrics: PerformanceMetrics,
                                       trades: List[Trade]):
        """Add individual strategy section to multi-strategy report."""
        story.append(Paragraph(f"Strategy: {strategy.name}", styles['Heading1']))
        story.append(Spacer(1, 12))
        
        # Brief summary
        summary_text = f"""
        {strategy.description or 'No description provided.'}
        
        Total trades: {metrics.trade_statistics.total_trades}
        Total P&L: ${float(metrics.pnl_summary.total_pnl):,.2f}
        Win rate: {metrics.trade_statistics.win_rate:.1f}%
        """
        
        story.append(Paragraph(summary_text, styles['Normal']))
        story.append(Spacer(1, 20))
        story.append(PageBreak())