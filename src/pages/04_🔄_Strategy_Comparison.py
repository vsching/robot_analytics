"""
Strategy Comparison Page

This page enables comparison of multiple trading strategies with correlation analysis,
relative performance charts, and portfolio optimization suggestions.
"""

import streamlit as st
from typing import List

from src.db.connection import get_db_manager
from src.services.strategy_manager import StrategyManager
from src.analytics import AnalyticsEngine, CachedAnalyticsEngine, MetricsCacheManager
from src.components import VisualizationComponents
from src.components.comparison_dashboard import ComparisonDashboard


def main():
    st.set_page_config(
        page_title="Strategy Comparison - Trading Strategy Analyzer",
        page_icon="🔄",
        layout="wide"
    )
    
    st.title("🔄 Multi-Strategy Comparison")
    st.markdown("Compare multiple strategies side-by-side with correlation analysis and portfolio optimization")
    
    # Initialize components
    db_manager = get_db_manager()
    strategy_manager = StrategyManager(db_manager)
    cache_manager = MetricsCacheManager(db_manager)
    analytics_engine = AnalyticsEngine(db_manager)
    cached_analytics = CachedAnalyticsEngine(analytics_engine, cache_manager)
    viz = VisualizationComponents(theme="dark")
    
    # Create comparison dashboard
    comparison_dashboard = ComparisonDashboard(
        strategy_manager=strategy_manager,
        analytics_engine=cached_analytics,
        viz_components=viz
    )
    
    # Strategy selection
    selected_strategy_ids = comparison_dashboard.render_strategy_selector(
        min_strategies=2, 
        max_strategies=10
    )
    
    if len(selected_strategy_ids) >= 2:
        # Build comparison metrics table
        st.divider()
        st.subheader("📊 Performance Comparison")
        
        comparison_df = comparison_dashboard.build_comparison_table(selected_strategy_ids)
        
        if not comparison_df.empty:
            # Display metrics with rankings
            tab1, tab2, tab3 = st.tabs(["📋 Comparison Table", "📊 Rankings", "🎯 Key Metrics"])
            
            with tab1:
                # Format the comparison table
                formatted_df = comparison_df.copy()
                
                # Format numeric columns
                format_dict = {
                    'Total P&L': '${:,.2f}',
                    'Win Rate': '{:.1f}%',
                    'Avg Win': '${:,.2f}',
                    'Avg Loss': '${:,.2f}',
                    'Profit Factor': '{:.2f}',
                    'Sharpe Ratio': '{:.2f}',
                    'Sortino Ratio': '{:.2f}',
                    'Max Drawdown': '{:.1%}',
                    'Calmar Ratio': '{:.2f}'
                }
                
                # Apply formatting
                for col, fmt in format_dict.items():
                    if col in formatted_df.columns:
                        formatted_df[col] = formatted_df[col].apply(lambda x: fmt.format(x) if x != 0 else 'N/A')
                
                # Display only non-rank columns
                display_cols = [col for col in formatted_df.columns if '_rank' not in col]
                st.dataframe(
                    formatted_df[display_cols],
                    use_container_width=True,
                    hide_index=True
                )
                
                # Download button
                csv = comparison_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Comparison Data",
                    data=csv,
                    file_name="strategy_comparison.csv",
                    mime="text/csv"
                )
            
            with tab2:
                # Ranking visualizations
                col1, col2 = st.columns(2)
                
                with col1:
                    sort_metric = st.selectbox(
                        "Sort by Metric",
                        ["Total P&L", "Win Rate", "Sharpe Ratio", "Profit Factor", "Max Drawdown"],
                        index=0
                    )
                
                ranking_fig = comparison_dashboard.render_performance_ranking(
                    comparison_df, 
                    sort_by=sort_metric
                )
                st.plotly_chart(ranking_fig, use_container_width=True)
            
            with tab3:
                # Key metrics overview
                st.markdown("### 🏆 Top Performers")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Highest P&L",
                        comparison_df.loc[comparison_df['Total P&L'].idxmax(), 'Strategy'],
                        f"${comparison_df['Total P&L'].max():,.2f}"
                    )
                
                with col2:
                    st.metric(
                        "Best Win Rate",
                        comparison_df.loc[comparison_df['Win Rate'].idxmax(), 'Strategy'],
                        f"{comparison_df['Win Rate'].max():.1f}%"
                    )
                
                with col3:
                    if 'Sharpe Ratio' in comparison_df.columns:
                        best_sharpe_idx = comparison_df['Sharpe Ratio'].idxmax()
                        st.metric(
                            "Best Sharpe Ratio",
                            comparison_df.loc[best_sharpe_idx, 'Strategy'],
                            f"{comparison_df['Sharpe Ratio'].max():.2f}"
                        )
        
        # Correlation Analysis
        st.divider()
        st.subheader("🔗 Correlation Analysis")
        
        correlation_matrix, returns_df = comparison_dashboard.calculate_correlation_matrix(selected_strategy_ids)
        
        if not correlation_matrix.empty:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Correlation heatmap
                corr_fig = comparison_dashboard.render_correlation_heatmap(correlation_matrix)
                st.plotly_chart(corr_fig, use_container_width=True)
            
            with col2:
                # Correlation insights
                st.markdown("### 📊 Correlation Insights")
                
                # Find highest and lowest correlations
                corr_values = []
                for i in range(len(correlation_matrix)):
                    for j in range(i+1, len(correlation_matrix)):
                        corr_values.append({
                            'pair': f"{correlation_matrix.index[i]} - {correlation_matrix.columns[j]}",
                            'correlation': correlation_matrix.iloc[i, j]
                        })
                
                if corr_values:
                    sorted_corr = sorted(corr_values, key=lambda x: x['correlation'])
                    
                    st.markdown("**Most Negatively Correlated:**")
                    st.info(f"{sorted_corr[0]['pair']}: {sorted_corr[0]['correlation']:.3f}")
                    
                    st.markdown("**Most Positively Correlated:**")
                    st.info(f"{sorted_corr[-1]['pair']}: {sorted_corr[-1]['correlation']:.3f}")
                    
                    # Portfolio diversification suggestion
                    avg_correlation = correlation_matrix.values[correlation_matrix.values != 1].mean()
                    if avg_correlation < 0.3:
                        st.success("✅ Low average correlation indicates good diversification potential")
                    elif avg_correlation < 0.6:
                        st.warning("⚠️ Moderate correlation - consider adding more diverse strategies")
                    else:
                        st.error("❌ High correlation - strategies may not provide good diversification")
        
        # Relative Performance
        st.divider()
        st.subheader("📈 Relative Performance")
        
        col1, col2 = st.columns([3, 1])
        
        with col2:
            initial_capital = st.number_input(
                "Initial Capital ($)",
                min_value=1000,
                max_value=1000000,
                value=10000,
                step=1000
            )
        
        with col1:
            perf_fig = comparison_dashboard.render_relative_performance(
                selected_strategy_ids,
                initial_capital=initial_capital
            )
            st.plotly_chart(perf_fig, use_container_width=True)
        
        # Portfolio Optimization
        st.divider()
        st.subheader("🎯 Portfolio Optimization")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("### Optimization Settings")
            
            optimization_type = st.radio(
                "Optimization Goal",
                ["Maximum Sharpe Ratio", "Target Return"],
                index=0
            )
            
            target_return = None
            if optimization_type == "Target Return":
                target_return = st.slider(
                    "Target Annual Return (%)",
                    min_value=0.0,
                    max_value=50.0,
                    value=10.0,
                    step=0.5
                ) / 100
            
            if st.button("🚀 Optimize Portfolio", type="primary"):
                with st.spinner("Calculating optimal portfolio weights..."):
                    optimization_results = comparison_dashboard.calculate_portfolio_optimization(
                        selected_strategy_ids,
                        target_return=target_return
                    )
                    
                    if optimization_results['status'] == 'success':
                        st.success("✅ Portfolio optimization completed successfully!")
                        
                        # Store in session state
                        st.session_state['optimization_results'] = optimization_results
                    else:
                        st.error(f"❌ {optimization_results.get('message', 'Optimization failed')}")
        
        with col2:
            if 'optimization_results' in st.session_state:
                results = st.session_state['optimization_results']
                
                if results['status'] == 'success':
                    # Portfolio weights visualization
                    weights_fig = comparison_dashboard.render_portfolio_weights(results)
                    st.plotly_chart(weights_fig, use_container_width=True)
                    
                    # Optimization results
                    st.markdown("### 📊 Optimization Results")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Expected Return", f"{results['expected_return']*252:.1%} annual")
                    with col2:
                        st.metric("Expected Volatility", f"{results['expected_volatility']*16:.1%} annual")
                    with col3:
                        st.metric("Sharpe Ratio", f"{results['sharpe_ratio']:.2f}")
                    
                    # Weight allocation table
                    with st.expander("📋 Detailed Weight Allocation"):
                        weights_data = []
                        for strategy_name, weight in results['weights'].items():
                            weights_data.append({
                                'Strategy': strategy_name,
                                'Weight': f"{weight:.1%}",
                                'Dollar Allocation': f"${initial_capital * weight:,.2f}"
                            })
                        
                        st.dataframe(weights_data, use_container_width=True, hide_index=True)
                    
                    # Implementation guide
                    with st.expander("📖 Implementation Guide"):
                        st.markdown("""
                        ### How to Implement This Portfolio:
                        
                        1. **Allocate Capital**: Divide your initial capital according to the weights shown above
                        2. **Execute Trades**: Follow each strategy's signals proportionally
                        3. **Rebalance**: Review and rebalance quarterly or when weights drift by >5%
                        4. **Monitor**: Track actual performance against expected metrics
                        
                        **Risk Considerations:**
                        - Past performance does not guarantee future results
                        - Correlation patterns may change over time
                        - Consider transaction costs when rebalancing
                        - Maintain adequate cash reserves for drawdowns
                        """)
        
        # Export options
        st.divider()
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📥 Export Comparison Report"):
                st.info("Comparison report export will be implemented in the export module")
        
        with col2:
            if st.button("📊 Export Correlation Data"):
                if not correlation_matrix.empty:
                    csv = correlation_matrix.to_csv()
                    st.download_button(
                        label="Download Correlation Matrix",
                        data=csv,
                        file_name="correlation_matrix.csv",
                        mime="text/csv"
                    )
        
        with col3:
            if st.button("🎯 Export Portfolio Weights"):
                if 'optimization_results' in st.session_state:
                    results = st.session_state['optimization_results']
                    if results['status'] == 'success':
                        import json
                        weights_json = json.dumps(results['weights'], indent=2)
                        st.download_button(
                            label="Download Portfolio Weights",
                            data=weights_json,
                            file_name="portfolio_weights.json",
                            mime="application/json"
                        )
    
    else:
        # Instructions when no strategies are selected
        st.info("👆 Select at least 2 strategies above to begin comparison analysis")
        
        with st.expander("📚 Learn About Strategy Comparison"):
            st.markdown("""
            ### What You Can Do Here:
            
            **1. Performance Comparison**
            - Compare key metrics across multiple strategies
            - Identify top performers by various criteria
            - Spot strengths and weaknesses of each approach
            
            **2. Correlation Analysis**
            - Understand how strategies move together
            - Find diversification opportunities
            - Identify redundant strategies
            
            **3. Portfolio Optimization**
            - Calculate optimal allocation weights
            - Maximize risk-adjusted returns
            - Build diversified strategy portfolios
            
            **4. Relative Performance**
            - Track strategies against each other
            - Identify consistent outperformers
            - Analyze performance patterns
            
            ### Tips for Better Analysis:
            - Compare strategies with similar capital requirements
            - Consider different market conditions
            - Look for negative correlations for hedging
            - Don't over-optimize based on past data
            """)


if __name__ == "__main__":
    main()