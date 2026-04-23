import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from src.extractor.extractor import DataExtractor
from src.database.database import SupabaseDatabase
from src.main import run_pipeline
import datetime

# Page Config
st.set_page_config(
    page_title="Pro-Grade Stock Forecast",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Look
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
        color: #ffffff;
    }
    .stMetric {
        background-color: #1e2130;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #3e4150;
    }
    .stPlotlyChart {
        border-radius: 15px;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)

# Streamlit requires cached inputs to be immutable objects (Tuples, not Lists)
@st.cache_data(ttl=86400, show_spinner=False)
def get_historical_cache(symbols_tuple):
    """Caches Alpha Vantage data for 24 hours to prevent freezes."""
    extractor = DataExtractor(list(symbols_tuple))
    return extractor.fetch_historical_data(period="1y")

def main():
    st.title("📈 AI Stock Forecasting & Portfolio Optimization")
    st.markdown("---")

    # Dynamic Sidebar
    st.sidebar.header("Controls")
    
    # 1. Expanded list of available stocks
    available_tickers = ["AAPL", "GOOG", "PLTR", "JPM", "MSFT", "TSLA", "NVDA", "AMZN", "META"]
    
    # 2. Add an interactive multi-select UI
    symbols = st.sidebar.multiselect(
        "Select Target Stocks:", 
        options=available_tickers, 
        default=["AAPL", "GOOG"]
    )
    
    # 3. Fail-safe if the user unselects everything
    if not symbols:
        st.warning("⚠️ Please select at least one stock to begin.")
        st.stop()

    if st.sidebar.button("🚀 Run Prediction Pipeline"):
        with st.spinner(f"Executing Deep Learning Pipeline for {symbols}..."):
            # Only runs the pipeline on the selected stocks! Huge speed boost.
            run_pipeline(symbols)
            st.sidebar.success("Pipeline executed successfully!")

    # Database connection
    db = SupabaseDatabase()
    
    # Layout
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Historical Performance & Trends")
        # Must pass symbols as a tuple for caching to work
        historical_df = get_historical_cache(tuple(symbols))
        
        if not historical_df.empty:
            # Normalize for comparison
            normalized_df = historical_df / historical_df.iloc[0]
            fig = px.line(normalized_df, labels={"value": "Normalized Price", "index": "Date"})
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Optimized Portfolio Weights")
        if db.client:
            # Only grab weights for the dynamically selected symbols if possible
            weights_data = db.client.table("portfolio_weights").select("*").in_("symbol", symbols).execute().data
            if weights_data:
                weights_df = pd.DataFrame(weights_data)
                fig_pie = px.pie(weights_df, values='weight', names='symbol', hole=.4,
                                 color_discrete_sequence=px.colors.sequential.RdBu)
                fig_pie.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_pie, use_container_width=True)
            else:
                st.info(f"No optimization data yet for {symbols}. Run the pipeline.")
        else:
            st.warning("Supabase not connected. Showing mock weights.")
            
    st.markdown("---")
    st.subheader("1-Week Deep Learning Projective Forecast")
    
    # Dynamically scale the UI columns based on how many stocks they selected
    cols = st.columns(len(symbols))

    if db.client:
        # Fetch future predictions strictly greater than or equal to tomorrow
        today_str = datetime.date.today().isoformat()
        preds = db.client.table("predictions").select("*").gt("prediction_date", today_str).execute().data
        
        # We need to map { 'AAPL': [ {date, price}, ... ] }
        chart_data = {}
        for p in preds:
            sym = p['symbol']
            if sym not in chart_data:
                chart_data[sym] = []
            chart_data[sym].append({"date": p['prediction_date'], "price": p['predicted_price']})
        
        # Draw metric boxes directly into dynamic columns
        for i, symbol in enumerate(symbols):
            with cols[i]:
                st.markdown(f"#### **{symbol} Trajectory**")
                if symbol in chart_data and len(chart_data[symbol]) > 0:
                    df_symbol = pd.DataFrame(chart_data[symbol]).sort_values("date")
                    
                    # Also calculate the next day value for a quick glance
                    next_day_val = df_symbol.iloc[0]['price']
                    st.markdown(f"**Tomorrow's Target:** `${next_day_val:.2f}`")
                    
                    fig = px.line(df_symbol, x="date", y="price", markers=True)
                    fig.update_layout(height=250, margin=dict(l=0, r=0, t=10, b=0), template="plotly_dark", yaxis_title="Price", xaxis_title="")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("N/A (Run Pipeline)")
    else:
        for i, symbol in enumerate(symbols):
            with cols[i]:
                st.metric(label=f"{symbol} Target", value="$0.00", delta="Connect Supabase")

if __name__ == "__main__":
    main()
