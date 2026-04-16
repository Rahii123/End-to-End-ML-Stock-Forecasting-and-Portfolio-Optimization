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

@st.cache_data(ttl=86400, show_spinner=False)
def get_historical_cache(symbols):
    """Caches Alpha Vantage data for 24 hours to prevent 60-second freezes."""
    extractor = DataExtractor(symbols)
    return extractor.fetch_historical_data(period="1y")

def main():
    st.title("📈 AI Stock Forecasting & Portfolio Optimization")
    st.markdown("---")

    symbols = ["AAPL", "GOOG", "PLTR", "JPM"]
    
    # Sidebar
    st.sidebar.header("Controls")
    if st.sidebar.button("🚀 Run Prediction Pipeline"):
        with st.spinner("Executing ML Pipeline..."):
            run_pipeline(symbols)
            st.sidebar.success("Pipeline executed successfully!")

    # Database connection
    db = SupabaseDatabase()
    
    # Layout
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Historical Performance & Trends")
        historical_df = get_historical_cache(symbols)
        
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
            weights_data = db.client.table("portfolio_weights").select("*").execute().data
            if weights_data:
                weights_df = pd.DataFrame(weights_data)
                fig_pie = px.pie(weights_df, values='weight', names='symbol', hole=.4,
                                 color_discrete_sequence=px.colors.sequential.RdBu)
                fig_pie.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_pie, use_container_width=True)
            else:
                st.info("No optimization data yet. Run the pipeline.")
        else:
            st.warning("Supabase not connected. Showing mock weights.")
            mock_weights = {"AAPL": 0.4, "GOOG": 0.3, "PLTR": 0.2, "JPM": 0.1}
            fig_pie = px.pie(names=list(mock_weights.keys()), values=list(mock_weights.values()), hole=.4)
            fig_pie.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown("---")
    st.subheader("Next-Day Price Predictions")
    
    pred_col1, pred_col2, pred_col3, pred_col4 = st.columns(4)
    cols = [pred_col1, pred_col2, pred_col3, pred_col4]

    if db.client:
        preds = db.get_latest_predictions()
        # Find latest prediction for each symbol
        latest_preds = {}
        for p in preds:
            if p['symbol'] not in latest_preds:
                latest_preds[p['symbol']] = p
        
        for i, symbol in enumerate(symbols):
            with cols[i]:
                if symbol in latest_preds:
                    val = latest_preds[symbol]['predicted_price']
                    st.metric(label=f"{symbol} Target", value=f"${val:.2f}")
                else:
                    st.metric(label=f"{symbol} Target", value="N/A")
    else:
        for i, symbol in enumerate(symbols):
            with cols[i]:
                st.metric(label=f"{symbol} Target", value="$0.00", delta="Connect Supabase")

if __name__ == "__main__":
    main()
