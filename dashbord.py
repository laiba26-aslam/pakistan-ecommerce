# =========================================================
# 🇵🇰 Pakistan E-Commerce Analytics Dashboard
# Built with Streamlit + Plotly
# Run: streamlit run dashboard.py
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────
st.set_page_config(
    page_title="PK E-Commerce Analytics",
    page_icon="🇵🇰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}
h1, h2, h3 { font-family: 'Syne', sans-serif; }

/* Dark sidebar */
section[data-testid="stSidebar"] {
    background: #0f1117;
    border-right: 1px solid #1e2130;
}
section[data-testid="stSidebar"] * { color: #e0e0e0 !important; }

/* Metric cards */
div[data-testid="metric-container"] {
    background: linear-gradient(135deg, #1a1d2e 0%, #12151f 100%);
    border: 1px solid #2a2d3e;
    border-radius: 12px;
    padding: 16px 20px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
}
div[data-testid="metric-container"] label {
    color: #7b82a0 !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
    color: #f0f4ff !important;
    font-family: 'Syne', sans-serif;
    font-size: 1.6rem !important;
    font-weight: 700;
}

/* Main background */
.main { background: #0a0c14; }
.block-container { padding-top: 2rem; }

/* Section headers */
.section-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    color: #7dd3fc;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin: 1.5rem 0 0.5rem 0;
    border-left: 3px solid #38bdf8;
    padding-left: 10px;
}

/* Tab styling */
button[data-baseweb="tab"] {
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
}
</style>
""", unsafe_allow_html=True)

# ── Load Data ─────────────────────────────────────────────
@st.cache_data
def load_data():
    try:
        df           = pd.read_csv("cleaned_ecommerce.csv", parse_dates=["Order_Date"])
        category_kpi = pd.read_csv("category_kpi.csv")
        monthly      = pd.read_csv("monthly_sales.csv")
        forecast     = pd.read_csv("forecast_summary.csv")
        return df, category_kpi, monthly, forecast
    except FileNotFoundError as e:
        st.error(f"❌ File not found: {e}\n\nMake sure you run **analysis.py** first to generate the CSV files.")
        st.stop()

df, category_kpi, monthly_sales, forecast_df = load_data()

# ── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🇵🇰 PK Analytics")
    st.markdown("---")

    categories = ["All"] + sorted(df["Category"].dropna().unique().tolist())
    selected_cat = st.selectbox("📦 Filter by Category", categories)

    if "Year" in df.columns:
        years = ["All"] + sorted(df["Year"].dropna().astype(int).unique().tolist())
        selected_year = st.selectbox("📅 Filter by Year", years)
    else:
        selected_year = "All"

    st.markdown("---")
    st.markdown("### 📥 Download Data")
    st.download_button(
        label="⬇ Cleaned Dataset",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="cleaned_ecommerce.csv",
        mime="text/csv"
    )
    st.download_button(
        label="⬇ Category KPIs",
        data=category_kpi.to_csv(index=False).encode("utf-8"),
        file_name="category_kpi.csv",
        mime="text/csv"
    )
    st.download_button(
        label="⬇ Forecast Summary",
        data=forecast_df.to_csv(index=False).encode("utf-8"),
        file_name="forecast_summary.csv",
        mime="text/csv"
    )

    st.markdown("---")
    st.caption("Built with Streamlit + Plotly\nPakistan E-Commerce Dataset")

# ── Apply Filters ─────────────────────────────────────────
filtered = df.copy()
if selected_cat != "All":
    filtered = filtered[filtered["Category"] == selected_cat]
if selected_year != "All":
    filtered = filtered[filtered["Year"] == int(selected_year)]

# ── Header ────────────────────────────────────────────────
st.markdown("""
<h1 style='font-family:Syne,sans-serif; font-size:2.2rem; color:#f0f4ff; margin-bottom:0;'>
🇵🇰 Pakistan E-Commerce Analytics
</h1>
<p style='color:#7b82a0; margin-top:4px; font-size:0.95rem;'>
Sales · Profit · Forecasting · Outlier Intelligence
</p>
<hr style='border:1px solid #1e2130; margin: 12px 0 20px 0;'>
""", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview",
    "📈 Sales Trends",
    "🗂️ Categories",
    "🔮 Forecast",
    "🚨 Outliers"
])

# ═══════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW KPIs
# ═══════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-title">Key Performance Indicators</div>', unsafe_allow_html=True)

    total_sales   = filtered["Sales"].sum()
    total_profit  = filtered["Profit"].sum()
    total_orders  = len(filtered)
    avg_rating    = filtered["Rating"].mean() if "Rating" in filtered.columns else 0
    profit_margin = (total_profit / total_sales * 100) if total_sales > 0 else 0

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("💰 Total Sales",    f"PKR {total_sales:,.0f}")
    c2.metric("📈 Total Profit",   f"PKR {total_profit:,.0f}")
    c3.metric("🛒 Total Orders",   f"{total_orders:,}")
    c4.metric("⭐ Avg Rating",     f"{avg_rating:.2f}")
    c5.metric("📊 Profit Margin",  f"{profit_margin:.1f}%")

    st.markdown('<div class="section-title">Sales Distribution</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        fig_hist = px.histogram(
            filtered, x="Sales", nbins=40,
            title="Sales Distribution",
            color_discrete_sequence=["#38bdf8"],
            template="plotly_dark"
        )
        fig_hist.update_layout(
            paper_bgcolor="#12151f", plot_bgcolor="#12151f",
            font_color="#e0e0e0", title_font_family="Syne"
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with col2:
        if "outlier_type" in filtered.columns:
            ot_counts = filtered["outlier_type"].value_counts().reset_index()
            ot_counts.columns = ["Type", "Count"]
            fig_pie = px.pie(
                ot_counts, names="Type", values="Count",
                title="Order Types",
                color_discrete_sequence=px.colors.sequential.Blues_r,
                template="plotly_dark"
            )
            fig_pie.update_layout(
                paper_bgcolor="#12151f",
                font_color="#e0e0e0", title_font_family="Syne"
            )
            st.plotly_chart(fig_pie, use_container_width=True)

# ═══════════════════════════════════════════════════════════
# TAB 2 — SALES TRENDS
# ═══════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-title">Monthly Sales Trend</div>', unsafe_allow_html=True)

    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(
        x=monthly_sales.index,
        y=monthly_sales["Sales"],
        mode="lines",
        name="Actual Sales",
        line=dict(color="#38bdf8", width=2.5)
    ))
    if "MA_3" in monthly_sales.columns:
        fig_trend.add_trace(go.Scatter(
            x=monthly_sales.index,
            y=monthly_sales["MA_3"],
            mode="lines",
            name="3-Month MA",
            line=dict(color="#f59e0b", width=2, dash="dash")
        ))
    fig_trend.update_layout(
        title="Monthly Sales with 3-Month Moving Average",
        paper_bgcolor="#12151f", plot_bgcolor="#12151f",
        font_color="#e0e0e0", title_font_family="Syne",
        legend=dict(bgcolor="#1a1d2e", bordercolor="#2a2d3e"),
        xaxis_title="Period", yaxis_title="Sales (PKR)"
    )
    st.plotly_chart(fig_trend, use_container_width=True)

    # Monthly heatmap by year/month
    if "Year" in filtered.columns and "Month" in filtered.columns:
        st.markdown('<div class="section-title">Sales Heatmap by Month & Year</div>', unsafe_allow_html=True)
        pivot = filtered.groupby(["Year", "Month"])["Sales"].sum().unstack(fill_value=0)
        fig_heat = px.imshow(
            pivot,
            labels=dict(x="Month", y="Year", color="Sales"),
            color_continuous_scale="Blues",
            title="Sales Heatmap",
            template="plotly_dark"
        )
        fig_heat.update_layout(
            paper_bgcolor="#12151f", plot_bgcolor="#12151f",
            font_color="#e0e0e0", title_font_family="Syne"
        )
        st.plotly_chart(fig_heat, use_container_width=True)

# ═══════════════════════════════════════════════════════════
# TAB 3 — CATEGORIES
# ═══════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-title">Category Performance</div>', unsafe_allow_html=True)

    cat_sorted = category_kpi.sort_values("Total_Sales", ascending=False)

    col1, col2 = st.columns(2)
    with col1:
        fig_sales = px.bar(
            cat_sorted, x="Category", y="Total_Sales",
            title="Total Sales by Category",
            color="Total_Sales",
            color_continuous_scale="Blues",
            template="plotly_dark"
        )
        fig_sales.update_layout(
            paper_bgcolor="#12151f", plot_bgcolor="#12151f",
            font_color="#e0e0e0", title_font_family="Syne",
            xaxis_tickangle=-35, showlegend=False
        )
        st.plotly_chart(fig_sales, use_container_width=True)

    with col2:
        cat_margin = cat_sorted.copy()
        cat_margin["Margin_%"] = cat_margin["Profit_Margin"].fillna(0) * 100
        fig_margin = px.bar(
            cat_margin, x="Category", y="Margin_%",
            title="Profit Margin % by Category",
            color="Margin_%",
            color_continuous_scale="Greens",
            template="plotly_dark"
        )
        fig_margin.update_layout(
            paper_bgcolor="#12151f", plot_bgcolor="#12151f",
            font_color="#e0e0e0", title_font_family="Syne",
            xaxis_tickangle=-35, showlegend=False
        )
        st.plotly_chart(fig_margin, use_container_width=True)

    st.markdown('<div class="section-title">Category KPI Table</div>', unsafe_allow_html=True)
    display_kpi = cat_sorted.copy()
    display_kpi["Profit_Margin"] = (display_kpi["Profit_Margin"].fillna(0) * 100).round(2)
    display_kpi["Total_Sales"]   = display_kpi["Total_Sales"].round(0)
    display_kpi["Total_Profit"]  = display_kpi["Total_Profit"].round(0)
    display_kpi["Avg_Rating"]    = display_kpi["Avg_Rating"].round(2)
    st.dataframe(
        display_kpi.rename(columns={"Profit_Margin": "Margin %"}),
        use_container_width=True,
        hide_index=True
    )

    # Scatter: Sales vs Profit by Category
    st.markdown('<div class="section-title">Sales vs Profit Bubble Chart</div>', unsafe_allow_html=True)
    fig_bubble = px.scatter(
        cat_sorted,
        x="Total_Sales", y="Total_Profit",
        size="Order_Count", color="Category",
        hover_name="Category",
        title="Sales vs Profit (bubble = order count)",
        template="plotly_dark"
    )
    fig_bubble.update_layout(
        paper_bgcolor="#12151f", plot_bgcolor="#12151f",
        font_color="#e0e0e0", title_font_family="Syne"
    )
    st.plotly_chart(fig_bubble, use_container_width=True)

# ═══════════════════════════════════════════════════════════
# TAB 4 — FORECAST
# ═══════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-title">3-Month Sales Forecast</div>', unsafe_allow_html=True)

    fig_fc = go.Figure()

    # Historical
    fig_fc.add_trace(go.Scatter(
        x=list(range(len(monthly_sales))),
        y=monthly_sales["Sales"],
        mode="lines",
        name="Historical Sales",
        line=dict(color="#38bdf8", width=2.5)
    ))

    future_x = [len(monthly_sales), len(monthly_sales)+1, len(monthly_sales)+2]

    # ARIMA
    if "ARIMA_Forecast" in forecast_df.columns:
        fig_fc.add_trace(go.Scatter(
            x=future_x, y=forecast_df["ARIMA_Forecast"],
            mode="lines+markers",
            name="ARIMA Forecast",
            line=dict(color="#f43f5e", width=2.5, dash="dot"),
            marker=dict(size=10, symbol="circle")
        ))

    # Linear Regression
    if "LR_Forecast" in forecast_df.columns:
        fig_fc.add_trace(go.Scatter(
            x=future_x, y=forecast_df["LR_Forecast"],
            mode="lines+markers",
            name="Linear Regression",
            line=dict(color="#a78bfa", width=2.5, dash="dash"),
            marker=dict(size=10, symbol="diamond")
        ))

    fig_fc.add_vline(
        x=len(monthly_sales)-1,
        line_dash="dot", line_color="#4b5563",
        annotation_text="Forecast Start",
        annotation_font_color="#9ca3af"
    )

    fig_fc.update_layout(
        title="Historical Sales + ARIMA & Linear Regression Forecast",
        paper_bgcolor="#12151f", plot_bgcolor="#12151f",
        font_color="#e0e0e0", title_font_family="Syne",
        legend=dict(bgcolor="#1a1d2e", bordercolor="#2a2d3e"),
        xaxis_title="Period Index", yaxis_title="Sales (PKR)"
    )
    st.plotly_chart(fig_fc, use_container_width=True)

    # Forecast numbers
    st.markdown('<div class="section-title">Forecast Values</div>', unsafe_allow_html=True)
    fc_display = forecast_df.copy()
    fc_display.columns = ["Month Offset", "Linear Regression", "ARIMA"]
    fc_display["Month Offset"] = fc_display["Month Offset"].apply(lambda x: f"+{x} month")
    st.dataframe(fc_display, use_container_width=True, hide_index=True)

    st.info("💡 **ARIMA** captures time-series patterns. **Linear Regression** shows the overall trend direction. Compare both to understand forecast confidence.")

# ═══════════════════════════════════════════════════════════
# TAB 5 — OUTLIERS
# ═══════════════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="section-title">Outlier Explorer</div>', unsafe_allow_html=True)

    if "outlier_type" in df.columns:
        outlier_filter = st.multiselect(
            "Filter by Outlier Type",
            options=df["outlier_type"].unique().tolist(),
            default=[t for t in df["outlier_type"].unique() if t != "normal"]
        )
        outlier_df = df[df["outlier_type"].isin(outlier_filter)]

        c1, c2, c3 = st.columns(3)
        c1.metric("🚨 Extreme Anomalies",    (df["outlier_type"] == "extreme_anomaly").sum())
        c2.metric("💼 Business Opportunities", (df["outlier_type"] == "business_opportunity").sum())
        c3.metric("⚠️ Data Issues",           (df["outlier_type"] == "data_issue").sum())

        # Scatter: Sales vs Profit colored by outlier type
        fig_out = px.scatter(
            df, x="Sales", y="Profit",
            color="outlier_type",
            title="Sales vs Profit — Colored by Outlier Type",
            color_discrete_map={
                "normal":               "#334155",
                "extreme_anomaly":      "#f43f5e",
                "business_opportunity": "#22d3ee",
                "data_issue":           "#f59e0b"
            },
            template="plotly_dark",
            opacity=0.7
        )
        fig_out.update_layout(
            paper_bgcolor="#12151f", plot_bgcolor="#12151f",
            font_color="#e0e0e0", title_font_family="Syne"
        )
        st.plotly_chart(fig_out, use_container_width=True)

        st.markdown('<div class="section-title">Flagged Records</div>', unsafe_allow_html=True)
        cols_to_show = [c for c in ["Order_Date","Category","Sales","Profit","Quantity","Price","outlier_type","z_score"]
                        if c in outlier_df.columns]
        st.dataframe(
            outlier_df[cols_to_show].sort_values("Sales", ascending=False).head(200),
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "⬇ Download Outlier Report",
            data=outlier_df.to_csv(index=False).encode("utf-8"),
            file_name="outlier_report.csv",
            mime="text/csv"
        )
    else:
        st.warning("No outlier_type column found. Run the full analysis pipeline first.")