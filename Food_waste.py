import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Food Waste Analysis Dashboard",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {font-size: 2.5rem; color: #2e8b57; font-weight: 700;}
    .section-header {font-size: 1.8rem; color: #2e8b57; border-bottom: 2px solid #2e8b57; padding-bottom: 0.3rem;}
    .metric-label {font-weight: 600; color: #2e8b57;}
    .positive-metric {color: #228B22;}
    .negative-metric {color: #DC143C;}
    .info-text {background-color: #f0f8f0; padding: 15px; border-radius: 5px; border-left: 4px solid #2e8b57;}
</style>
""", unsafe_allow_html=True)

# Check if statsmodels is available for trendlines
try:
    import statsmodels.api as sm
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    st.sidebar.warning("Statsmodels not installed. Trendlines will be disabled.")

# App title
st.markdown('<p class="main-header">🍎 Food Waste Analysis Dashboard</p>', unsafe_allow_html=True)
st.markdown("Analyze food waste patterns from production and inventory data")

# Generate sample data if not uploaded
def generate_sample_data():
    products = ["Tomatoes", "Potatoes", "Rice", "Chicken", "Milk", "Apples"]
    units = ["tons", "tons", "tons", "tons", "liters", "tons"]
    years = [2020, 2021, 2022]
    months = list(range(1, 13))

    data = []
    for product, unit in zip(products, units):
        for year in years:
            for month in months:
                begin_inventory = np.random.randint(100, 500)
                production = np.random.randint(1000, 5000)
                domestic = np.random.randint(800, 4500)
                export = np.random.randint(100, 800)
                total = domestic + export
                shipment_value = np.random.randint(50000, 250000)
                end_inventory = np.random.randint(100, 500)
                capacity = np.random.randint(5000, 10000)

                data.append({
                    "Product": product,
                    "Unit": unit,
                    "Year": year,
                    "Month": month,
                    "Begin month inventory": begin_inventory,
                    "Production": production,
                    "Domestic": domestic,
                    "Export": export,
                    "Total": total,
                    "Shipment value (thousand baht)": shipment_value,
                    "Month-end inventory": end_inventory,
                    "Capacity": capacity
                })

    return pd.DataFrame(data)

# Load data
with st.sidebar:
    st.header("Data Configuration")

    st.subheader("Upload Data")
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success("Data uploaded successfully!")
    else:
        st.info("Using sample data for demonstration")
        df = generate_sample_data()

    st.subheader("Analysis Parameters")
    selected_products = st.multiselect(
        "Select Products to Analyze",
        options=df['Product'].unique(),
        default=df['Product'].unique()[:3]
    )

    year_range = st.slider(
        "Select Year Range",
        min_value=int(df['Year'].min()),
        max_value=int(df['Year'].max()),
        value=(int(df['Year'].min()), int(df['Year'].max()))
    )

    st.subheader("Waste Calculation")
    st.info("Waste = (Begin Inventory + Production) - (Domestic + Export + End Inventory)")

# Filter data based on selections
df_filtered = df[
    (df['Product'].isin(selected_products)) &
    (df['Year'] >= year_range[0]) &
    (df['Year'] <= year_range[1])
]

# Calculate waste metrics
df_filtered['waste'] = (df_filtered['Begin month inventory'] + df_filtered['Production']) - (df_filtered['Domestic'] + df_filtered['Export'] + df_filtered['Month-end inventory'])
df_filtered['waste_rate'] = df_filtered['waste'] / df_filtered['Production']
df_filtered['avg_inventory'] = (df_filtered['Begin month inventory'] + df_filtered['Month-end inventory']) / 2
df_filtered['inventory_turnover'] = df_filtered['Domestic'] / df_filtered['avg_inventory']
df_filtered['capacity_utilization'] = df_filtered['Production'] / df_filtered['Capacity']
df_filtered['value_per_unit'] = df_filtered['Shipment value (thousand baht)'] / df_filtered['Total']
df_filtered['waste_value'] = df_filtered['waste'] * df_filtered['value_per_unit']

# Create tabs for different analyses
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Overview",
    "Waste Analysis",
    "Inventory Analysis",
    "Production vs Demand",
    "Economic Impact"
])

with tab1:
    st.markdown('<p class="section-header">📊 Overview Metrics</p>', unsafe_allow_html=True)

    # Calculate overall metrics
    total_waste = df_filtered['waste'].sum()
    total_production = df_filtered['Production'].sum()
    overall_waste_rate = total_waste / total_production
    total_waste_value = df_filtered['waste_value'].sum()
    avg_turnover = df_filtered['inventory_turnover'].mean()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Waste", f"{total_waste:,.0f} units")
    with col2:
        st.metric("Overall Waste Rate", f"{overall_waste_rate:.2%}")
    with col3:
        st.metric("Value of Waste", f"฿{total_waste_value:,.0f}")
    with col4:
        st.metric("Avg Inventory Turnover", f"{avg_turnover:.2f}")

    # Time series of waste
    st.markdown("#### Waste Trends Over Time")
    waste_by_time = df_filtered.groupby(['Year', 'Month']).agg({'waste': 'sum'}).reset_index()
    waste_by_time['date'] = pd.to_datetime(waste_by_time['Year'].astype(str) + '-' + waste_by_time['Month'].astype(str))

    fig = px.line(waste_by_time, x='date', y='waste',
                  title="Total Waste Over Time")
    st.plotly_chart(fig, use_container_width=True)

    # Waste by product
    st.markdown("#### Waste by Product")
    waste_by_product = df_filtered.groupby('Product').agg({
        'waste': 'sum',
        'Production': 'sum',
        'waste_value': 'sum'
    }).reset_index()
    waste_by_product['waste_rate'] = waste_by_product['waste'] / waste_by_product['Production']

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(waste_by_product, x='Product', y='waste',
                     title="Total Waste by Product")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.bar(waste_by_product, x='Product', y='waste_rate',
                     title="Waste Rate by Product")
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.markdown('<p class="section-header">📈 Waste Analysis</p>', unsafe_allow_html=True)

    # Seasonal waste patterns
    st.markdown("#### Seasonal Waste Patterns")
    waste_by_month = df_filtered.groupby(['Month', 'Product']).agg({'waste': 'mean'}).reset_index()

    fig = px.line(waste_by_month, x='Month', y='waste', color='Product',
                  title="Average Waste by Month")
    st.plotly_chart(fig, use_container_width=True)

    # Waste distribution
    st.markdown("#### Waste Distribution by Product")
    fig = px.box(df_filtered, x='Product', y='waste',
                 title="Distribution of Waste Amounts by Product")
    st.plotly_chart(fig, use_container_width=True)

    # Yearly comparison
    st.markdown("#### Yearly Waste Comparison")
    waste_by_year = df_filtered.groupby(['Year', 'Product']).agg({'waste': 'sum'}).reset_index()

    fig = px.bar(waste_by_year, x='Year', y='waste', color='Product',
                 barmode='group', title="Total Waste by Year and Product")
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.markdown('<p class="section-header">📦 Inventory Analysis</p>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("##### Inventory Turnover by Product")
        turnover_by_product = df_filtered.groupby('Product').agg({
            'inventory_turnover': 'mean'
        }).reset_index().sort_values('inventory_turnover', ascending=False)

        fig = px.bar(turnover_by_product, x='Product', y='inventory_turnover',
                     title="Average Inventory Turnover Ratio")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("##### Days of Supply")
        df_filtered['days_of_supply'] = (df_filtered['Month-end inventory'] / df_filtered['Domestic']) * 30
        days_supply_by_product = df_filtered.groupby('Product').agg({
            'days_of_supply': 'mean'
        }).reset_index().sort_values('days_of_supply', ascending=False)

        fig = px.bar(days_supply_by_product, x='Product', y='days_of_supply',
                     title="Average Days of Supply")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("##### Inventory vs Waste Relationship")
        # Only add trendline if statsmodels is available
        if HAS_STATSMODELS:
            fig = px.scatter(df_filtered, x='avg_inventory', y='waste', color='Product',
                             trendline="ols", title="Relationship Between Inventory Levels and Waste")
        else:
            fig = px.scatter(df_filtered, x='avg_inventory', y='waste', color='Product',
                             title="Relationship Between Inventory Levels and Waste (No Trendline - Install statsmodels)")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("##### Monthly Inventory Patterns")
        inventory_by_month = df_filtered.groupby(['Month', 'Product']).agg({
            'Begin month inventory': 'mean',
            'Month-end inventory': 'mean'
        }).reset_index()

        fig = px.line(inventory_by_month, x='Month', y='Begin month inventory', color='Product',
                      title="Average Beginning Inventory by Month")
        st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.markdown('<p class="section-header">⚖️ Production vs Demand Analysis</p>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("##### Production vs Domestic Sales")
        prod_vs_domestic = df_filtered.groupby('Product').agg({
            'Production': 'sum',
            'Domestic': 'sum'
        }).reset_index()

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=prod_vs_domestic['Product'],
            y=prod_vs_domestic['Production'],
            name='Production'
        ))
        fig.add_trace(go.Bar(
            x=prod_vs_domestic['Product'],
            y=prod_vs_domestic['Domestic'],
            name='Domestic Sales'
        ))
        fig.update_layout(barmode='group', title="Total Production vs Domestic Sales")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("##### Capacity Utilization")
        utilization_by_product = df_filtered.groupby('Product').agg({
            'capacity_utilization': 'mean'
        }).reset_index()

        fig = px.bar(utilization_by_product, x='Product', y='capacity_utilization',
                     title="Average Capacity Utilization by Product")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("##### Production-Demand Mismatch")
        df_filtered['production_demand_gap'] = df_filtered['Production'] - df_filtered['Domestic']

        gap_by_product = df_filtered.groupby('Product').agg({
            'production_demand_gap': 'mean'
        }).reset_index()

        fig = px.bar(gap_by_product, x='Product', y='production_demand_gap',
                     title="Average Production-Demand Gap")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("##### Production vs Waste Correlation")
        # Only add trendline if statsmodels is available
        if HAS_STATSMODELS:
            fig = px.scatter(df_filtered, x='Production', y='waste', color='Product',
                             trendline="ols", title="Correlation Between Production Volume and Waste")
        else:
            fig = px.scatter(df_filtered, x='Production', y='waste', color='Product',
                             title="Correlation Between Production Volume and Waste (No Trendline - Install statsmodels)")
        st.plotly_chart(fig, use_container_width=True)

with tab5:
    st.markdown('<p class="section-header">💰 Economic Impact Analysis</p>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("##### Value of Waste by Product")
        waste_value_by_product = df_filtered.groupby('Product').agg({
            'waste_value': 'sum'
        }).reset_index().sort_values('waste_value', ascending=False)

        fig = px.bar(waste_value_by_product, x='Product', y='waste_value',
                     title="Total Value of Waste by Product (Thousand Baht)")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("##### Monthly Waste Value Trends")
        waste_value_by_month = df_filtered.groupby(['Year', 'Month']).agg({
            'waste_value': 'sum'
        }).reset_index()
        waste_value_by_month['date'] = pd.to_datetime(waste_value_by_month['Year'].astype(str) + '-' + waste_value_by_month['Month'].astype(str))

        fig = px.line(waste_value_by_month, x='date', y='waste_value',
                      title="Trends in Waste Value Over Time")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("##### Waste as Percentage of Shipment Value")
        total_values = df_filtered.groupby('Product').agg({
            'Shipment value (thousand baht)': 'sum',
            'waste_value': 'sum'
        }).reset_index()
        total_values['waste_pct_of_value'] = total_values['waste_value'] / total_values['Shipment value (thousand baht)'] * 100

        fig = px.bar(total_values, x='Product', y='waste_pct_of_value',
                     title="Waste Value as Percentage of Total Shipment Value")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("##### Cost of Waste by Season")
        waste_value_by_season = df_filtered.groupby('Month').agg({
            'waste_value': 'mean'
        }).reset_index()

        fig = px.line(waste_value_by_season, x='Month', y='waste_value',
                      title="Average Monthly Waste Value (Seasonal Pattern)")
        st.plotly_chart(fig, use_container_width=True)

# Add summary and recommendations
st.markdown("---")
st.markdown('<p class="section-header">📋 Key Insights and Recommendations</p>', unsafe_allow_html=True)

# Generate insights based on data
if not df_filtered.empty:
    # Find product with highest waste rate
    waste_by_product = df_filtered.groupby('Product').agg({
        'waste': 'sum',
        'Production': 'sum'
    }).reset_index()
    waste_by_product['waste_rate'] = waste_by_product['waste'] / waste_by_product['Production']

    highest_waste_product = waste_by_product.loc[waste_by_product['waste_rate'].idxmax()]
    lowest_waste_product = waste_by_product.loc[waste_by_product['waste_rate'].idxmin()]

    # Find month with highest waste
    waste_by_month = df_filtered.groupby('Month').agg({'waste': 'mean'}).reset_index()
    highest_waste_month = waste_by_month.loc[waste_by_month['waste'].idxmax()]

    # Inventory turnover analysis
    turnover_by_product = df_filtered.groupby('Product').agg({'inventory_turnover': 'mean'}).reset_index()
    lowest_turnover = turnover_by_product.loc[turnover_by_product['inventory_turnover'].idxmin()]

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Key Findings")
        st.markdown(f"""
        - **{highest_waste_product['Product']}** has the highest waste rate at **{highest_waste_product['waste_rate']:.2%}**
        - **{lowest_waste_product['Product']}** has the lowest waste rate at **{lowest_waste_product['waste_rate']:.2%}**
        - Month **{int(highest_waste_month['Month'])}** typically has the highest waste levels
        - **{lowest_turnover['Product']}** has the slowest inventory turnover
        """)

    with col2:
        st.markdown("#### Recommendations")
        st.markdown("""
        - Implement better inventory management for low turnover products
        - Adjust production schedules based on seasonal demand patterns
        - Improve storage conditions for high-waste products
        - Develop strategies to redirect potential waste to alternative markets
        - Enhance demand forecasting to reduce production-demand mismatch
        """)

# Data download option
st.sidebar.markdown("---")
st.sidebar.markdown("### Download Results")
st.sidebar.download_button(
    label="Download Analysis Results",
    data=df_filtered.to_csv(index=False),
    file_name="food_waste_analysis.csv",
    mime="text/csv"
)

# Installation instructions if statsmodels is missing
if not HAS_STATSMODELS:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Install Additional Package")
    st.sidebar.code("pip install statsmodels")
    st.sidebar.info("Install statsmodels to enable trendline functionality in charts")







