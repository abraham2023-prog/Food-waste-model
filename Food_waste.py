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

# App title
st.markdown('<p class="main-header">🍎 Food Waste Analysis Dashboard</p>', unsafe_allow_html=True)
st.markdown("Analyze food waste patterns from production and inventory data")

# Generate sample data if not uploaded
def generate_sample_data():
    # Raw product list from your dataset
    raw_products = [
        'Ready made pig feed', 'Ready made chicken feed', 'Ready made fish feed',
        'Ready made shrimp feed', 'Premix', 'Dried fruits and vegetables',
        'Ice cream', 'Yogurt', 'Soy milk', 'Tapioca flour', 'Cake',
        'Other baked goods (pizza donuts sandwich bread)', 'Wafer biscuit',
        'Cookie', 'Toasted bread/Cracker/Biscuit',
        'Other crispy snacks (Corn chips prawn crackers etc)', 'Molasses',
        'Instant noodles', 'Table condiments',
        'Soy sauce fermented soybean paste light soy sauce ',
        'Monosodium glutamate', 'Ready to cook meals (others)', 'Pet feed',
        'Frozen and chilled pork', 'Frozen and chilled chicken meat',
        'ham', 'bacon', 'sausage', 'seasoned chicken meat', 'frozen fish',
        'minced fish meat', 'frozen shrimp', 'frozen squid', 'canned tuna',
        'canned sardines', 'frozen fruits and vegetables', 'canned pineapple',
        'other canned fruits', 'canned sweet corn', 'canned pickles',
        'dried fruits & vegetables', 'coconut milk', 'ice cream',
        'tapioca starch', 'cake', 'other baked goods', 'cookie',
        'biscuits/crackers', 'other crispy baked snacks', 'raw sugar',
        'white sugar', 'pure white sugar', 'molasses', 'sweep/suck',
        'instant noodles', 'Frozen prepared food',
        'small condiments or seasoning dispensers',
        'Soy sauce fermented soybean paste dark soy sauce',
        'Ready made pet feed', 'Ready-made pig feed', 'Ready made duck feed',
        'ready made feed for other livestock'
    ]

    # Normalize product names (mapping variations to consistent labels)
    product_mapping = {
        "ice cream": "Ice Cream",
        "Ice cream": "Ice Cream",
        "cake": "Cake",
        "Cake": "Cake",
        "cookie": "Cookie",
        "Cookie": "Cookie",
        "dried fruits & vegetables": "Dried Fruits and Vegetables",
        "Dried fruits and vegetables": "Dried Fruits and Vegetables",
        "other baked goods": "Other Baked Goods",
        "Other baked goods (pizza donuts sandwich bread)": "Other Baked Goods",
        "biscuits/crackers": "Biscuits/Crackers",
        "Toasted bread/Cracker/Biscuit": "Biscuits/Crackers",
        "other crispy baked snacks": "Other Crispy Snacks",
        "Other crispy snacks (Corn chips prawn crackers etc)": "Other Crispy Snacks",
        "molasses": "Molasses",
        "Molasses": "Molasses",
        "instant noodles": "Instant Noodles",
        "Instant noodles": "Instant Noodles",
    }

    # Apply normalization
    products = [product_mapping.get(p, p) for p in raw_products]
    products = sorted(set(products))  # remove duplicates & sort alphabetically

    # Assign units
    units = []
    for p in products:
        if any(x in p.lower() for x in ["milk", "yogurt"]):
            units.append("liter")
        elif "sauce" in p.lower() or "soy" in p.lower():
            units.append("Thousand liter")
        else:
            units.append("ton")

    years = [2020, 2021, 2022, 2023]
    months = list(range(1, 13))

    data = []
    for product, unit in zip(products, units):
        for year in years:
            for month in months:
                begin_inventory = np.random.randint(50, 500)
                production = np.random.randint(500, 5000)
                domestic = np.random.randint(400, 4500)
                export = np.random.randint(50, 800)
                total = domestic + export
                shipment_value = np.random.randint(20000, 250000)
                end_inventory = np.random.randint(50, 500)
                capacity = np.random.randint(3000, 10000)

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

# Function to normalize column names
def normalize_column_names(df):
    """Normalize column names to handle different naming conventions"""
    # Create a mapping of possible column name variations
    column_mapping = {
        'tsic': 'TSIC',
        'code': 'Code',
        'product': 'Product',
        'unit': 'Unit',
        'year': 'Year',
        'month': 'Month',
        'begin month inventory': 'Begin month inventory',
        'begin_month_inventory': 'Begin month inventory',
        'beginmonthinventory': 'Begin month inventory',
        'production': 'Production',
        'domestic': 'Domestic',
        'export': 'Export',
        'total': 'Total',
        'shipment value (thousand baht)': 'Shipment value (thousand baht)',
        'shipment_value_thousand_baht': 'Shipment value (thousand baht)',
        'month-end inventory': 'Month-end inventory',
        'month_end_inventory': 'Month-end inventory',
        'monthendinventory': 'Month-end inventory',
        'capacity': 'Capacity'
    }
    
    # Normalize the column names
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('-', '_')
    
    # Map to standardized names
    new_columns = []
    for col in df.columns:
        normalized = col.lower().replace(' ', '_').replace('-', '_')
        new_columns.append(column_mapping.get(normalized, col))
    
    df.columns = new_columns
    return df

# Load and clean data
with st.sidebar:
    st.header("Data Configuration")
    st.subheader("Upload Data")
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    # Show original column names for debugging
    st.sidebar.info("Original columns in your dataset:")
    for col in df.columns:
        st.sidebar.write(f"- '{col}'")
    
    # Normalize column names
    df = normalize_column_names(df)
    
    st.sidebar.info("Normalized columns:")
    for col in df.columns:
        st.sidebar.write(f"- '{col}'")
    
    # List of expected numeric columns
    numeric_cols = [
        'Begin month inventory', 'Production', 'Domestic', 'Export',
        'Total', 'Shipment value (thousand baht)', 'Month-end inventory', 'Capacity'
    ]
    
    # Remove commas and convert to float for available numeric columns
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(',', '').astype(float)
        else:
            st.sidebar.warning(f"Column '{col}' not found in dataset")
    
    # Check for missing required columns
    required_cols = ['Begin month inventory', 'Production', 'Domestic', 'Export', 'Month-end inventory']
    missing_cols = [c for c in required_cols if c not in df.columns]
    
    if missing_cols:
        st.error(f"Missing required columns in dataset: {missing_cols}")
        st.info("Please check if your dataset contains these columns with similar names")
        # Stop execution if required columns are missing
        st.stop()
    else:
        st.success("Data uploaded, cleaned, and converted successfully!")

else:
    st.info("Using sample data for demonstration")
    df = generate_sample_data()

# Analysis parameters in sidebar
with st.sidebar:
    st.subheader("Analysis Parameters")
    selected_products = st.multiselect(
        "Select Products to Analyze",
        options=df['Product'].unique(),
        default=df['Product'].unique()[:3] if len(df['Product'].unique()) > 3 else df['Product'].unique()
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
].copy()

# Calculate waste metrics
df_filtered['waste'] = (df_filtered['Begin month inventory'] + df_filtered['Production']) - (df_filtered['Domestic'] + df_filtered['Export'] + df_filtered['Month-end inventory'])
df_filtered['waste_rate'] = df_filtered['waste'] / df_filtered['Production']
df_filtered['avg_inventory'] = (df_filtered['Begin month inventory'] + df_filtered['Month-end inventory']) / 2
df_filtered['inventory_turnover'] = df_filtered['Domestic'] / df_filtered['avg_inventory'].replace(0, 0.001)  # Avoid division by zero
df_filtered['capacity_utilization'] = df_filtered['Production'] / df_filtered['Capacity'].replace(0, 0.001)  # Avoid division by zero
df_filtered['value_per_unit'] = df_filtered['Shipment value (thousand baht)'] / df_filtered['Total'].replace(0, 0.001)  # Avoid division by zero
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
    overall_waste_rate = total_waste / total_production if total_production > 0 else 0
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
    waste_by_product['waste_rate'] = waste_by_product['waste'] / waste_by_product['Production'].replace(0, 0.001)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(waste_by_product, x='Product', y='waste',
                     title="Total Waste by Product")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.bar(waste_by_product, x='Product', y='waste_rate',
                     title="Waste Rate by Product")
        st.plotly_chart(fig, use_container_width=True)

# Continue with the rest of your tabs and visualizations...
# [The rest of your tab2, tab3, tab4, tab5 code remains the same]

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
    waste_by_product['waste_rate'] = waste_by_product['waste'] / waste_by_product['Production'].replace(0, 0.001)

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















