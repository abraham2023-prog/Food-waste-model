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

# Initialize session state
if 'df_processed' not in st.session_state:
    st.session_state.df_processed = None
if 'column_mapping' not in st.session_state:
    st.session_state.column_mapping = {}
if 'mapping_complete' not in st.session_state:
    st.session_state.mapping_complete = False

# Load and clean data
with st.sidebar:
    st.header("Data Configuration")
    st.subheader("Upload Data")
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

# Process uploaded file
if uploaded_file is not None:
    if st.session_state.df_processed is None:
        df = pd.read_csv(uploaded_file)
        
        # Store original column names for reference
        original_columns = df.columns.tolist()
        st.session_state.original_columns = original_columns
        
        # Simple normalization (convert to lowercase and replace spaces with underscores)
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('-', '_')
        st.session_state.df_raw = df
        
        # Check for required columns
        required_cols = ['begin_month_inventory', 'production', 'domestic', 'export', 'month_end_inventory']
        available_cols = df.columns.tolist()
        
        missing_cols = [c for c in required_cols if c not in available_cols]
        
        if missing_cols:
            st.session_state.missing_cols = missing_cols
            st.session_state.mapping_needed = True
        else:
            st.session_state.mapping_needed = False
            st.session_state.df_processed = df
            st.session_state.mapping_complete = True

# Show column mapping interface if needed
if uploaded_file is not None and st.session_state.get('mapping_needed', False):
    st.warning("Some required columns are missing from your dataset.")
    st.info("Please help us map your dataset columns to the required columns:")
    
    st.write("**Your dataset columns:**")
    for col in st.session_state.original_columns:
        st.write(f"- '{col}'")
    
    st.write("---")
    
    # Create mapping interface
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.write("**Required Column**")
        st.write("Begin month inventory")
        st.write("Month-end inventory")
        st.write("Production")
        st.write("Domestic")
        st.write("Export")
    
    with col2:
        st.write("**Map to Your Column**")
        
        # Get available columns for mapping
        available_options = [""] + st.session_state.df_raw.columns.tolist()
        
        # Create select boxes for each required column
        begin_map = st.selectbox(
            "Select column for Begin month inventory",
            options=available_options,
            key="begin_map",
            label_visibility="collapsed"
        )
        
        end_map = st.selectbox(
            "Select column for Month-end inventory",
            options=available_options,
            key="end_map",
            label_visibility="collapsed"
        )
        
        production_map = st.selectbox(
            "Select column for Production",
            options=available_options,
            key="production_map",
            label_visibility="collapsed"
        )
        
        domestic_map = st.selectbox(
            "Select column for Domestic",
            options=available_options,
            key="domestic_map",
            label_visibility="collapsed"
        )
        
        export_map = st.selectbox(
            "Select column for Export",
            options=available_options,
            key="export_map",
            label_visibility="collapsed"
        )
    
    if st.button("Apply Mapping"):
        # Store the mapping
        mapping = {
            'begin_month_inventory': begin_map,
            'month_end_inventory': end_map,
            'production': production_map,
            'domestic': domestic_map,
            'export': export_map
        }
        
        # Apply the mapping
        df_processed = st.session_state.df_raw.copy()
        
        for required_col, dataset_col in mapping.items():
            if dataset_col and dataset_col in df_processed.columns:
                df_processed[required_col] = df_processed[dataset_col]
        
        # Check if we have all required columns now
        missing_after_mapping = [c for c in ['begin_month_inventory', 'month_end_inventory', 'production', 'domestic', 'export'] 
                               if c not in df_processed.columns]
        
        if missing_after_mapping:
            st.error(f"Still missing columns after mapping: {missing_after_mapping}")
        else:
            st.session_state.df_processed = df_processed
            st.session_state.mapping_complete = True
            st.session_state.column_mapping = mapping
            st.success("Column mapping applied successfully!")
            st.rerun()

# Use sample data if no file uploaded or mapping not complete
if uploaded_file is None or not st.session_state.get('mapping_complete', False):
    if uploaded_file is None:
        st.info("Using sample data for demonstration")
    else:
        st.info("Please complete the column mapping above to proceed with your data")
    
    df = generate_sample_data()
else:
    df = st.session_state.df_processed
    st.success("Using your uploaded data with applied column mapping")

# Now process the numeric columns
numeric_cols = ['begin_month_inventory', 'production', 'domestic', 'export', 
                'month_end_inventory', 'capacity', 'shipment_value_thousand_baht', 'total']

for col in numeric_cols:
    if col in df.columns:
        df[col] = df[col].astype(str).str.replace(',', '').astype(float)

# Analysis parameters in sidebar
with st.sidebar:
    st.subheader("Analysis Parameters")
    
    # Get unique products, handling potential missing 'Product' column
    if 'product' in df.columns:
        product_options = df['product'].unique()
    else:
        product_options = ["Unknown Product"]
        df['product'] = "Unknown Product"
    
    selected_products = st.multiselect(
        "Select Products to Analyze",
        options=product_options,
        default=product_options[:3] if len(product_options) > 3 else product_options
    )

    # Get year range, handling potential missing 'year' column
    if 'year' in df.columns:
        min_year = int(df['year'].min())
        max_year = int(df['year'].max())
    else:
        min_year = 2020
        max_year = 2023
        df['year'] = 2022  # Default year
    
    year_range = st.slider(
        "Select Year Range",
        min_value=min_year,
        max_value=max_year,
        value=(min_year, max_year)
    )

    st.subheader("Waste Calculation")
    st.info("Waste = (Begin Inventory + Production) - (Domestic + Export + End Inventory)")

# Filter data based on selections
df_filtered = df[
    (df['product'].isin(selected_products)) &
    (df['year'] >= year_range[0]) &
    (df['year'] <= year_range[1])
].copy()

# Calculate waste metrics (with checks for required columns)
required_waste_cols = ['begin_month_inventory', 'production', 'domestic', 'export', 'month_end_inventory']
has_all_waste_cols = all(col in df_filtered.columns for col in required_waste_cols)

if has_all_waste_cols:
    df_filtered['waste'] = (df_filtered['begin_month_inventory'] + df_filtered['production']) - (df_filtered['domestic'] + df_filtered['export'] + df_filtered['month_end_inventory'])
    df_filtered['waste_rate'] = df_filtered['waste'] / df_filtered['production'].replace(0, 0.001)
    df_filtered['avg_inventory'] = (df_filtered['begin_month_inventory'] + df_filtered['month_end_inventory']) / 2
    df_filtered['inventory_turnover'] = df_filtered['domestic'] / df_filtered['avg_inventory'].replace(0, 0.001)
else:
    st.warning("Cannot calculate waste metrics - missing required inventory columns")
    df_filtered['waste'] = 0
    df_filtered['waste_rate'] = 0
    df_filtered['avg_inventory'] = 0
    df_filtered['inventory_turnover'] = 0

# Calculate other metrics
if 'capacity' in df_filtered.columns:
    df_filtered['capacity_utilization'] = df_filtered['production'] / df_filtered['capacity'].replace(0, 0.001)
else:
    df_filtered['capacity_utilization'] = 0

if all(col in df_filtered.columns for col in ['shipment_value_thousand_baht', 'total']):
    df_filtered['value_per_unit'] = df_filtered['shipment_value_thousand_baht'] / df_filtered['total'].replace(0, 0.001)
    df_filtered['waste_value'] = df_filtered['waste'] * df_filtered['value_per_unit']
else:
    df_filtered['value_per_unit'] = 0
    df_filtered['waste_value'] = 0

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
if has_all_waste_cols and not df_filtered.empty:
    st.markdown("---")
    st.markdown('<p class="section-header">📋 Key Insights and Recommendations</p>', unsafe_allow_html=True)

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

# Reset button
if st.sidebar.button("Reset Data & Mapping"):
    for key in ['df_processed', 'df_raw', 'mapping_complete', 'mapping_needed', 'column_mapping']:
        if key in st.session_state:
            del st.session_state[key]
    st.rerun()

















