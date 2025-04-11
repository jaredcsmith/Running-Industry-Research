import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Page configuration
st.set_page_config(
    page_title="Running Industry Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Replace the current custom styling section with:

# Custom styling
st.markdown("""
    <style>
        .main > div {
            max-width: 1200px;
            padding-left: 1rem;
            padding-right: 1rem;
        }
        .stDataFrame {
            width: 100%;
        }
        
        .big-font {
            font-size: 24px !important;
            font-weight: bold;
            color: var(--text-color);
        }
        .medium-font {
            font-size: 18px !important;
            color: var(--text-color);
        }
        .insight-box {
            background-color: #f7f7f7; /* Subtle gray background */
            border-radius: 10px; /* Rounded corners */
            border: 1px solid #dcdcdc; /* Light gray border */
            padding: 10px;
            margin-bottom: 10px;
        }
        .header-style {
            background-color: #f7f7f7; /* Subtle gray background */
            border-radius: 10px; /* Rounded corners */
            border: 1px solid #dcdcdc; /* Light gray border */
            padding: 10px;
            margin-bottom: 15px;
        }
        /* Custom metric styling */
        .metric-container {
            background: var(--background-color);
            color: var(--text-color);
            border: 1px solid var(--primary-color);
            border-radius: 7px;
            padding: 15px;
            text-align: center;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .metric-label {
            font-size: 14px;
            color: var(--text-color);
            margin-bottom: 5px;
        }
        .metric-value {
            font-size: 24px;
            font-weight: bold;
            color: var(--text-color);
        }
        
        /* Dark mode specific overrides */
        [data-testid="stSidebar"] {
            background-color: var(--background-color);
            color: var(--text-color);
        }
        
        /* Make text in markdown elements readable in both modes */
        .element-container {
            color: var(--text-color);
        }
        
        /* Ensure box backgrounds are visible but not too strong */
        .insight-box, .header-style {
            background-color: color-mix(in srgb, var(--background-color) 95%, var(--primary-color) 5%);
        }
        
        /* Style links appropriately for both modes */
        a {
            color: var(--primary-color) !important;
        }
        a:hover {
            color: var(--primary-color) !important;
            opacity: 0.8;
        }
            

    </style>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('FootwearData_cleaned.csv')
    df = df[(df['Brand'] != 'Jordan') & (df['Brand'] != 'Kailas') & (df['Brand'] != 'N') & (df['Brand'] != 'La Sportiva' )& (df['Brand'] != 'The' )]
   
   # Clean and format data
    df['Price'] = df['Price'].replace('[\$,]', '', regex=True).astype(float)
    # Create Rating Category
    df['Rating Category'] = pd.cut(
        df['Audience Rating'],
        bins=[0, 75, 90, 100],
        labels=['Low (<75)', 'Medium (75-90)', 'High (90+)']
    )
    # Make Daily Running
    df['Pace'] = df['Pace'].replace({'Daily running': 'Daily Running'})
    # Make Pace ordered
    df['Pace'] = pd.Categorical(
        df['Pace'],
        categories=['Daily Running', 'Tempo', 'Competition'],
        ordered=True
    )
    return df
df = load_data()
running_data = df[df['Category'] == 'Running'].copy()

# Sidebar
with st.sidebar:
    st.markdown("""
    <style>
        [data-testid="stSidebar"] {
            color: gray;
            border: 1px solid #dcdcdc;
            border-radius: 10px;
            padding: 10px;
            background-color: #f9f9f9;
        }
        .stButton > button {
            background-color: #007bff;
            color: gray;
            border: none;
            border-radius: 5px;
            padding: 8px 16px;
            font-size: 14px;
            cursor: pointer;
        }
        .stButton > button:hover {
            background-color: #0056b3;
        }
        .stSelectbox > div {
            border: 1px solid gray;
            border-radius: 5px;
            padding: 5px;
        }
    </style>
    """, unsafe_allow_html=True)

    st.title("Running Industry Research 👟")
    st.markdown("### Explore the latest trends in the running footwear market")

    # Filter options
    st.subheader("Analysis Filters")
    selected_brand = st.selectbox("Select Brand", ["ALL"] + sorted(running_data['Brand'].unique()))
    
    # Dynamic product filter based on brand selection
    if selected_brand != "ALL":
        product_filter = ["ALL"] + sorted(running_data[running_data['Brand'] == selected_brand]['Product Name'].dropna().unique())
    else:
        product_filter = ["ALL"] + sorted(running_data['Product Name'].dropna().unique())
    chosen_product = st.selectbox("Select Product", product_filter)
    
    # Feature selection
    selected_feature = st.selectbox("Select Metric for Analysis", sorted([
        'Drop (mm)', 
        'Flexibility / Stiffness (average) (N)', 
        'Forefoot stack (mm)', 
        'Heel stack (mm)', 
        'Insole thickness (mm)', 
        'Midsole width - forefoot (mm)', 
        'Midsole width - heel (mm)', 
        'Outsole hardness (HC)', 
        'Outsole thickness (mm)', 
        'Price', 
        'Stiffness in cold (%)', 
        'Stiffness in cold (N)', 
        'Tongue padding (mm)', 
        'Weight (g)', 
        'Weight (oz)'
    ]))
    
    # Key metrics summary
    st.subheader("Market Overview")
    total_products = running_data['Brand'].count()
    avg_rating = running_data['Audience Rating'].mean()
    top_brand = running_data.groupby('Brand').size().idxmax()
    
    col1, col2 = st.columns(2)
    col1.metric("Number of Products", f"{total_products:.0f}")
    col2.metric("Avg. Rating", f"{avg_rating:.1f}")
    
    # Download option
    st.download_button(
        "Download Data",
        data=running_data.to_csv().encode(),
        file_name="running_research_data.csv",
        mime="text/csv"
    )

# Filter data based on selections
if selected_brand != "ALL":
    plot_data = running_data[running_data['Brand'] == selected_brand]
else:
    plot_data = running_data

# Main content
# Header with key insights
st.markdown("<h1 style='text-align: center;'>Running Shoe Market Analysis Dashboard</h1>", unsafe_allow_html=True)

# Brief data story
st.markdown("""
<div class='header-style'>
<b>The Story Behind the Data:</b> This dashboard explores the technical specifications, price points, and customer satisfaction 
across different running shoe categories. Discover how features like stack height, weight, and flexibility 
influence performance ratings and how brands position themselves across different running segments.
</div>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: left;'>Product Analysis - Select a Product in the sidebar to Research</h1>", unsafe_allow_html=True)

# Section above row 1 with 1 row and 4 columns
col1, col2, col3, col4 = st.columns(4)

# Filter for selected product
filtered = running_data.copy()
if selected_brand != "ALL":
    filtered = filtered[filtered['Brand'] == selected_brand]
if chosen_product != "ALL":
    filtered = filtered[filtered['Product Name'] == chosen_product]

if not filtered.empty:
    shoe = filtered.iloc[0]

    # Product Name
    col1.markdown(
        f"<div style='background:linear-gradient(to right, #6a11cb, #2575fc); padding:10px; border-radius:5px; color:white; text-align:center;'>"
        f"<h3>{shoe['Brand']} { shoe['Product Name']}</h3>"
        f"</div>",
        unsafe_allow_html=True
    )
    # Price
    price = shoe['Price']
    average_price = running_data['Price'].mean()
    price_difference = price - average_price

    if price_difference > 0:
        arrow = "⬆️"
        diff_color = "red"
    else:
        arrow = "⬇️"
        diff_color = "green"

    col3.markdown(
        f"<div style='background:linear-gradient(to right, #ff7e5f, #feb47b); padding:10px; border-radius:5px; color:white; text-align:center;'>"
        f"<h3>Price: ${price:.2f} <span style='color:{diff_color};'>{arrow} $({price_difference:+.2f})</span></h3>"
        f"</div>",
        unsafe_allow_html=True
    )

    # Pace
    col2.markdown(
        f"<div style='background:linear-gradient(to right, #43cea2, #185a9d); padding:10px; border-radius:5px; color:white; text-align:center;'>"
        f"<h3>{shoe['Pace']}</h3>"
        f"</div>",
        unsafe_allow_html=True
    )

    audience_rating = shoe['Audience Rating']
    average_rating = running_data['Audience Rating'].mean()
    rating_difference = audience_rating - average_rating

    if audience_rating < 80:
        rating_color = "linear-gradient(to right, #ff4e50, #f9d423)"
    elif 80 <= audience_rating <= 90:
        rating_color = "linear-gradient(to right, #f7971e, #ffd200)"
    else:
        rating_color = "linear-gradient(to right, #56ab2f, #a8e063)"

    if rating_difference > 0:
        arrow = "⬆️"
        diff_color = "green"
    else:
        arrow = "⬇️"
        diff_color = "red"

    col4.markdown(
    f"<div style='background:{rating_color}; padding:10px; border-radius:5px; color:white; text-align:center;'>"
    f"<h3>Rating: {audience_rating:.1f} <span style='color:{diff_color};'>{arrow} ({rating_difference:+.1f})</span></h3>"
    f"</div>",
    unsafe_allow_html=True
    )


# Create two rows with two columns each using custom widths
row1_col1, row1_col2 = st.columns([5, 5])
st.markdown("<h1 style='text-align: left;'>Technical Brand Analysis - Select a Brand and Feature to Research</h1>", unsafe_allow_html=True)

# Create a row with three columns
col1, col2, col3 = st.columns(3)

# Filtered data based on selected brand
filtered = running_data.copy()
if selected_brand != "ALL":
    filtered = filtered[filtered['Brand'] == selected_brand]

if not filtered.empty:
    brand_name = selected_brand
    total_products = filtered['Product Name'].nunique()
    avg_audience_rating = filtered['Audience Rating'].mean()

    # Brand Name
    col1.markdown(
        f"<div style='background:linear-gradient(to right, #6a11cb, #2575fc); padding:10px; border-radius:5px; color:white; text-align:center;'>"
        f"<h3>{brand_name}</h3>"
        f"</div>",
        unsafe_allow_html=True
    )

    # Total Products
    col2.markdown(
        f"<div style='background:linear-gradient(to right, #ff7e5f, #feb47b); padding:10px; border-radius:5px; color:white; text-align:center;'>"
        f"<h3>Total Products: {total_products}</h3>"
        f"</div>",
        unsafe_allow_html=True
    )

    # Average Audience Rating
    audience_rating = avg_audience_rating
    average_rating = running_data['Audience Rating'].mean()
    rating_difference = audience_rating - average_rating

    if audience_rating < 80:
        rating_color = "linear-gradient(to right, #ff4e50, #f9d423)"
    elif 80 <= audience_rating <= 90:
        rating_color = "linear-gradient(to right, #f7971e, #ffd200)"
    else:
        rating_color = "linear-gradient(to right, #56ab2f, #a8e063)"

    if rating_difference > 0:
        arrow = "⬆️"
        diff_color = "green"
    else:
        arrow = "⬇️"
        diff_color = "red"

    col3.markdown(
        f"<div style='background:{rating_color}; padding:10px; border-radius:5px; color:white; text-align:center;'>"
        f"<h3>Avg. Rating: {audience_rating:.1f} <span style='color:{diff_color};'>{arrow} ({rating_difference:+.1f})</span></h3>"
        f"</div>",
        unsafe_allow_html=True
    )




row2_col1, row2_col2 = st.columns([5, 5])



# Plot 3: Radar Chart (Product Comparison)
with row1_col1:
    
    st.markdown("### Product Technical Profile")
    
    # Filter for selected product
    filtered = running_data.copy()
    if selected_brand != "ALL":
        filtered = filtered[filtered['Brand'] == selected_brand]
    if chosen_product != "ALL":
        filtered = filtered[filtered['Product Name'] == chosen_product]
    
    # Add insights box if a product is selected
    if chosen_product != "ALL":
        product_name = filtered.iloc[0]['Product Name'] if not filtered.empty else chosen_product
        st.markdown(f"""
        <div class='insight-box'>
        <b>Insight:</b> This radar chart shows how {product_name} compares to the market average across 
        key technical specifications. The larger the blue area, the more the shoe exceeds average values 
        for these metrics. 
        Note - These values are normailized to a scale of 0-1 for comparison purposes (original are in parentheses)
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class='insight-box'>
        <b>Insight:</b> Select a specific product from the sidebar to see how it compares to market averages
        across key technical specifications.
        </div>
        """, unsafe_allow_html=True)
    

    # Create radar chart
    scaler = MinMaxScaler()
    radar_features = [
        'Price', 
        'Drop (mm)', 
        'Forefoot stack (mm)', 
        'Heel stack (mm)', 
        'Insole thickness (mm)', 
        'Midsole width - forefoot (mm)', 
        'Midsole width - heel (mm)', 
        'Outsole thickness (mm)', 
        'Toebox width - widest part (average) (mm)', 
        'Tongue padding (mm)', 
        'Weight (g)'
    ]
    
    # Filter features that exist in the dataset
    valid_features = [f for f in radar_features if f in running_data.columns]
    
    normalized_data = running_data[valid_features].copy()
    normalized_data = pd.DataFrame(
        scaler.fit_transform(normalized_data),
        columns=normalized_data.columns,
        index=running_data.index
    )
    
    shoe_index = shoe.name
    shoe_normalized = normalized_data.loc[shoe_index].dropna()
    
    # Calculate averages for comparison
    normalized_data_without_selected = normalized_data.drop(index=shoe_index)
    average_normalized_values = normalized_data_without_selected.mean()
    
    # High rating category average
    rating_category_averages = {}
    high_category_data = running_data[running_data['Rating Category'] == 'High (90+)']
    if not high_category_data.empty:
        high_indices = high_category_data.index
        high_category_normalized = normalized_data.loc[high_indices]
        rating_category_averages['High (90+)'] = high_category_normalized.mean()
    
    # Create dataframe for plotting
    plot_df = pd.DataFrame({
        'Feature': shoe_normalized.index,
        'Normalized Value': shoe_normalized.values,
        'Average Value': average_normalized_values[shoe_normalized.index].values
    })
    
    # Create radar chart
    categories = plot_df['Feature']
    values = plot_df['Normalized Value'].values.tolist()
    values += values[:1]  # Close the loop
    
    average_values = plot_df['Average Value'].values.tolist()
    average_values += average_values[:1]  # Close the loop
    
    category_values = {}
    for category, averages in rating_category_averages.items():
        category_values[category] = averages[plot_df['Feature']].values.tolist()
        category_values[category] += category_values[category][:1]  # Close the loop
    
    # Configure the plot
    num_vars = len(categories)
    angles = [n / float(num_vars) * 2 * np.pi for n in range(num_vars)]
    angles += angles[:1]  # Close the loop
    
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    
    # Plot selected product
    ax.plot(angles, values, linewidth=2.5, linestyle='solid', label='Selected Product', color='#1f77b4')
    ax.fill(angles, values, color='#1f77b4', alpha=0.25)
    
    # Plot average
    ax.plot(angles, average_values, linewidth=2, linestyle='dashed', label='Market Average', color='#ff7f0e')
    
    # Plot high rating category if available
    for category, category_vals in category_values.items():
        ax.plot(angles, category_vals, linestyle='dotted', linewidth=1.5, 
                label=f'Top Rated Average', color='#2ca02c')
    
    # Customize
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], color="gray", size=8)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([f.split(' (')[0] for f in categories], fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.7)
            
    # Add legend with better positioning
    ax.legend(loc='lower right', bbox_to_anchor=(0.1, 0.1))
    
    plt.tight_layout()
    st.pyplot(fig)

# Plot 4: Enhanced Analysis (Price vs Rating Scatter)
with row1_col2:
    st.markdown("### Price vs. Rating Relationship")
    
    # Add insights box
    st.markdown("""
    <div class='insight-box'>
    <b>Insight:</b> This visualization explores the relationship between price and customer satisfaction 
    across different performance categories, helping to identify value leaders and premium performers.
    </div>
    """, unsafe_allow_html=True)
    
    # Use the full dataset for this plot, unaffected by brand selection
    full_plot_data = running_data.copy()
    
    # Create an enhanced scatter plot
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Create categorical colors and markers for Pace categories
    pace_colors = {'Daily Running': '#3498db', 'Tempo': '#e74c3c', 'Competition': '#2ecc71'}
    pace_markers = {'Daily Running': 'o', 'Tempo': 's', 'Competition': '^'}
    
    # Plot each pace category 
    for pace, group in full_plot_data.groupby('Pace'):
        ax.scatter(
            group['Price'], 
            group['Audience Rating'],
            s=80,
            alpha=0.7,
            c=pace_colors[pace],
            marker=pace_markers[pace],
            label=pace,
            edgecolors='white',
            linewidths=0.5
        )
    
    # Add trend line for each category
    for pace, group in full_plot_data.groupby('Pace'):
        if len(group) > 1:  # Need at least 2 points for regression
            x = group['Price']
            y = group['Audience Rating']
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            ax.plot(x, p(x), linestyle='--', color=pace_colors[pace], alpha=0.6)
    
    # If a specific product is selected, highlight it
    if chosen_product != "ALL" and chosen_product in full_plot_data['Product Name'].values:
        product_data = full_plot_data[full_plot_data['Product Name'] == chosen_product]
        if not product_data.empty:
            ax.scatter(
                product_data['Price'],
                product_data['Audience Rating'],
                s=150,
                facecolors='none',
                edgecolors='black',
                linewidths=2,
                zorder=10
            )
            # Add annotation
            for _, row in product_data.iterrows():
                ax.annotate(
                    row['Product Name'],
                    (row['Price'], row['Audience Rating']),
                    xytext=(10, 5),
                    textcoords='offset points',
                    fontsize=9,
                    weight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
                )
    
    # Add quadrant lines and labels
    avg_price = full_plot_data['Price'].mean()
    avg_rating = full_plot_data['Audience Rating'].mean()
    
    # Draw quadrant lines
    ax.axhline(y=avg_rating, color='gray', linestyle=':', alpha=0.6)
    ax.axvline(x=avg_price, color='gray', linestyle=':', alpha=0.6)
    
    # Add quadrant labels
    ax.text(
        full_plot_data['Price'].max() * 0.95, 
        avg_rating + (full_plot_data['Audience Rating'].max() - avg_rating) * 0.75, 
        "Premium Performers", 
        fontsize=9, 
        ha='right',
        bbox=dict(boxstyle="round,pad=0.3", fc="#f0f2f6", ec="gray", alpha=0.8)
    )
    ax.text(
        avg_price - (avg_price - full_plot_data['Price'].min()) * 0.5, 
        avg_rating + (full_plot_data['Audience Rating'].max() - avg_rating) * 0.75, 
        "Value Leaders", 
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="#f0f2f6", ec="gray", alpha=0.8)
    )
    
    # Customize
    ax.set_xlabel('Price ($)', fontweight='bold')
    ax.set_ylabel('Audience Rating', fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # Add legend
    ax.legend(title="Performance Category", loc='lower right')
    
    # Add correlation coefficient
    corr = full_plot_data['Price'].corr(full_plot_data['Audience Rating'])
    ax.text(
        0.05, 0.05, 
        f"Correlation: {corr:.2f}", 
        transform=ax.transAxes, 
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
    )
    
    plt.tight_layout()
    st.pyplot(fig)


# Plot 1: Feature Distribution by Pace and Rating
with row2_col1:
    if selected_feature in plot_data.columns:
        st.markdown(f"### {selected_feature} Distribution by Performance Category")
        
        # Add insights box
        st.markdown(f"""
        <div class='insight-box'>
        <b>Insight:</b> This visualization reveals how {selected_feature.lower()} varies across different 
        performance categories and rating classes. Identifying these patterns can help determine optimal 
        specifications for each shoe type.
        </div>
        """, unsafe_allow_html=True)
        
        # Create the plot
        grouped_data = plot_data.groupby(['Pace', 'Rating Category'])[selected_feature].apply(list).unstack()
        fig, ax = plt.subplots(figsize=(6, 4))
        positions = []
        labels = []
        colors = ['#FF5A5F', '#FFC107', '#4CAF50']  # More professional colors for Low, Medium, High
        
        # Add color indicators above the plot
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(
            f"<div style='background:{colors[0]}; padding:8px; border-radius:5px; color:white; text-align:center;'>"
            f"<b>Low Rating (<75)</b>"
            f"</div>",
            unsafe_allow_html=True
            )
        with col2:
            st.markdown(
            f"<div style='background:{colors[1]}; padding:8px; border-radius:5px; color:white; text-align:center;'>"
            f"<b>Medium Rating (75-90)</b>"
            f"</div>",
            unsafe_allow_html=True
            )
        with col3:
            st.markdown(
            f"<div style='background:{colors[2]}; padding:8px; border-radius:5px; color:white; text-align:center;'>"
            f"<b>High Rating (90+)</b>"
            f"</div>",
            unsafe_allow_html=True
            )
            
        for i, (rating_category, color) in enumerate(zip(grouped_data.columns, colors)):
            for j, pace in enumerate(grouped_data.index):
                data = grouped_data.loc[pace, rating_category]
                if not (pd.isna(data) if isinstance(data, (float, int, str)) else pd.isna(data).any()):
                    positions.append(j + i * 0.2)  # Offset positions for each category
                    ax.boxplot(
                        data, 
                        positions=[j + i * 0.2], 
                        widths=0.15, 
                        patch_artist=True, 
                        boxprops=dict(facecolor=color, color='black'), 
                        whiskerprops=dict(color='black'), 
                        capprops=dict(color='black'), 
                        medianprops=dict(color='black')
                    )
            labels.append(rating_category)
            
        ax.set_xticks(range(len(grouped_data.index)))
        ax.set_xticklabels(grouped_data.index, rotation=45)
        ax.set_xlabel("Performance Category", fontweight='bold')
        ax.set_ylabel(selected_feature, fontweight='bold')
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        st.pyplot(fig)

# Plot 2: Brand Performance Table
with row2_col2:
    st.markdown("### Brand Performance by Category")
    
    # Add insights box
    st.markdown("""
    <div class='insight-box'>
    <b>Insight:</b> This table highlights how different brands perform across pace categories, showing the 
    correlation between price point, audience satisfaction, and market presence. Brands targeting specific 
    niches often show distinct performance patterns.
    </div>
    """, unsafe_allow_html=True)
    
    top_brands = running_data.groupby(['Brand', 'Pace']).size().reset_index(name='Count')
    top_brands = top_brands[top_brands['Count'] >= 1]['Brand'].unique().tolist()
    top_brands_data = running_data[running_data['Brand'].isin(top_brands)]
    
    top_brands_grouped = top_brands_data.groupby(['Brand', 'Pace']).agg({
        'Audience Rating': 'mean',
        'Price': 'mean',
        'Pace': 'count'
    }).rename(columns={'Pace': 'Count'}).reset_index().rename(columns={'Pace': 'Category'})
    
    top_brands_grouped = top_brands_grouped.rename(columns={
        'Audience Rating': 'Avg. Rating',
        'Price': 'Avg. Price ($)'
    })
    
    # Format columns
    top_brands_grouped['Avg. Rating'] = top_brands_grouped['Avg. Rating'].round(1)
    top_brands_grouped['Avg. Price ($)'] = top_brands_grouped['Avg. Price ($)'].round(2)
    top_brands_grouped['Brand'] = top_brands_grouped['Brand'].str.title()
    
    # Sort and filter
    top_brands_grouped = top_brands_grouped.sort_values(by=['Brand', 'Category'])
    top_brands_grouped = top_brands_grouped[top_brands_grouped['Count'] > 0]
    
    # Highlight function
    def highlight_selected_brand(row):
        if selected_brand != "ALL" and row['Brand'] == selected_brand.title():
            return ['background-color: rgba(255, 127, 14, 0.2)'] * len(row)
        return [''] * len(row)
    
    # Color code ratings
    def color_rating(val):
        if val >= 90:
            return f'background-color: rgba(76, 175, 80, 0.2)'
        elif val >= 75:
            return f'background-color: rgba(255, 193, 7, 0.2)'
        else:
            return f'background-color: rgba(255, 90, 95, 0.2)'
    
    styled_df = top_brands_grouped.style\
        .apply(highlight_selected_brand, axis=1)\
        .format({'Avg. Price ($)': '${:.2f}', 'Avg. Rating': '{:.1f}'})\
        .background_gradient(subset=['Count'], cmap='Blues')\
        .applymap(lambda x: color_rating(x) if isinstance(x, (int, float)) and 0 <= x <= 100 else '', subset=['Avg. Rating'])
    
    st.dataframe(
        styled_df,
        use_container_width=True,
        height=400
    )

# Plot 5: Enhanced Analysis (Price vs Selected Feature Scatter)
# with row1_col2:
#     st.markdown(f"### {selected_feature} vs. Audience Rating Relationship")
    
#     # Add insights box
#     st.markdown(f"""
#     <div class='insight-box'>
#     <b>Insight:</b> This visualization explores the relationship between {selected_feature.lower()} and audience rating 
#     across different performance categories, helping to identify trends and outliers.
#     </div>
#     """, unsafe_allow_html=True)
    
#     # Create an enhanced scatter plot
#     fig, ax = plt.subplots(figsize=(8, 5))
    
#     # Create categorical colors and markers for Pace categories
#     pace_colors = {'Daily running': '#3498db', 'Tempo': '#e74c3c', 'Competition': '#2ecc71'}
#     pace_markers = {'Daily running': 'o', 'Tempo': 's', 'Competition': '^'}
    
#     # Plot each pace category 
#     for pace, group in plot_data.groupby('Pace'):
#         ax.scatter(
#             group[selected_feature], 
#             group['Audience Rating'],
#             s=80,
#             alpha=0.7,
#             c=pace_colors[pace],
#             marker=pace_markers[pace],
#             label=pace,
#             edgecolors='white',
#             linewidths=0.5
#         )
    
#     # Add trend line for each category
#     for pace, group in plot_data.groupby('Pace'):
#         if len(group) > 1 and np.ptp(group[selected_feature]) > 0 and np.ptp(group['Audience Rating']) > 0:  # Ensure sufficient variance
#             x = group[selected_feature]
#             y = group['Audience Rating']
#             try:
#                 z = np.polyfit(x, y, 1)
#                 p = np.poly1d(z)
#                 ax.plot(x, p(x), linestyle='--', color=pace_colors[pace], alpha=0.6)
#             except np.linalg.LinAlgError:
#                 st.warning(f"Could not compute regression for {pace} due to numerical instability.")
    
#     # If a specific product is selected, highlight it
#     if chosen_product != "ALL" and chosen_product in plot_data['Product Name'].values:
#         product_data = plot_data[plot_data['Product Name'] == chosen_product]
#         if not product_data.empty:
#             ax.scatter(
#                 product_data[selected_feature],
#                 product_data['Audience Rating'],
#                 s=150,
#                 facecolors='none',
#                 edgecolors='black',
#                 linewidths=2,
#                 zorder=10
#             )
#             # Add annotation
#             for _, row in product_data.iterrows():
#                 ax.annotate(
#                     row['Product Name'],
#                     (row[selected_feature], row['Audience Rating']),
#                     xytext=(10, 5),
#                     textcoords='offset points',
#                     fontsize=9,
#                     weight='bold',
#                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
#                 )
    
#     # Add quadrant lines and labels
#     avg_feature = plot_data[selected_feature].mean()
#     avg_rating = plot_data['Audience Rating'].mean()
    
#     # Draw quadrant lines
#     ax.axhline(y=avg_rating, color='gray', linestyle=':', alpha=0.6)
#     ax.axvline(x=avg_feature, color='gray', linestyle=':', alpha=0.6)
    
#     # Add quadrant labels
#     ax.text(
#         avg_feature + (plot_data[selected_feature].max() - avg_feature) * 0.75, 
#         plot_data['Audience Rating'].max() * 0.95, 
#         "High Performers", 
#         fontsize=9, 
#         ha='right',
#         bbox=dict(boxstyle="round,pad=0.3", fc="#f0f2f6", ec="gray", alpha=0.8)
#     )
#     ax.text(
#         avg_feature + (plot_data[selected_feature].max() - avg_feature) * 0.75, 
#         avg_rating - (avg_rating - plot_data['Audience Rating'].min()) * 0.5, 
#         "Improvement Needed", 
#         fontsize=9,
#         bbox=dict(boxstyle="round,pad=0.3", fc="#f0f2f6", ec="gray", alpha=0.8)
#     )
    
#     # Customize
#     ax.set_xlabel(selected_feature, fontweight='bold')
#     ax.set_ylabel('Audience Rating', fontweight='bold')
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
#     ax.grid(True, linestyle='--', alpha=0.3)
    
#     # Add legend
#     ax.legend(title="Performance Category", loc='lower right')
    
#     # Add correlation coefficient
#     corr = plot_data[selected_feature].corr(plot_data['Audience Rating'])
#     ax.text(
#         0.05, 0.05, 
#         f"Correlation: {corr:.2f}", 
#         transform=ax.transAxes, 
#         fontsize=9,
#         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
#     )
    
#     plt.tight_layout()
#     st.pyplot(fig)
# Footer with methodology note
st.markdown("""
---
**Methodology Note:** This analysis is based on technical measurements and audience ratings 
collected from running footwear testing. All measurements follow industry standards for 
consistency and comparability.
""")