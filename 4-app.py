import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Running Industry Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
    # Rename Pace categories
    df['Pace'] = df['Pace'].replace({
        'Daily running': 'Everyday',
        'Tempo': 'Fast',
        'Competition': 'Race'
    })
    
    # Make Pace ordered
    df['Pace'] = pd.Categorical(
        df['Pace'],
        categories=['Everyday', 'Fast', 'Race'],
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
            # Extract the year from the 'Original Review Date' column
    # Extract the year from the 'Original Review Date' column
    running_data['Year'] = pd.to_datetime(running_data['Original Review Date'], format='%b %d, %Y').dt.year

    # Create a separate copy of the data for time series analysis
    time_series_data = running_data.copy()

    # Add a year filter to the sidebar
    with st.sidebar:
        
        available_years = sorted(running_data['Year'].dropna().unique(), reverse=True)  # Get unique years in descending order
        available_years = [year for year in available_years if year >= 2021]  # Remove years before 2021
        selected_year = st.selectbox("Select Year", ["ALL"] + [str(year) for year in available_years])

    # Filter the data based on the selected year
    if selected_year != "ALL":
        running_data = running_data[running_data['Year'] == int(selected_year)]
    
    selected_brand = st.selectbox("Select Brand", ["ALL"] + sorted(running_data['Brand'].unique()))
    
    # Dynamic product filter based on brand selection
    if selected_brand != "ALL":
        product_filter = ["ALL"] + sorted(running_data[running_data['Brand'] == selected_brand]['Product Name'].dropna().unique())
    else:
        product_filter = ["ALL"] + sorted(running_data['Product Name'].dropna().unique())
    chosen_product = st.selectbox("Select Product", product_filter)
    # Feature selection
    selected_feature = st.selectbox(
        "Select Metric for Analysis", 
        sorted([
            'Drop (mm)', 
            'Flexibility / Stiffness (average) (N)', 
            'Forefoot stack (mm)', 
            'Heel stack (mm)', 
            'Insole thickness (mm)', 
            'Midsole width - forefoot (mm)', 
            'Midsole width - heel (mm)', 
            'Midsole softness (HA)',
            'Outsole hardness (HC)', 
            'Outsole thickness (mm)', 
            'Price', 
            'Stiffness in cold (%)', 
            'Stiffness in cold (N)', 
            'Tongue padding (mm)', 
            'Weight (g)', 
            'Weight (oz)'
        ]),
        index=9 # Ensure 'Price' is selected by default
    )

# Key metrics summary
    st.subheader("Market Overview")
    total_products = running_data['Brand'].count()
    avg_rating = running_data['Audience Rating'].mean()
    top_brand = running_data.groupby('Brand').size().idxmax()
    
    col1, col2 = st.columns(2)
    col1.metric("Number of Products", f"{total_products:.0f}")
    col2.metric("Avg. Rating", f"{avg_rating:.1f}")
    


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
<b>The Story Behind the Data:</b> This dashboard explores the technical specifications, price points, and audience reviews from e-commerce and wear-testing reviews
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
        f"<h3>{shoe['Product Name']}</h3>"
        f"</div>",
        unsafe_allow_html=True
    )

    # Price
    col2.markdown(
        f"<div style='background:linear-gradient(to right, #ff7e5f, #feb47b); padding:10px; border-radius:5px; color:white; text-align:center;'>"
        f"<h3>Price: ${shoe['Price']}</h3>"
        f"</div>",
        unsafe_allow_html=True
    )

    # Pace
    col3.markdown(
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

    # Color scheme
pace_colors = {'Everyday': '#3498db', 'Fast': '#e74c3c', 'Race': '#2ecc71'}
pace_symbols = {'Everyday': 'circle', 'Fast': 'square', 'Race': 'triangle-up'}

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

        Note - These values are normalized to a scale of 0-1 for comparison purposes - Hover over the chart to see the original values for each feature.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class='insight-box'>
        <b>Action :</b> Select a specific product from the sidebar to see how it compares to market averages
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
    
    # Prepare data for Plotly radar chart
    categories = plot_df['Feature'].tolist()
    values = plot_df['Normalized Value'].tolist()
    values.append(values[0])  # Close the loop
    
    average_values = plot_df['Average Value'].tolist()
    average_values.append(average_values[0])  # Close the loop
    
    category_values = {}
    for category, averages in rating_category_averages.items():
        category_values[category] = averages[plot_df['Feature']].tolist()
        category_values[category].append(category_values[category][0])  # Close the loop
    
    # Create Plotly radar chart
    original_values = running_data.loc[shoe_index, valid_features]
    
    # Create Plotly radar chart with modified hover template
    fig = go.Figure()


    # Format the original values, adding a dollar sign for Price
    text_values = [
        f"{cat}: ${orig:.2f}" if cat == "Price " else f"{cat}: {orig:.2f}"
        for cat, orig in zip(categories, original_values)
    ]

    # Add selected product trace with both normalized and original values
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories + [categories[0]],
        fill='toself',
        name='Selected Product',
        line=dict(color='#1f77b4'),
        text=text_values,  # Pass the formatted text values
        hovertemplate=(
            "<b>%{theta}</b><br>" +
            "Normalized: %{r:.2f}<br>" +
            "Ground Truth: %{text}<br>" +
            "<extra></extra>"
        )
    ))
    
    # Add pace category averages
    for pace in ['Everyday', 'Fast', 'Race']:
        pace_data = running_data[running_data['Pace'] == pace]
        if not pace_data.empty:
            pace_averages = pace_data[valid_features].mean()
            normalized_pace_averages = scaler.transform([pace_averages])[0]
            fig.add_trace(go.Scatterpolar(
                r=list(normalized_pace_averages) + [normalized_pace_averages[0]],
                theta=categories + [categories[0]],
                fill='none',
                name=f'{pace} Average',
                line=dict(color=pace_colors[pace], dash='dot'),
                text=[f"{cat}: {avg:.2f}" for cat, avg in zip(categories, pace_averages)],
                hovertemplate=(
                    "<b>%{theta}</b><br>" +
                    "Normalized: %{r:.2f}<br>" +
                    "Original: %{text}<br>" +
                    "<extra></extra>"
                )
            ))
    # Update layout
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickfont=dict(size=10)
            )
        ),
        showlegend=True,
        legend=dict(
            title="Legend",
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5
        ),
        margin=dict(l=20, r=20, t=20, b=20)
    )
    
    # Display the radar chart
    st.plotly_chart(fig, use_container_width=True)

# Plot 4: Enhanced Analysis (Price vs Rating Scatter)
with row1_col2:

    st.markdown(f"### {selected_feature} vs. Rating Relationship")
    
    # Add insights box
    st.markdown(f"""
    <div class='insight-box'>
    <b>Insight:</b> This visualization explores the relationship between {selected_feature.lower()} and percieved audience rating, helping to identify optimal specifications.
    <br>
    <b>Action :</b>  Change the brand, selected product and features in the sidebar to see how the correlation changes and highlight the brand's positioning.
    </div>
    """, unsafe_allow_html=True)
    
    # Use the full dataset for this plot
    full_plot_data = running_data.copy()
    
    # Create Plotly figure
    fig = go.Figure()
    

    
    # Add traces for each pace category
    for pace in full_plot_data['Pace'].dropna().unique():
        mask = full_plot_data['Pace'] == pace
        group = full_plot_data[mask]
        
        # Add scatter plot
        fig.add_trace(go.Scatter(
            x=group[selected_feature],  # Use selected_feature instead of Price
            y=group['Audience Rating'],
            name=pace,
            mode='markers',
            marker=dict(
                size=10,
                symbol=pace_symbols[pace],
                color=pace_colors[pace],
                line=dict(width=1, color='white')
            ),
            text=group.apply(
                lambda row: f"{row['Brand']} - {row['Product Name']}" +
                            (f"<br>Release Date: {row['Release Date']}" if 'Release Date' in group.columns and pd.notna(row['Release Date']) else ""),
                axis=1
            ),
            hovertemplate=(
                "<b>%{text}</b><br>" +
                f"{selected_feature}: " + "%{x:.2f}<br>" +
                "Rating: %{y:.1f}<br>" +
                "<extra></extra>"
            )
        ))
        
        # Add trend line with error handling
        if len(group) > 1:
            try:
                valid_mask = ~np.isnan(group[selected_feature]) & ~np.isnan(group['Audience Rating'])
                if sum(valid_mask) > 1:
                    x = group[selected_feature][valid_mask]
                    y = group['Audience Rating'][valid_mask]
                    
                    if np.ptp(x) > 0 and np.ptp(y) > 0:
                        A = np.vstack([x, np.ones(len(x))]).T
                        slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
                        
                        x_range = np.linspace(x.min(), x.max(), 100)
                        y_range = slope * x_range + intercept
                        
                        fig.add_trace(go.Scatter(
                            x=x_range,
                            y=y_range,
                            mode='lines',
                            line=dict(color=pace_colors[pace], dash='dash', width=1),
                            showlegend=False,
                            hoverinfo='skip'
                        ))
            except Exception as e:
                st.warning(f"Could not compute trend line for {pace} category due to insufficient or invalid data.")
                continue
    
    # Highlight selected brand's products
    if selected_brand != "ALL" and selected_brand in full_plot_data['Brand'].values:
            brand_data = full_plot_data[full_plot_data['Brand'] == selected_brand]
            
            # Create a dictionary to map products to their pace symbols
            product_pace_symbols = {}
            for _, row in brand_data.iterrows():
                if row['Pace'] in pace_symbols:
                    product_pace_symbols[row['Product Name']] = pace_symbols[row['Pace']]
                else:
                    product_pace_symbols[row['Product Name']] = 'diamond'  # default symbol
            
            fig.add_trace(go.Scatter(
                x=brand_data[selected_feature],
                y=brand_data['Audience Rating'],
                mode='markers',
                marker=dict(
                    size=12,
                    symbol=[product_pace_symbols[name] for name in brand_data['Product Name']],  # Dynamic symbols
                    color=pace_colors.get(selected_brand, '#FFD700'),
                    line=dict(width=2, color='black')
                ),
                name=f"{selected_brand} Products",
                text=brand_data.apply(
                    lambda row: (
                        f"{row['Product Name']}<br>"
                        f"Brand: {row['Brand']}<br>"
                        f"{selected_feature}: {row[selected_feature]:.2f}<br>"
                        f"Performance: {row['Pace']}<br>"
                        f"Release Date: {row['Release Date'] if 'Release Date' in brand_data.columns else 'N/A'}"
                    ),
                    axis=1
                ),
                hovertemplate=(
                    "<b>%{text}</b><br>" +
                    "Rating: %{y:.1f}<br>" +
                    "<extra></extra>"
                )
            ))

    # Add average lines
    avg_feature = full_plot_data[selected_feature].mean()
    avg_rating = full_plot_data['Audience Rating'].mean()
    
    fig.add_hline(y=avg_rating, line_dash="dot", line_color="black", opacity=0.6)
    fig.add_vline(x=avg_feature, line_dash="dot", line_color="black", opacity=0.6)
    
    # Highlight selected product if any
    if chosen_product != "ALL" and chosen_product in full_plot_data['Product Name'].values:
        product_data = full_plot_data[full_plot_data['Product Name'] == chosen_product]
        if not product_data.empty:
            fig.add_trace(go.Scatter(
                x=product_data[selected_feature],
                y=product_data['Audience Rating'],
                mode='markers',
                marker=dict(
                    size=19,
                    symbol=pace_symbols.get(product_data.iloc[0]['Pace'], 'star'),  # Match the chosen_product's original pace icon
                    color='cyan',
                    line=dict(width=2, color='black')
                ),
                name=product_data.iloc[0]['Product Name'],  # Use the product name as the legend label
                text=product_data.apply(
                    lambda row: f"{row['Product Name']}<br>Brand: {row['Brand']}<br>{selected_feature}: {row[selected_feature]:.2f}<br>Performance: {row['Pace']}<br>Release Date: {row['Release Date'] if 'Release Date' in product_data.columns else 'N/A'}",
                    axis=1
                ),
                hovertemplate=(
                    "<b>%{text}</b><br>" +
                    "Rating: %{y:.1f}<br>" +
                    "<extra></extra>"
                )
            ))

    # Update layout
    fig.update_layout(
        xaxis_title=dict(text=selected_feature, font=dict(size=16, color='black')),
        yaxis_title=dict(text="Audience Rating", font=dict(size=16, color='black')),
        showlegend=True,
        legend_title=dict(text="Performance Category", font=dict(color='black')),
        legend=dict(font=dict(color='black')),
        hovermode='closest',
        template='plotly_white',
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor='white',
        plot_bgcolor='white',
        xaxis=dict(tickfont=dict(color='black')),
        yaxis=dict(tickfont=dict(color='black'))
    )
    
    # Add correlation coefficient
    corr = full_plot_data[selected_feature].corr(full_plot_data['Audience Rating'])
    fig.add_annotation(
        x=0.05,
        y=0.05,
        xref="paper",
        yref="paper",
        text=f"Correlation: {corr:.2f}",
        showarrow=False,
        bgcolor="white",
        bordercolor="gray",
        borderwidth=1,
        font=dict(size=12, color="black")
    )
    
    # Display the plot
    st.plotly_chart(fig, use_container_width=True)
# Plot 1: Feature Distribution by Pace and Rating
# Add debug prints to check data at each step
with row2_col1:
    st.markdown(f"### {selected_feature} Distribution by Performance Category")
    if selected_feature in plot_data.columns:
        # Add insights box
        st.markdown(f"""
        <div class='insight-box'>
        <b>Insight:</b> This visualization reveals how {selected_feature.lower()} varies across different 
        performance categories and rating classes. The box represents 75% of the data, while the line inside represents the median - hover over the boxes to get a deeper insight . 
        
  
        <b>Action :</b> Change the brand and selected feature in the sidebar to see how the distribution changes.
        </div>
        """, unsafe_allow_html=True)
        
        # Prepare data for Plotly boxplot
        filtered_data = plot_data.dropna(subset=[selected_feature, 'Pace', 'Rating Category'])
        
        # Create Plotly boxplot with custom hover template
        fig = px.box(
            filtered_data,
            x='Pace',
            y=selected_feature,
            color='Rating Category',
            category_orders={
                'Pace': ['Everyday', 'Fast', 'Race'],  # Ensure correct x-axis order
                'Rating Category': ['Low (<75)', 'Medium (75-90)', 'High (90+)']
            },
            color_discrete_map={
                'Low (<75)': '#FF5A5F',
                'Medium (75-90)': '#FFC107',
                'High (90+)': '#4CAF50'
            },
            points='outliers',  # Show only outlier points
            title=f"{selected_feature} Distribution by Pace and Rating Category",
            labels={
                'Pace': 'Performance Category',
                selected_feature: selected_feature,
                'Rating Category': 'Rating Class'
            },
            hover_data=['Brand', 'Product Name'],  # Add these fields to hover info
            custom_data=['Brand', 'Product Name']  # Include in custom data for hover template
        )

        # Update hover template for outlier points
        fig.update_traces(
            hovertemplate=(
                "<b>%{customdata[1]}</b><br>" +  # Product Name
                "Brand: %{customdata[0]}<br>" +   # Brand
                f"{selected_feature}: %{{y:.2f}}<br>" +  # Selected feature value
                "Performance: %{x}<br>" +         # Pace category
                "<extra></extra>"
            )
        )
        # Update layout
        fig.update_layout(
            boxmode='group',
            plot_bgcolor='white',
            paper_bgcolor='white',
            xaxis_title=dict(
            text="<b>Performance Category</b>",  # Make x-axis title bold
            font=dict(color='black', size=16)
            ),
            yaxis_title=dict(
            text=f"<b>{selected_feature}</b>",  # Make y-axis title bold
            font=dict(color='black', size=16)
            ),
            legend_title=dict(
            text="<b>Audience Rating</b>",  # Make legend title bold
            font=dict(color='black')
            ),
            legend=dict(
            font=dict(color='black')  # Ensure legend values are black
            ),
            template='plotly_white',
            margin=dict(l=20, r=20, t=40, b=20),
            xaxis=dict(
            showgrid=True,
            gridcolor='black',
            zeroline=False,
            linecolor='black',
            tickfont=dict(color='black', size=14),
            categoryorder='array',
            categoryarray=['Everyday', 'Fast', 'Race']
            ),
            yaxis=dict(
            showgrid=True,
            gridcolor='black',
            zeroline=False,
            linecolor='black',
            tickfont=dict(color='black', size=14)
            )
        )
        
        # Update traces to add black outlines to the boxes
        fig.update_traces(
            marker=dict(
                line=dict(
                    color='black',
                    width=1
                )
            )
        )
        
        # Display the plot
        st.plotly_chart(fig, use_container_width=True)

# Plot 2: Brand Performance Table
with row2_col2:
    st.markdown("### Brand Performance by Category")
    
    # Add insights box
    st.markdown("""
    <div class='insight-box'>
    <b>Insight:</b> This table highlights how different brands perform across pace categories, showing the 
    correlation between price point, audience satisfaction, and market presence. Brands targeting specific 
    niches often show distinct performance patterns.
    
    Action : Select a brand in the sidebar and see how they perform across various categories.
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

# New Time Series Analysis Plot
st.markdown("<h1 style='text-align: left;'>Feature Changes Over Time - Use Sidebar to Select Different Features and Track Their Changes</h2>", unsafe_allow_html=True)


# Create a new row for the time series plot
row3_col1, row3_col2 = st.columns([3, 1])
with row3_col1:
    st.markdown(f"### {selected_feature} Trends Over Time by Performance Category")
    
    # Add insights box
    st.markdown(f"""
    <div class='insight-box'>
    <b>Insight:</b> This visualization shows how {selected_feature.lower()} has evolved over time
    for different performance categories, highlighting how specifications vary across different use cases.
    Data points represent yearly averages within each category.
    </div>
    """, unsafe_allow_html=True)
    
    try:
        time_series_data['Year'] = pd.to_datetime(time_series_data['Original Review Date'], format='%b %d, %Y').dt.year
        running_data = time_series_data[time_series_data['Year'] >= 2021]
        
        # Create figure
        fig = go.Figure()
        
        # Color scheme for pace categories
        pace_colors = {
            'Everyday': '#3498db',
            'Fast': '#e74c3c',
            'Race': '#2ecc71'
        }
        
        # Process each pace category
        for pace in ['Everyday', 'Fast', 'Race']:
            # Filter data for this pace
            pace_data = running_data[running_data['Pace'] == pace]
            
            # Calculate yearly averages for the selected feature
            yearly_data = pace_data.groupby('Year')[selected_feature].agg(['mean', 'count', 'std']).reset_index()
            yearly_data = yearly_data[yearly_data['count'] > 0]  # Remove years with no data
            
            # Add main line
            fig.add_trace(go.Scatter(
                x=yearly_data['Year'],
                y=yearly_data['mean'],
                mode='lines+markers',
                name=pace,
                line=dict(color=pace_colors[pace], width=2),
                marker=dict(size=8),
                hovertemplate=(
                    f"<b>{pace}</b><br>" +
                    "Year: %{x}<br>" +
                    f"{selected_feature}: %{{y:.2f}}<br>" +
                    "Sample size: %{customdata[0]}<br>" +
                    "Std: %{customdata[1]:.2f}<br>" +
                    "<extra></extra>"
                ),
                customdata=yearly_data[['count', 'std']]
            ))
            
          
        
        # Update layout
        fig.update_layout(
            xaxis_title=dict(text="Year", font=dict(size=14)),
            yaxis_title=dict(text=selected_feature, font=dict(size=18)),
            template='plotly_white',
            hovermode='x unified',
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(l=20, r=20, t=40, b=20),
            xaxis=dict(
                showgrid=True,
                gridcolor='lightgray',
                dtick=1,  # Show every year
                tickangle=45
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor='lightgray',
                zeroline=False
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.warning("Unable to process time series data. Please check if the date format is consistent.")
        st.error(f"Error details: {str(e)}")

with row3_col2:
    # Add summary statistics
    if 'Year' in running_data.columns:
        st.markdown("### Quick Stats")

        # Calculate latest year stats by pace category
        latest_year = running_data['Year'].max()
        earliest_year = running_data['Year'].min()
        total_years = latest_year - earliest_year

        # Show data range
        st.markdown(f"""
        <div class='metric-container' style='background-color: #1e3d59; color: white;'>
            <div class='metric-label' style='color: white;'>Year Range</div>
            <div class='metric-value' style='color: white;'>{earliest_year} - {latest_year}</div>
        </div>
        """, unsafe_allow_html=True)

        # Calculate and show changes for each pace category
        for pace in ['Everyday', 'Fast', 'Race']:
            pace_data = running_data[running_data['Pace'] == pace]
            yearly_avgs = pace_data.groupby('Year')[selected_feature].mean()
            
            if not yearly_avgs.empty:
                latest_avg = yearly_avgs.get(latest_year, 0)
                earliest_avg = yearly_avgs.get(earliest_year, 0)
                
                if earliest_avg != 0:
                    pct_change = ((latest_avg - earliest_avg) / earliest_avg) * 100
                    
                    background_color = pace_colors[pace]
                    text_color = '#FFFFFF'  # White text for better contrast
                    st.markdown(f"""
                    <div class='metric-container' style='background-color: {background_color}; color: {text_color};'>
                        <div class='metric-label' style='color: {text_color};'>{pace} Change</div>
                        <div class='metric-value' style='color: {text_color};'>{pct_change:.1f}%</div>
                        <div class='metric-label' style='color: {text_color};'>({latest_avg:.1f} from {earliest_avg:.1f})</div>
                    </div>
                    """, unsafe_allow_html=True)


row4_col1, row4_col2 = st.columns([3, 1])


with row4_col1:
    # New Time Series Analysis Plot
    st.markdown("<h1 style='text-align: left;'>Custom Product - Technical Feature Comparison</h1>", unsafe_allow_html=True)

    st.markdown("### Product Technical Profile (US M9)")

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

        Note - These values are normalized to a scale of 0-1 for comparison purposes - Hover over the chart to see the original values for each feature.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class='insight-box'>
        <b>Action :</b> Select a specific product from the sidebar to see how it compares to market averages
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

    # User Input Form
    with st.expander("Input Your Own Product Features "):
        st.markdown("### Enter the values for your custom product: (US M9)")
        user_input = {}
        for feature in valid_features:
            user_input[feature] = st.number_input(
                f"{feature} (e.g., {running_data[feature].mean():.2f})",
                min_value=float(running_data[feature].min()),
                max_value=float(running_data[feature].max()),
                value=float(running_data[feature].mean())
            )

        # Submit button with custom style
        custom_button = st.markdown("""
            <style>
            div.stButton > button {
            background-color: white !important;
            color: black !important;
            border: 1px solid #ccc !important;
            border-radius: 5px !important;
            font-weight: bold !important;
            }
            div.stButton > button:hover {
            background-color: #f0f0f0 !important;
            color: black !important;
            }
            </style>
        """, unsafe_allow_html=True)
        if st.button("Add Custom Product"):
            # Normalize user input
            user_input_normalized = scaler.transform(pd.DataFrame([user_input]))
            user_normalized_values = user_input_normalized[0]

            # Add custom product to radar chart
            custom_product_name = "Custom Product"
            st.session_state['custom_product'] = {
            'name': custom_product_name,
            'values': user_normalized_values
            }

    # Prepare data for Plotly radar chart
    shoe_index = shoe.name if chosen_product != "ALL" else None
    shoe_normalized = normalized_data.loc[shoe_index].dropna() if shoe_index else None

    # Calculate averages for comparison
    normalized_data_without_selected = normalized_data.drop(index=shoe_index) if shoe_index else normalized_data
    average_normalized_values = normalized_data_without_selected.mean()

    # Create dataframe for plotting
    plot_df = pd.DataFrame({
        'Feature': shoe_normalized.index if shoe_index else valid_features,
        'Normalized Value': shoe_normalized.values if shoe_index else [0] * len(valid_features),
        'Average Value': average_normalized_values[valid_features].values
    })

# Prepare data for Plotly radar chart
categories = plot_df['Feature'].tolist()
values = plot_df['Normalized Value'].tolist()
values.append(values[0])  # Close the loop

# Create Plotly radar chart
fig = go.Figure()

# Add selected product trace
if shoe_index:
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Selected Product',
        line=dict(color='#1f77b4')
    ))

# Add average traces for the highest-rated products in each pace category
for pace in ['Everyday', 'Fast', 'Race']:
    pace_data = running_data[(running_data['Pace'] == pace) & (running_data['Audience Rating'] > 87.5)]
    if not pace_data.empty:
        # Calculate the average of the highest-rated products
        pace_averages = pace_data[valid_features].mean()
        normalized_pace_averages = scaler.transform([pace_averages])[0]
        pace_values = list(normalized_pace_averages) + [normalized_pace_averages[0]]  # Close the loop

        # Prepare hover text for original values
        hover_text = [f"{cat}: {avg:.2f}" for cat, avg in zip(categories, pace_averages)]

        # Add trace for the pace category
        fig.add_trace(go.Scatterpolar(
            r=pace_values,
            theta=categories + [categories[0]],  # Close the loop
            fill='none',
            name=f'{pace} (Highest Rated Avg)',
            line=dict(color=pace_colors[pace], dash='dot'),
            text=hover_text + [hover_text[0]],  # Close the loop for hover text
            hovertemplate=(
                "<b>%{theta}</b><br>" +
                "Normalized: %{r:.2f}<br>" +
                "Original: %{text}<br>" +
                "<extra></extra>"
            )
        ))

# Add custom product trace if available
# Add custom product trace if available
if 'custom_product' in st.session_state:
    custom_product = st.session_state['custom_product']
    custom_values = list(custom_product['values']) + [custom_product['values'][0]]  # Close the loop

    # Prepare hover text for custom product
    custom_hover_text = [
        f"{cat}: {orig:.2f}" for cat, orig in zip(categories, custom_product['values'])
    ] + [f"{categories[0]}: {custom_product['values'][0]:.2f}"]  # Close the loop for hover text

    # Add trace for the custom product
    fig.add_trace(go.Scatterpolar(
        r=custom_values,
        theta=categories + [categories[0]],  # Close the loop
        name=custom_product['name'],
        line=dict(color='orange'),
        text=custom_hover_text,  # Pass the hover text
        hovertemplate=(
            "<b>%{theta}</b><br>" +
            "Normalized: %{r:.2f}<br>" +
            "Original: %{text}<br>" +
            "<extra></extra>"
        )
    ))

# Update layout
fig.update_layout(
    polar=dict(
        radialaxis=dict(
            visible=True,
            range=[0, 1]
        )
    ),
    showlegend=True,
    legend=dict(
        title="Legend",
        orientation="h",
        yanchor="bottom",
        y=-0.2,
        xanchor="center",
        x=0.5
    ),
    margin=dict(l=20, r=20, t=20, b=20)
)

# Display the radar chart
st.plotly_chart(fig, use_container_width=True)

# Check if custom product exists before creating diff_df
if 'custom_product' in st.session_state:
    custom_product = st.session_state['custom_product']
    custom_values = custom_product['values']
    
    # Get original (unnormalized) user input
    user_input_original = pd.Series(user_input)

    # Compute dataset average (original values, not normalized)
    dataset_avg = running_data[valid_features].mean()

    # Create a DataFrame showing differences
    diff_df = pd.DataFrame({
        'Feature': valid_features,
        'Custom Value': user_input_original.values,
        'Market Average': dataset_avg.values,
    })

    diff_df['Absolute Diff'] = diff_df['Custom Value'] - diff_df['Market Average']
    diff_df['% Diff'] = (diff_df['Absolute Diff'] / diff_df['Market Average']) * 100

    st.markdown("### 📊 Difference from Market Average")
    st.dataframe(
        diff_df.set_index('Feature')[['Custom Value', 'Market Average', 'Absolute Diff', '% Diff']].style.format({
            'Custom Value': '{:.2f}',
            'Market Average': '{:.2f}',
            'Absolute Diff': '{:+.2f}',
            '% Diff': '{:+.1f}%'
        }),
        use_container_width=True
    )

    # Find the feature with the max percentage difference
    max_diff_feature = diff_df.loc[diff_df['% Diff'].abs().idxmax()]

    st.markdown(f"""
    <div class='insight-box'>
    <b>Highlight:</b> The custom product differs most in <b>{max_diff_feature['Feature']}</b> 
    with a <b>{max_diff_feature['% Diff']:+.1f}%</b> difference from the market average.
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class='insight-box'>
    <b>Note:</b> Please create a custom product to see how it compares to the market average.
    </div>
    """, unsafe_allow_html=True)
# Add insight box
st.markdown("""
<div class='insight-box'>
<b>Insight:</b> This radar chart compares the selected product and custom product against the 
highest-rated products (Audience Rating > 87.5) in each performance category (Everyday, Fast, Race). 
The dashed lines represent the average feature values of the highest-rated products in each category.
</div>
""", unsafe_allow_html=True)

# Footer with methodology note
st.markdown("""
---
**Methodology Note:** This analysis is based on technical measurements and audience ratings 
collected from RunRepeat.com. 
""")
