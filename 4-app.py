import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Load data
df = pd.read_csv('FootwearData_cleaned.csv')

# Layout
with st.sidebar:
    st.title("Running Industry Research")
    st.markdown("A look into recent trends in the running industry...")
    st.download_button("Download CSV", data=df.to_csv().encode(), file_name="running_research_data.csv")

# Main visuals
st.title("Initial Insights")

# Plot 1 Features vs Shoe Type
# Filter for running category
running_data = df[df['Category'] == 'Running'].copy()

# Clean and format data
running_data['Price'] = running_data['Price'].replace('[\$,]', '', regex=True).astype(float)

# Create Rating Category
running_data['Rating Category'] = pd.cut(
    running_data['Audience Rating'], 
    bins=[0, 80, 90, 100], 
    labels=['Low (<80)', 'Medium (80-90)', 'High (90+)']
)

# Make Pace ordered
running_data['Pace'] = pd.Categorical(
    running_data['Pace'], 
    categories=[
        'Daily running',
        # 'Daily running, Tempo', 
        'Tempo', 
        # 'Tempo, Competition', 
        'Competition'
    ], 
    ordered=True
)

# Sidebar controls
st.sidebar.header("Filter Options")
selected_brand = st.sidebar.selectbox("Select Brand", ["ALL"] + sorted(running_data['Brand'].unique()))
selected_feature = st.sidebar.selectbox("Select Y-axis Feature", sorted([
    # 'Breathability', doesn't show shit
    'Drop (mm)', 
    'Flexibility / Stiffness (average) (N)', 
    'Forefoot stack (mm)', 
    # 'Heel counter stiffness', 
    # 'Heel feel', 
    # 'Heel padding durability', 
    'Heel stack (mm)', 
    'Insole thickness (mm)', 
    # 'Lug depth (mm)', 
    # 'Midfoot feel', 
    # 'Midsole softness (HA)', 
    # 'Midsole softness in cold (%)', 
    # 'Midsole softness in cold (HA)', 
    'Midsole width - forefoot (mm)', 
    'Midsole width - heel (mm)', 
    # 'Outsole durability (mm)', 
    'Outsole hardness (HC)', 
    'Outsole thickness (mm)', 
    'Price', 
    # 'Reflective elements', 
    # 'Secondary foam softness (HA)', 
    'Stiffness in cold (%)', 
    'Stiffness in cold (N)', 
    # 'Torsional rigidity', 
    # 'Toe guard durability', 
    # 'Toebox durability',  #
    # 'Toebox feel', #
    # 'Toebox height (mm)', #
    # 'Toebox width - big toe (average) (mm)', #
    # 'Toebox width - widest part (average) (mm)', 
    'Tongue padding (mm)', 
    'Weight (g)', 
    'Weight (oz)'
]))

# Filter by brand if not "ALL"
if selected_brand != "ALL":
    plot_data = running_data[running_data['Brand'] == selected_brand]
else:
    plot_data = running_data

# Handle missing feature gracefully
if selected_feature not in plot_data.columns:
    st.error(f"Feature '{selected_feature}' not found in dataset.")
else:
    st.subheader(f"📊 {selected_feature} by Performance Category (Colored by Audience Rating Class)")

    # Prepare data for plotting
    grouped_data = plot_data.groupby(['Pace', 'Rating Category'])[selected_feature].apply(list).unstack()

    # Create the boxplot using Matplotlib
    fig, ax = plt.subplots(figsize=(12, 6))
    positions = []
    labels = []
    colors = ['red', 'orange', 'green']  # Colors for Low, Medium, High categories
    for i, (rating_category, color) in enumerate(zip(grouped_data.columns, colors)):
        for j, pace in enumerate(grouped_data.index):
            data = grouped_data.loc[pace, rating_category]
            if not (pd.isna(data) if isinstance(data, (float, int, str)) else pd.isna(data).any()):
                positions.append(j + i * 0.2)  # Offset positions for each category
                ax.boxplot(data, positions=[j + i * 0.2], widths=0.15, patch_artist=True, boxprops=dict(facecolor=color))
        labels.append(rating_category)

    # Set x-axis labels
    ax.set_xticks(range(len(grouped_data.index)))
    ax.set_xticklabels(grouped_data.index, rotation=45)
    ax.set_xlabel("Performance Category")
    ax.set_ylabel(selected_feature)
    ax.set_title(f"{selected_feature} by Performance Category (Colored by Audience Rating Class)")
    ax.legend(handles=[plt.Line2D([0], [0], color=color, lw=4) for color in colors], labels=labels, title="Rating Category")

    # Display the plot in Streamlit
    st.pyplot(fig)

# Plot 2: Brand and Product Analysis
# Features to choose from
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

# --- Feature list ---
available_features = sorted([
    'Price', 
    'Drop (mm)', 
    # 'Flexibility / Stiffness (average) (N)', 
    'Forefoot stack (mm)', 
    'Heel stack (mm)', 
    'Insole thickness (mm)', 
    'Midsole width - forefoot (mm)', 
    'Midsole width - heel (mm)', 
    # 'Outsole hardness (HC)', 
    'Outsole thickness (mm)', 
    # 'Stiffness in cold (%)', 
    # 'Stiffness in cold (N)', 
    'Toebox width - widest part (average) (mm)', 
    'Tongue padding (mm)', 
    'Weight (g)'
])

# --- Sidebar filters ---
st.sidebar.header("Shoe Profile Plot")
# Use the chosen brand from Plot 1
brand_filter = ["ALL"] + sorted(running_data['Brand'].dropna().unique())
chosen_brand = selected_brand  # Feed from Plot 1's selected brand

# Filter product names based on chosen brand
if chosen_brand != "ALL":
    product_filter = ["ALL"] + sorted(running_data[running_data['Brand'] == chosen_brand]['Product Name'].dropna().unique())
else:
    product_filter = ["ALL"] + sorted(running_data['Product Name'].dropna().unique())

chosen_product = st.sidebar.selectbox("Select Product", product_filter)

# pace_filter = ["ALL"] + sorted(running_data['Pace'].dropna().unique())
# chosen_pace = st.sidebar.selectbox("Select Pace", pace_filter)

# --- Filter data to 1 row ---
filtered = running_data.copy()
if chosen_brand != "ALL":
    filtered = filtered[filtered['Brand'] == chosen_brand]

if chosen_product != "ALL":
    filtered = filtered[filtered['Product Name'] == chosen_product]

if filtered.empty:
    st.warning("No data for selected filters.")
else:
    shoe = filtered.iloc[0]

    # Normalize across the dataset for all features
    scaler = MinMaxScaler()
    normalized_data = running_data[available_features].copy()
    normalized_data = pd.DataFrame(
        scaler.fit_transform(normalized_data),
        columns=available_features
    )

    # Get normalized values for selected shoe
    shoe_index = shoe.name  # index of the selected shoe
    shoe_normalized = normalized_data.loc[shoe_index].dropna()

    # Remove the selected product from the dataset for average calculation
    normalized_data_without_selected = normalized_data.drop(index=shoe_index)

    # Calculate the average normalized values for all other products
    average_normalized_values = normalized_data_without_selected.mean()

    # Calculate averages for each Rating Category
    rating_category_averages = {}
    for category in running_data['Rating Category'].unique():
        category_data = running_data[running_data['Rating Category'] == category]
        if not category_data.empty:
            category_normalized = pd.DataFrame(
                scaler.transform(category_data[available_features]),
                columns=available_features
            )
            rating_category_averages[category] = category_normalized.mean()

    # Build DataFrame for plotting
    plot_df = pd.DataFrame({
        'Feature': shoe_normalized.index,
        'Normalized Value': shoe_normalized.values,
        'Average Value': average_normalized_values[shoe_normalized.index].values
    })

    st.subheader(f"📊 Radar Chart for {shoe['Product Name']} (with Average Comparison)")

    # Create a grid layout for the markdowns
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            f"<div style='background:linear-gradient(to right, #6a11cb, #2575fc); padding:10px; border-radius:5px; color:white; text-align:center;'>"
            f"<h2>  {shoe['Product Name']}</h2>"
            f"</div>",
            unsafe_allow_html=True
        )

        st.markdown(
            f"<div style='background:linear-gradient(to right, #ff7e5f, #feb47b); padding:10px; border-radius:5px; color:white; text-align:center;'>"
            f"<h2>Price: ${shoe['Price']}</h2>"
            f"</div>",
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            f"<div style='background:linear-gradient(to right, #43cea2, #185a9d); padding:10px; border-radius:5px; color:white; text-align:center;'>"
            f"<h2>{shoe['Pace']}</h2>"
            f"</div>",
            unsafe_allow_html=True
        )

        # Display Audience Rating in a big number format with color coding and arrow
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

        st.markdown(
            f"<div style='background:{rating_color}; padding:10px; border-radius:5px; color:white; text-align:center;'>"
            f"<h2> Rating: {audience_rating} <span style='color:{diff_color};'>{arrow} ({rating_difference:+.0f})</span></h2>"
            f"</div>",
            unsafe_allow_html=True
        )

    # Prepare data for radar chart
    categories = plot_df['Feature']
    values = plot_df['Normalized Value'].values.tolist()
    values += values[:1]  # Close the radar chart loop

    average_values = plot_df['Average Value'].values.tolist()
    average_values += average_values[:1]  # Close the radar chart loop

    # Add data for each Rating Category
    category_values = {}
    for category, averages in rating_category_averages.items():
        category_values[category] = averages[plot_df['Feature']].values.tolist()
        category_values[category] += category_values[category][:1]  # Close the loop

    # Radar chart setup
    num_vars = len(categories)
    angles = [n / float(num_vars) * 2 * 3.14159 for n in range(num_vars)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    # Plot selected product
    ax.plot(angles, values, linewidth=2, linestyle='solid', label='Selected Product', color='dodgerblue')
    ax.fill(angles, values, color='dodgerblue', alpha=0.25)

    # Plot average values
    ax.plot(angles, average_values, linewidth=2, linestyle='dashed', label='Average (Others)', color='orange')

    # Plot each Rating Category
    for category, category_vals in category_values.items():
        ax.plot(angles, category_vals, linestyle='dotted', linewidth=1.5, label=f'Average ({category})')

    # Add labels and legend
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], color="gray", size=8)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10, rotation=45, ha='right')
    ax.set_title(f"Radar Chart for {shoe['Product Name']}", size=16, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

    st.pyplot(fig)




# Plot #3 


# Table of the top 10 Brand's (Count by # of appearances) and Group by Pace 
# Display Brand, Pace, Average Audience Rating, and Average Price, # of Products (Count)
st.subheader("Brands and Products Analysis")
top_brands = running_data.groupby(['Brand', 'Pace']).size().reset_index(name='Count')
top_brands = top_brands[top_brands['Count'] >= 1]['Brand'].unique().tolist()
top_brands_data = running_data[running_data['Brand'].isin(top_brands)]
top_brands_grouped = top_brands_data.groupby(['Brand', 'Pace']).agg({
    'Audience Rating': 'mean',
    'Price': 'mean',
    'Pace': 'count'  # Add count for each group
}).rename(columns={'Pace': 'Count'}).reset_index().rename(columns={'Pace': 'Category'})

top_brands_grouped['Price'] = top_brands_grouped['Price'].replace('[\$,]', '', regex=True).astype(float)
top_brands_grouped = top_brands_grouped.rename(columns={
    'Audience Rating': 'Avg. Audience Rating',
    'Price': 'Avg. Price'
})
top_brands_grouped = top_brands_grouped.sort_values(by='Count', ascending=False)
top_brands_grouped['Brand'] = top_brands_grouped['Brand'].str.title()

# Highlight the selected brand
def highlight_selected_brand(row):
    if row['Brand'] == selected_brand.title():
        return ['background-color: lightcoral'] * len(row)
    return [''] * len(row)
    # Remove rows where Count is 0
top_brands_grouped = top_brands_grouped[top_brands_grouped['Count'] > 0]
st.dataframe(
    top_brands_grouped.style.apply(highlight_selected_brand, axis=1),
    use_container_width=True
)

# Plot

