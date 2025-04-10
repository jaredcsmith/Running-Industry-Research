import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('FootwearData_cleaned.csv')

# Layout
with st.sidebar:
    st.title("Running Industry Research")
    st.markdown("A look into recent trends in the running industry...")
    st.download_button("Download CSV", data=df.to_csv().encode(), file_name="running_research_data.csv")

# Main visuals
st.title("Initial Insights")

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
        'Daily running, Tempo', 
        'Tempo', 
        'Tempo, Competition', 
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
    'Insole', 
    'Insole thickness (mm)', 
    'Lug depth (mm)', 
    'Midfoot feel', 
    'Midsole softness (HA)', 
    'Midsole softness in cold (%)', 
    'Midsole softness in cold (HA)', 
    'Midsole width - forefoot (mm)', 
    'Midsole width - heel (mm)', 
    'Outsole durability (mm)', 
    'Outsole hardness (HC)', 
    'Outsole thickness (mm)', 
    'Price', 
    'Reflective elements', 
    'Secondary foam softness (HA)', 
    'Stiffness in cold (%)', 
    'Stiffness in cold (N)', 
    'Torsional rigidity', 
    'Toe guard durability', 
    'Toebox durability', 
    'Toebox feel', 
    'Toebox height (mm)', 
    'Toebox width - big toe (average) (mm)', 
    'Toebox width - widest part (average) (mm)', 
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
    st.subheader(f"📊 {selected_feature} by Pace (Colored by Audience Rating Category)")

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
    ax.set_xlabel("Pace")
    ax.set_ylabel(selected_feature)
    ax.set_title(f"{selected_feature} by Pace (Colored by Audience Rating Category)")
    ax.legend(handles=[plt.Line2D([0], [0], color=color, lw=4) for color in colors], labels=labels, title="Rating Category")

    # Display the plot in Streamlit
    st.pyplot(fig)