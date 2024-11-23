import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import streamlit as st

# Load the dataset (Replace with your dataset path)
@st.cache
def load_data():
    dataset = pd.read_csv('FoodFix/food_dataset.csv')  # Replace with your dataset file name
    return dataset

dataset = load_data()

# List of columns
numeric_columns = ['Sugar_Content (g)', 'Calories (kcal)', 'Protein (g)', 'Fiber (g)', 'Carbohydrates (g)', 'Glycemic_Index', 'Average_Weight (g)']
categorical_columns = ['Category', 'Ripeness', 'Size', 'Fruit']
nutrition_columns = ['Calories (kcal)', 'Protein (g)', 'Fiber (g)', 'Carbohydrates (g)', 'Sugar_Content (g)']

# Function to plot a simulated 3D Pie Chart
def plot_3d_pie_chart(total_nutrition, nutrition_columns):
    try:
        # Filter out the 'Sugar_Content (g)' from columns
        labels = [col for col in nutrition_columns if col != 'Sugar_Content (g)']
        sizes = [total_nutrition[i] for i, col in enumerate(nutrition_columns) if col != 'Sugar_Content (g)']
        explode = [0.1] * len(sizes)  # Add some separation for better visualization
        colors = plt.cm.viridis(np.linspace(0, 1, len(sizes)))

        # Simulating 3D effect by shadow and adjustments
        fig, ax = plt.subplots(figsize=(8, 6))
        wedges, texts, autotexts = ax.pie(
            sizes,
            labels=labels,
            explode=explode,
            autopct='%1.1f%%',
            shadow=True,
            startangle=140,
            colors=colors
        )

        # Styling
        for text in texts:
            text.set_color('grey')
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')

        plt.title("Simulated 3D Pie Chart of Predicted Nutrition", fontsize=14)
        st.pyplot(fig)  # Use Streamlit to display the plot
    except Exception as e:
        st.error(f"Error in plotting simulated 3D pie chart: {e}")

# Main Function for Prediction and Recommendation
def predict_and_recommend():
    st.title("Nutrition Prediction and Food Recommendations")

    try:
        # Load the trained model
        model = joblib.load('FoodFix/best_nutrition_model.pkl')

        # Predefined glycemic index mapping based on sweetness, size, and ripeness
        glycemic_index_map = {
            "Sweetness": {"Low": 20, "Medium": 40, "High": 60},
            "Size": {"Small": 5, "Medium": 10, "Large": 15},
            "Ripeness": {"Unripe": -10, "Ripe": 0, "Overripe": 10},
        }

        # Inputs from the user
        sugar_limit = st.number_input("Enter your maximum sugar intake level (in grams):", min_value=0.0, step=0.1)
        food_items_input = st.text_input("Enter the food items (comma-separated):")
        ripeness_input = st.text_input("Enter the ripeness levels (comma-separated):")
        size_input = st.text_input("Enter the sizes (comma-separated):")
        sweetness_input = st.text_input("Enter the sweetness levels (comma-separated - Low, Medium, High):")

        if st.button("Predict and Recommend"):
            if not all([food_items_input, ripeness_input, size_input, sweetness_input]):
                st.error("All inputs must be provided!")
                return

            # Prepare inputs for processing
            food_items = [item.strip().capitalize() for item in food_items_input.split(",")]
            ripeness = [rip.strip().capitalize() for rip in ripeness_input.split(",")]
            sizes = [size.strip().capitalize() for size in size_input.split(",")]
            sweetness_levels = [sweet.strip().capitalize() for sweet in sweetness_input.split(",")]

            # Ensure inputs have the same length
            if not (len(food_items) == len(ripeness) == len(sizes) == len(sweetness_levels)):
                st.error("All inputs must have the same number of items!")
                return

            # Calculate glycemic index for each food item
            glycemic_indices = []
            for size, ripeness_level, sweetness in zip(sizes, ripeness, sweetness_levels):
                if size not in glycemic_index_map["Size"] or ripeness_level not in glycemic_index_map["Ripeness"] or sweetness not in glycemic_index_map["Sweetness"]:
                    st.error("Invalid input for size, ripeness, or sweetness.")
                    return
                gi = (
                    glycemic_index_map["Size"][size]
                    + glycemic_index_map["Ripeness"][ripeness_level]
                    + glycemic_index_map["Sweetness"][sweetness]
                )
                glycemic_indices.append(max(0, gi))  # Ensure GI is non-negative

            # Prepare data for prediction
            rows = []
            for food_item, ripeness_level, size, glycemic_index in zip(food_items, ripeness, sizes, glycemic_indices):
                matching_row = dataset[dataset['Food_Item'] == food_item]
                if matching_row.empty:
                    st.error(f"Food item '{food_item}' not found in the dataset.")
                    return
                row = matching_row.iloc[0].copy()
                row['Ripeness'] = ripeness_level
                row['Size'] = size
                row['Glycemic_Index'] = glycemic_index
                rows.append(row)

            input_data = pd.DataFrame(rows)[numeric_columns + categorical_columns]

            # Predict Nutrition
            prediction = model.predict(input_data)
            total_nutrition = prediction.sum(axis=0)

            # Calculate total sugar content from the dataset
            selected_data = []
            for food_item, ripeness_level, size, glycemic_index in zip(food_items, ripeness, sizes, glycemic_indices):
                matching_row = dataset[dataset['Food_Item'] == food_item]
                if not matching_row.empty:
                    row = matching_row.iloc[0].copy()
                    row['Adjusted_Sugar_Content'] = row['Sugar_Content (g)']
                    selected_data.append(row)

            if not selected_data:
                st.error("None of the entered items match the dataset. Please check your input.")
                return

            selected_data = pd.DataFrame(selected_data)
            total_sugar = selected_data['Adjusted_Sugar_Content'].sum()

            # Reordered sequence based on your request
            st.subheader(f"Total Sugar Content in Selected Items: {total_sugar:.2f}g")
            st.subheader("Predicted Total Nutrition:")
            for i, column in enumerate(nutrition_columns):
                if column != 'Sugar_Content (g)':
                    st.write(f"{column}: {total_nutrition[i]:.2f}")

            if total_sugar <= sugar_limit:
                st.success(f"Your selected items are within your sugar limit of {sugar_limit}g. No replacements needed.")
                plot_3d_pie_chart(total_nutrition, nutrition_columns)
                return

            st.warning(f"Your selected items exceed your sugar limit of {sugar_limit}g. Suggesting replacements...")

            # Replacement logic
            replacements = []
            current_sugar = total_sugar

            for _, item in selected_data.iterrows():
                if current_sugar <= sugar_limit:
                    break
                lower_sugar_options = dataset[
                    (dataset['Sugar_Content (g)'] < item['Adjusted_Sugar_Content']) &
                    (dataset['Category'] == item['Category'])
                ].sort_values(by='Sugar_Content (g)').head(3)

                if not lower_sugar_options.empty:
                    replacement = lower_sugar_options.iloc[0]
                    replacements.append((item['Food_Item'], replacement['Food_Item']))
                    current_sugar -= (item['Adjusted_Sugar_Content'] - replacement['Sugar_Content (g)'])
                else:
                    replacements.append((item['Food_Item'], "No suitable replacement found"))

            if current_sugar > sugar_limit:
                st.error("Even after replacing some items, the total sugar content still exceeds the limit.")

            st.subheader("Recommendations:")
            for original, replacement in replacements:
                st.write(f"Replace '{original}' with '{replacement}'.")

            # Plot 3D Pie Chart
            plot_3d_pie_chart(total_nutrition, nutrition_columns)

    except Exception as e:
        st.error(f"Error: {e}")

# Run the prediction and recommendation
predict_and_recommend()
