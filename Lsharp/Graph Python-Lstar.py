import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Define the dataset file path
folder = "Experiment Results"
result_file = "Experiment3 Lstar.csv"  # Replace with your dataset name
file_path = os.path.join(folder, result_file)
df = pd.read_csv(file_path)

# Sort data by complexity and number of states
df = df.sort_values(by=['complexity', 'number_of_states'])

# Create a unique x-axis position for each model
unique_models = df['model'].unique()
model_to_xpos = {model: i for i, model in enumerate(unique_models)}  # Map models to x-axis positions
df['x_pos'] = df['model'].map(model_to_xpos)  # Map x positions to models

# Add new columns for the required sums
df['learn_resets_plus_steps'] = df['learn_resets'] + df['learn_steps']
df['total_steps_and_resets'] = df['learn_resets'] + df['learn_steps'] + df['test_resets'] + df['test_steps']

# New colors and markers
colors = ["blue", "k", "green", "red", "m", "y"]  # k is black
markers = ["D", "p", "s", "o", ">", "<"]

# Function to create scatter plots with error bars
def create_scatter_with_error_bars(data, x_axis, y_axes, log_scale=False):
    """
    Generate scatter plots with error bars showing average, min, and max values.
    :param data: DataFrame containing the data.
    :param x_axis: Column name to use for the x-axis.
    :param y_axes: List of column names to use for the y-axes.
    :param log_scale: Boolean, whether to use a logarithmic scale for the y-axis.
    """
    # Set Seaborn style
    sns.set(style="whitegrid", font_scale=1.1)

    # Loop through each y-axis column to create individual plots
    for i, y_axis in enumerate(y_axes):
        # Group data by 'x_pos' to calculate average, min, and max for each y-axis
        grouped_data = data.groupby(x_axis).agg(
            avg=(y_axis, 'mean'),
            min=(y_axis, 'min'),
            max=(y_axis, 'max')
        ).reset_index()

        plt.figure(figsize=(14, 6))  # Adjust figure size

        # Scatter plot for each model with error bars
        plt.errorbar(grouped_data[x_axis], grouped_data['avg'], 
                     yerr=[grouped_data['avg'] - grouped_data['min'], 
                           grouped_data['max'] - grouped_data['avg']],
                     fmt='o', color='blue', ecolor='lightgray', elinewidth=1.5, capsize=3,
                     alpha=0.7)  # Error bars with gray color

        # Apply log scale if needed
        if log_scale:
            plt.yscale('log')

        # Remove gridlines
        plt.grid(False)

        # Customize x-axis: keep x-ticks empty
        plt.xticks(range(len(unique_models)), [''] * len(unique_models))  # No model names on the x-axis

        # Customize the plot
        plt.title(f"{y_axis.replace('_', ' ').title()} by Models")
        plt.xlabel("Models (ascending order of complexity and states)")
        plt.ylabel(y_axis.replace('_', ' ').title())

        plt.tight_layout()
        plt.savefig(f"{y_axis}_scatter_with_errorbars{'_log' if log_scale else ''}.png", dpi=300)
        plt.show()

# Define the new y-axes to plot
y_axes = ["learn_resets_plus_steps", "total_steps_and_resets", "time"]  # Updated list of y-axes

# Call the function for normal scale
# create_scatter_with_error_bars(df, 'x_pos', y_axes, log_scale=False)

# Call the function for log scale
create_scatter_with_error_bars(df, 'x_pos', y_axes, log_scale=True)
