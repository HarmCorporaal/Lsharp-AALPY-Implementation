import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

folder = "Experiment Results"
result_file = "Experiment1 Perfect.csv"
file_path = os.path.join(folder, result_file)
df = pd.read_csv(file_path)

# Combine extension and separation rules into a single column for grouping
df['Group'] = df['extension_rule'] + ", " + df['seperation_rule']

# Sort data by complexity and number of states
df = df.sort_values(by=['complexity', 'number_of_states'])

# Create a unique x-axis position for each model
unique_models = df['model'].unique()
model_to_xpos = {model: i for i, model in enumerate(unique_models)}  # Map models to x-axis positions
df['x_pos'] = df['model'].map(model_to_xpos)  # Map x positions to models

# New colors and markers as requested
colors = ["blue", "k", "green", "red", "m", "y"]  # k is black
ecolors = ["lightblue", "lightgray", "lightgreen", "pink", "lightgray", "lightgray"]
markers = ["D", "p", "s", "o", ">", "<"]

# Function to create scatter plots with average, min, and max values
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
        # Group data by 'Group' and 'x_pos' to calculate average, min, and max for each y-axis
        grouped_data = data.groupby(['Group', x_axis]).agg(
            avg=(y_axis, 'mean'),
            min=(y_axis, 'min'),
            max=(y_axis, 'max')
        ).reset_index()

        plt.figure(figsize=(14, 6))  # Adjust figure size

        # Loop through each group for plotting
        for j, group in enumerate(grouped_data['Group'].unique()):
            group_data = grouped_data[grouped_data['Group'] == group]

            # Scatter plot: average values with new colors and markers
            plt.scatter(group_data[x_axis], group_data['avg'], 
                        label=group, 
                        color=colors[j], 
                        marker=markers[j],
                        s=100, edgecolor='k', alpha=0.7)  # Increase size and add transparency

            # Add error bars (min and max) with new error bar colors
            plt.errorbar(group_data[x_axis], group_data['avg'],
                         yerr=[group_data['avg'] - group_data['min'], 
                               group_data['max'] - group_data['avg']],
                         fmt='none', ecolor=ecolors[j], elinewidth=1.5, capsize=3)

        # Apply log scale if needed
        if log_scale:
            plt.yscale('log')

        # Remove gridlines
        plt.grid(False)

        # Customize x-axis: keep x-ticks empty
        plt.xticks(range(len(unique_models)), [''] * len(unique_models))  # No model names on the x-axis

        # Customize the plot
        plt.title(f"{y_axis.replace('_', ' ').title()} by Extension and Separation Rules")
        plt.xlabel("Models (ascending order of complexity and states)")
        plt.ylabel(y_axis.replace('_', ' ').title())

        # Move legend inside the graph at the upper-left corner
        plt.legend(title="Groups", loc="upper left", bbox_to_anchor=(0.0, 1.0))  # Adjust bbox_to_anchor if needed

        plt.tight_layout()
        plt.savefig(f"{y_axis}_scatter_with_errorbars{'_log' if log_scale else ''}.png", dpi=300)
        plt.show()

# Define the y-axes to plot
y_axes = ["learn_steps", "test_steps", "test_resets", "time"]  # List of y-axes

# Call the function for normal scale
create_scatter_with_error_bars(df, 'x_pos', y_axes, log_scale=False)

# Call the function for log scale
create_scatter_with_error_bars(df, 'x_pos', y_axes, log_scale=True)
