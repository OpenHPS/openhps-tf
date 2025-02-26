import json
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.interpolate import griddata
import os
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import LinearSegmentedColormap
# Define a single gradient color palette for accuracy in metres, darker green without white
colors = ["#e5f5e0", "#c7e9c0", "#a1d99b", "#74c476", "#41ab5d", "#238b45", "#006d2c", "#00441b"]
cmap = LinearSegmentedColormap.from_list("accuracy_palette", colors)
def plot_heatmaps(interpolation_method='cubic', interpolate=True):
    # Load the data from the JSON file
    with open('accuracy_results.json', 'r') as f:
        data = json.load(f)

    # Group the data by the 'k' value
    grouped_data = defaultdict(list)
    for entry in data:
        grouped_data[entry['k']].append(entry)

    # Create a 2D grid for the heatmap
    x_max, y_max = 46.275, 37.27
    grid_x, grid_y = np.mgrid[0:y_max, 0:x_max]

    # Load the overlay image
    dir_path = os.path.dirname(os.path.realpath(__file__))
    img = plt.imread(os.path.join(dir_path, "pleinlaan9.png"))

    for k, entries in grouped_data.items():
        points = []
        values = []
        
        # Collect the points and their corresponding accuracy values
        for entry in entries:
            x, y, accuracy = int(entry['x']), int(entry['y']), max(0, entry['accuracy'])  # Normalize accuracy to be >= 0
            if 0 <= x < x_max and 0 <= y < y_max:
                points.append((y, x))
                values.append(accuracy)
        
        # Interpolate the values to create a full heatmap if interpolation is enabled
        if interpolate:
            points = np.array(points)
            values = np.array(values)
            heatmap = griddata(points, values, (grid_x, grid_y), method=interpolation_method, fill_value=np.nan)
            
            # Ensure interpolated values are non-negative
            heatmap = np.maximum(heatmap, 0)
   
            # Create a mask for NaN values
            # Extrapolate the heatmap to the edges
            if interpolation_method in ['linear', 'nearest']:
                heatmap = griddata(points, values, (grid_x, grid_y), method=interpolation_method, fill_value=0)
            else:
                heatmap = griddata(points, values, (grid_x, grid_y), method=interpolation_method, fill_value=np.nan)
                mask = np.isnan(heatmap)
                heatmap[mask] = griddata(points, values, (grid_x[mask], grid_y[mask]), method='nearest')
            mask = np.isnan(heatmap)
        else:
            heatmap = np.full(grid_x.shape, np.nan)
            for (y, x), value in zip(points, values):
                heatmap[y, x] = value
            mask = np.isnan(heatmap)

        # Plot the heatmap
        fig = plt.figure(dpi=300)
        ax = fig.add_subplot(111)
        ax.set_xlabel("X Position (metres)", fontsize=12)
        ax.set_ylabel("Y Position (metres)", fontsize=12)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=10))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=10))
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)
        ax.set_xlim(0, x_max)
        ax.set_ylim(0, y_max)
        
        sns.heatmap(heatmap, mask=mask, cmap=cmap, cbar=False, vmin=0, vmax=10, ax=ax)
        ax.invert_yaxis()  # Invert the y-axis to match the image orientation
        # Display the image on top of the heatmap
        ax.imshow(np.flipud(img), extent=[0, x_max, y_max, 0], alpha=1, interpolation='none', origin='upper', zorder=10)  # Adjust extent, origin, and zorder
        ax.spines["left"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        
        # Save the plot to an SVG file
        plt.savefig(f'accuracy_heatmap_k{k}_{interpolation_method}.svg', format='svg', bbox_inches='tight')
        # Save the plot to a PDF file
        plt.savefig(f'accuracy_heatmap_k{k}_{interpolation_method}.pdf', format='pdf', bbox_inches='tight')
        plt.close()

    # Create and save the legend
    create_legend(cmap)

def create_legend(cmap):
    # Horizontal legend
    fig, ax = plt.subplots(figsize=(4, 2), dpi=300)
    norm = plt.Normalize(vmin=0, vmax=10)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation='horizontal')
    cbar.set_label('Accuracy')
    cbar.set_ticks([0, 3, 10])
    cbar.set_ticklabels(['0', '3', '10'])
    ax.remove()
    plt.savefig('accuracy_heatmap_legend_horizontal.svg', format='svg', bbox_inches='tight')
    plt.savefig('accuracy_heatmap_legend_horizontal.pdf', format='pdf', bbox_inches='tight')
    plt.close()

    # Vertical legend
    fig, ax = plt.subplots(figsize=(2, 4), dpi=300)
    norm = plt.Normalize(vmin=0, vmax=10)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation='vertical')
    cbar.set_label('Accuracy')
    cbar.set_ticks([0, 3, 10])
    cbar.set_ticklabels(['0', '3', '10'])
    ax.remove()
    plt.savefig('accuracy_heatmap_legend_vertical.svg', format='svg', bbox_inches='tight')
    plt.savefig('accuracy_heatmap_legend_vertical.pdf', format='pdf', bbox_inches='tight')
    plt.close()

# Example usage
plot_heatmaps(interpolation_method='cubic', interpolate=True)  # For interpolated values
