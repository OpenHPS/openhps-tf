import json
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.interpolate import griddata
import os

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
            mask = np.isnan(heatmap)
        else:
            heatmap = np.full(grid_x.shape, np.nan)
            for (y, x), value in zip(points, values):
                heatmap[y, x] = value
            mask = np.isnan(heatmap)

        # Plot the heatmap
        fig = plt.figure(dpi=300)
        ax = fig.add_subplot(111)
        ax.set_xlabel("X Position (metres)", fontsize=7)
        ax.set_ylabel("Y Position (metres)", fontsize=7)
        plt.xticks(fontsize=6)
        plt.yticks(fontsize=6)
        ax.set_xlim(0, x_max)
        ax.set_ylim(0, y_max)
        
        cmap = sns.color_palette("RdYlGn_r", as_cmap=True)
        sns.heatmap(heatmap, mask=mask, cmap=cmap, cbar_kws={'label': 'Accuracy'}, vmin=0, vmax=8, ax=ax)
        ax.collections[0].colorbar.set_ticks([0, 5, 8])
        ax.collections[0].colorbar.set_ticklabels(['0', '5', '8'])
        ax.invert_yaxis()  # Invert the y-axis to match the image orientation
        # Display the image on top of the heatmap
        ax.imshow(img, extent=[0, x_max, y_max, 0], alpha=0.5, interpolation='none', origin='upper')  # Adjust extent and origin
        
        ax.spines["left"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        
        plt.title(f'Accuracy Heatmap for k={k} (Method: {interpolation_method})')

        # Save the plot to an SVG file
        plt.savefig(f'accuracy_heatmap_k{k}_{interpolation_method}.svg', format='svg')
        plt.close()

# Example usage
plot_heatmaps(interpolation_method='cubic', interpolate=True)  # For interpolated values
