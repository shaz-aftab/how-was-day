import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from minisom import MiniSom

# Set random seed for reproducibility
np.random.seed(42)

# Load and preprocess data
rd = pd.read_csv('MOCK_DATA.csv')
rd = rd.drop(['firstThing', 'weather', 'goodBad', 'mood'], axis=1)

def time_to_minutes(time_str):
    try:
        parts = time_str.strip().split(':')
        if len(parts) != 2:
            return None  # Skip invalid values
        hours = int(parts[0])
        minutes = int(parts[1])
        return hours * 60 + minutes
    except (ValueError, AttributeError):
        return None

rd['time'] = rd['time'].apply(time_to_minutes)

# Store original min and max values for de-normalization
time_min, time_max = rd['time'].min(), rd['time'].max()
sleep_min, sleep_max = rd['sleepHours'].min(), rd['sleepHours'].max()

# Normalize data
rd['time'] = (rd['time'] - time_min) / (time_max - time_min)
rd['sleepHours'] = (rd['sleepHours'] - sleep_min) / (sleep_max - sleep_min)

som_size = (3, 3)

# Initialize and train SOM
som = MiniSom(som_size[0], som_size[1], rd.shape[1], sigma=1.0, learning_rate=0.5, random_seed=42)
som.random_weights_init(rd.values)
som.train_random(rd.values, 1000)

def minutes_to_time(minutes):
    hours = int(minutes // 60)
    mins = int(minutes % 60)
    return f"{hours:02d}:{mins:02d}"

# Compute BMU (Best Matching Unit) locations for all data points
mapped = np.array([som.winner(d) for d in rd.values])
density_map = Counter(tuple(x) for x in mapped)

# Function to visualize the SOM
def plot():    
    # Create a 2D array for the heatmap
    heatmap = np.zeros(som_size)  
    
    # Fill the heatmap with cluster densities
    for (x, y), count in density_map.items():
        heatmap[x, y] = count
    
    # Plot the SOM with readable labels
    plt.figure(figsize=(8, 8))
    plt.imshow(np.zeros(som_size).T, cmap="Blues", origin="lower", alpha=0.3)  # Light background for clarity
    
    for x in range(som_size[0]):
        for y in range(som_size[1]):
            cluster = next((c for c in clusters if c[0] == f"({x},{y})"), None)
            if cluster:
                label = (
                    f"Wake: {cluster[1]}\n"
                    f"Sleep: {cluster[2]} hrs\n"
                    f"Count: {cluster[3]}"
                )
            else:
                label = "No Data"
            
            plt.text(x, y, label, ha="center", va="center", fontsize=10, color="black", bbox=dict(facecolor="white", alpha=0.6, edgecolor="black"))
    
    plt.imshow(heatmap.T, cmap="Blues", origin="lower", alpha=0.6)  # Heatmap with transparency
    
    # Scatter plot for clusters
    for (x, y), count in density_map.items():
        plt.scatter(x, y, s=200 + count * 20, c="lightblue", edgecolors="black")
    
    plt.xticks(range(som_size[0]))
    plt.yticks(range(som_size[1]))
    plt.grid(True, color="gray", linestyle="--", linewidth=0.5)
    plt.title("Self-Organizing Map (SOM) - Human Readable Cluster Map")
    plt.colorbar(label="Number of Data Points")
    plt.show()


# Store cluster information
clusters = []
for i in range(som_size[0]):
    for j in range(som_size[1]):
        cluster_data = rd.values[[som.winner(x) == (i, j) for x in rd.values]]
        if len(cluster_data) > 0:
            avg_wakeup_time = np.mean(cluster_data[:, 0]) * (time_max - time_min) + time_min
            avg_sleep_hours = np.mean(cluster_data[:, 1]) * (sleep_max - sleep_min) + sleep_min
            clusters.append([
                f"({i},{j})",  # Cluster position
                minutes_to_time(avg_wakeup_time),  # Convert to HH:MM
                f"{avg_sleep_hours:.2f}",  # Sleep hours rounded
                len(cluster_data)  # Number of data points
            ])

# Convert cluster data to a DataFrame
df_clusters = pd.DataFrame(clusters, columns=["Cluster", "Wake-up Time", "Sleep Hours", "Data Points"])

# **Print Table Properly**
print(df_clusters.to_string(index=False))

# **Save to CSV Properly**
df_clusters.to_csv('cluster_data.csv', index=False)

plot()