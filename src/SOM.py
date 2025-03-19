import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from minisom import MiniSom
from sklearn.preprocessing import LabelEncoder

# Set random seed for reproducibility
np.random.seed(42)

# Load and preprocess data
rd = pd.read_csv('MOCK_DATA.csv')
#rd = rd.drop(['firstThing', 'weather', 'goodBad', 'mood'], axis=1)

#convert the data to integer format
categorical_columns = ['firstThing', 'weather', 'goodBad', 'mood']

label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    rd[col] = le.fit_transform(rd[col])  # Convert text to numbers
    label_encoders[col] = le  # Store encoder for future decoding
    

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

class Predict:
    def __init__(self, som, time_min, time_max, sleep_min, sleep_max, df_original, df_clusters, time_to_minutes):
        self.som = som
        self.time_min = time_min
        self.time_max = time_max
        self.sleep_min = sleep_min
        self.sleep_max = sleep_max
        self.df_original = df_original  # Full dataset with all features
        self.df_clusters = df_clusters  # Clustered dataset
        self.time_to_minutes = time_to_minutes
        self.minutes_to_time = minutes_to_time

    def normalize_input(self, input_data):
        """Normalize input values before passing to SOM."""
        if 'time' in input_data:
            input_data['time'] = (self.time_to_minutes(input_data['time']) - self.time_min) / (self.time_max - self.time_min)
        if 'sleepHours' in input_data:
            input_data['sleepHours'] = (input_data['sleepHours'] - self.sleep_min) / (self.sleep_max - self.sleep_min)
    
        # Ensure numerical format
        numeric_input = [input_data.get(col, np.nan) for col in self.df_original.columns]
        return np.array(numeric_input, dtype=float)  # Convert everything to float


    def predict_cluster(self, input_data):
        """Find the BMU (Best Matching Unit) for given input data."""
        norm_data = self.normalize_input(input_data)
        bmu = self.som.winner(norm_data)  # Uses the trained SOM to find the closest neuron
        return bmu

    def predict(self, **kwargs):
        """Predict missing information based on given inputs (flexible input system)."""
    
        # Encode categorical values before passing to SOM
        for col in categorical_columns:
            if col in kwargs:
                kwargs[col] = label_encoders[col].transform([kwargs[col]])[0]  # Convert to numeric
    
        bmu = self.predict_cluster(kwargs)
        cluster_data = self.df_original.values[[self.som.winner(x) == bmu for x in self.df_original.values]]
    
        if len(cluster_data) == 0:
            return "No matching data found"
    
        # Compute average values for missing fields
        predicted_data = {}
        for i, col in enumerate(self.df_original.columns):
            if col not in kwargs:  # Predict only missing fields
                avg_value = np.nanmean(cluster_data[:, i])
    
                # Convert back to readable values
                if col == 'time':
                    predicted_data[col] = self.minutes_to_time(avg_value * (self.time_max - self.time_min) + self.time_min)
                elif col == 'sleepHours':
                    predicted_data[col] = round(avg_value * (self.sleep_max - self.sleep_min) + self.sleep_min, 2)
                elif col in categorical_columns:  # Convert numerical category back to label
                    predicted_data[col] = label_encoders[col].inverse_transform([int(round(avg_value))])[0]
                else:
                    predicted_data[col] = avg_value  # Direct numerical/text value
    
        return predicted_data


predictor = Predict(som, time_min, time_max, sleep_min, sleep_max, rd, df_clusters, time_to_minutes)
print(f"\n{predictor.predict(sleepHours=9, firstThing='Meditate')}")
#plot()