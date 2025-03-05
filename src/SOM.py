import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from minisom import MiniSom

# Set random seed for reproducibility
np.random.seed(42)

# Load and preprocess data
rd = pd.read_csv('MOCK_DATA.csv')  # RD meaning raw_data
rd = rd.drop(['firstThing', 'weather', 'goodBad', 'mood'], axis=1)

def time_to_minutes(time_str):
    try:
        parts = time_str.strip().split(':')
        if len(parts) != 2:
            return None  # Skip invalid values
        hours = int(parts[0])  # Extract hours
        minutes = int(parts[1])  # Extract minutes
        return hours * 60 + minutes  # Convert to total minutes
    except (ValueError, AttributeError):
        return None  # Skip non-convertible values

rd['time'] = rd['time'].apply(time_to_minutes)

# Store original min and max values for de-normalization
time_min = rd['time'].min()
time_max = rd['time'].max()
sleepHours_min = rd['sleepHours'].min()
sleepHours_max = rd['sleepHours'].max()

# Normalization of data (putting into a 0-1 scale)
rd['sleepHours'] = (rd['sleepHours'] - sleepHours_min) / (sleepHours_max - sleepHours_min)
rd['time'] = (rd['time'] - time_min) / (time_max - time_min)

som_size = (3, 3)

# Initialize SOM with a specific random seed
som = MiniSom(som_size[0], som_size[1],  # Defines 3x3 Grid
              rd.shape[1],  # Number of features (wake-up time & sleep hours)
              sigma=1.0,  # Controls neighborhood influence
              learning_rate=0.5,  # Controls learning speed
              random_seed=42  # Set a specific random seed for reproducibility
             )

som.random_weights_init(rd.values)  # Initialize weights
som.train_random(rd.values, 1000)  # Train the SOM

# Function to convert minutes to HH:MM format
def minutes_to_time(minutes):
    """
    Convert minutes from midnight to HH:MM format.
    """
    hours = int(minutes // 60)  # Get the hours
    mins = int(minutes % 60)    # Get the minutes
    return f"{hours:02d}:{mins:02d}"  # Format as HH:MM

# Print human-readable version of each cluster with de-normalized values
for i in range(som_size[0]):
    for j in range(som_size[1]):
        cluster_data = rd.values[[som.winner(x) == (i, j) for x in rd.values]]
        if len(cluster_data) > 0:
            # De-normalize wake-up time and sleep hours
            avg_wakeup_time = np.mean(cluster_data[:, 0]) * (time_max - time_min) + time_min
            avg_sleep_hours = np.mean(cluster_data[:, 1]) * (sleepHours_max - sleepHours_min) + sleepHours_min
            
            # Convert average wake-up time to HH:MM format
            avg_wakeup_time_hhmm = minutes_to_time(avg_wakeup_time)
            
            print(f"Cluster ({i}, {j}):")
            print(f"  Average Wake-up Time: {avg_wakeup_time_hhmm}")
            print(f"  Average Sleep Hours: {avg_sleep_hours:.2f}")
            print(f"  Number of Data Points: {len(cluster_data)}")
            print()

# Plot the SOM
plt.figure(figsize=(8, 8))
for x in range(som_size[0]):
    for y in range(som_size[1]):  # Nested FOR loop drawing the SOM's grid cells
        plt.scatter(x, y,  # For each point a scatter plot is drawn
                    marker='s',  # Each point's represented as a square
                    color='lightgray',  # With a light grey colour
                    s=200  # The size of each square is 200
                   )

mapped = np.array([som.winner(d) for d in rd.values])  # Find the best matching unit (BMU) for each data point
plt.scatter(mapped[:, 0], mapped[:, 1], c='blue', marker='o', label="Data Points")
plt.title("Self-Organizing Map (SOM) - Ideal Wake-up Time Clusters")
plt.show()