import pandas as pd
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("MOCK_DATA.csv") # reading the CSV file

def time_to_minutes(time_str):
    try:
        # Ensure time is in 'HH:MM' format (handles cases like '5:30' -> '05:30')
        parts = time_str.strip().split(":")
        if len(parts) != 2:
            return None  # Skip invalid values
        
        hours = int(parts[0])  # Extract hours
        minutes = int(parts[1])  # Extract minutes
        
        return hours * 60 + minutes  # Convert to total minutes
    except (ValueError, AttributeError):
        return None  # Skip non-convertible values
    
# Converting non-integer attributes into integers

label_encoder = LabelEncoder()

# Convert time to minutes efficiently
df["time"] = df["time"].map(time_to_minutes)

# Encode categorical columns in one go
categorical_cols = ["firstThing", "weather", "goodBad", "mood"]
df[categorical_cols] = df[categorical_cols].apply(label_encoder.fit_transform)


# specifying features
features = ['time', 'sleepHours', 'firstThing', 'weather', 'mood']

# creating seperate variables for features & target

scaler = MinMaxScaler()

X = scaler.fit_transform(df[features]) # Normalizing 
y = df['goodBad']

# Using Gini = 1 - (x/n)2 - (y/n)2 to make the actual decision tree
dtree = DecisionTreeClassifier()
dtree = dtree.fit(X, y)

# Plotting the decision tree

def showTree():
    tree.plot_tree(dtree, feature_names=features, rounded=True, class_names=['Good', 'Bad'])
    plt.show()

# Predicting how your day will be

def predict(): 
    prediction = dtree.predict([[
                            time_to_minutes('11:00'), # Time you wake up (4:00 to 14:00)
                            14, # Amount of hours you slept (4 to 14 Hours)
                            3, # First thing you do in the morning { 'Phone': 0, 'Shower': 1, 'Eat' : 2, 'Dress' : 3, 'Excercise' : 4, 'Meditate' : 5 }
                            2, # Weather on that Day { 'Snowy': 0, 'Clear': 1, 'Rainy' : 2, 'Sunny' : 3, 'Cloudy' : 4, 'Foggy' : 5, 'Hail' : 6 }
                            0, # Mood when you wake up { 'refreshed' : 0, 'tired' : 1, 'happy' : 2, 'lazy' : 3, 'groggy' : 4, 'sad' : 5 }
                        ]])
    # [0] GOOD
    # [1] BAD
    if prediction == 0:
        return 'Based on what you told me, I predict your day will be GOOD.'
    else:
        return 'Based on what you\'ve told me I believe you day will be BAD.'
    
showTree()
#print(predict())