import pandas as pd
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

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
    
# Remapping

df["time"] = df["time"].apply(time_to_minutes)
#print(df['time'])
d = { 'Phone': 0, 'Shower': 1, 'Eat' : 2, 'Dress' : 3, 'Excercise' : 4, 'Meditate' : 5 }
df['firstThing'] = df['firstThing'].map(d)
d = { 'Snowy': 0, 'Clear': 1, 'Rainy' : 2, 'Sunny' : 3, 'Cloudy' : 4, 'Foggy' : 5, 'Hail' : 6 }
df['weather'] = df['weather'].map(d)
d = { False : 0, True : 1 }
df['goodBad'] = df['goodBad'].map(d)

# specifying features
features = ['time', 'sleepHours', 'firstThing', 'weather']

# creating seperate variables for features & target
X = df[features]
y = df['goodBad']

# Using Gini = 1 - (x/n)2 - (y/n)2 to make the actual decision tree
dtree = DecisionTreeClassifier()
dtree = dtree.fit(X, y)

# Plotting the decision tree

def showTree():
    #plt.figure(figsize=(6, 7))
    tree.plot_tree(dtree, feature_names=features, rounded=True, class_names=['Good', 'Bad'])
    plt.show()

# Predicting how your day will be

def predict(): 
    prediction = dtree.predict([[
                            time_to_minutes('11:00'), # Time you wake up (4:00 to 14:00)
                            10, # Amount of hours you slept (6 to 10 Hours)
                            0, # First thing you do in the morning { 'Phone': 0, 'Shower': 1, 'Eat' : 2, 'Dress' : 3, 'Excercise' : 4, 'Meditate' : 5 }
                            1 # Weather on that Day { 'Snowy': 0, 'Clear': 1, 'Rainy' : 2, 'Sunny' : 3, 'Cloudy' : 4, 'Foggy' : 5, 'Hail' : 6 }
                        ]])
    # [0] GOOD
    # [1] BAD
    if prediction == 0:
        return 'Based on what you told me, I predict your day will be GOOD.'
    else:
        return 'Based on what you\'ve told me I believe you day will be BAD.'
    
print(predict())