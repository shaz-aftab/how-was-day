import pandas as pd
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

df = pd.read_csv("./days.csv") # reading the CSV file

# Remapping

d = { '7AM' : 7, '8AM' : 8, '9AM' : 9, '10AM' : 10, '11AM' : 11 }
df['time'] = df['time'].map(d)
d = { 'Phone': 0, 'Shower': 1, 'Eat' : 2 }
df['firstThing'] = df['firstThing'].map(d)
d = { 'Snowy': 0, 'Clear': 1, 'Rainy' : 2, 'Sunny' : 3, 'Other' : 4 }
df['weather'] = df['weather'].map(d)
d = { 'Good' : 0, 'Bad' : 1 }
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
    plt.figure(figsize=(6, 7))
    tree.plot_tree(dtree, feature_names=features, rounded=True, class_names=['Good', 'Bad'])
    plt.show()

# Predicting how your day will be

def predict(): 
    prediction = dtree.predict([[
                            11, # Time you wake up (7AM to 11AM)
                            10, # Amount of hours you slept (6 to 10 Hours)
                            0, # First thing you do in the morning ('Phone': 0, 'Shower': 1, 'Eat' : 2)
                            1 # Weather on that Day ('Snowy': 0, 'Clear': 1, 'Rainy' : 2, 'Sunny' : 3, 'Other' : 4)
                        ]])
    # [0] GOOD
    # [1] BAD
    if prediction == 0:
        return 'Based on what you told me, I predict your day will be GOOD.'
    else:
        return 'Based on what you\'ve told me I believe you day will be BAD.'





