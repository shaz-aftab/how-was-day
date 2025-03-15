"""
 - Predicts your day using logistic regression based on stress levels
"""


import numpy as np
from sklearn import linear_model

x = np.array([8, 9, 2, 1, 10, 6, 3, 7, 5, 4,]).reshape(-1, 1) # Stress Levels in the morning (1-10 scale)
y = np.array([0, 0, 1, 1, 0, 0, 1, 1, 1, 0]) # Was the day good or bad (0 bad) (1 good)

logr = linear_model.LogisticRegression()
logr.fit(x, y)

def predict(stress): # Enter stress level and it predicts if the days gonna be good or bad solely based of that.
    predicted = logr.predict(np.array([stress]).reshape(-1,1))
    if predicted == 0:
        return f"I predict your day will be BAD. {predicted}"
    else:
        return f"I predict your day will be GOOD. {predicted}"

def odds(): # tells us that as stress levels increase by one the odds of it being a good day
    log_odds = logr.coef_
    odds = np.exp(log_odds)
    return(odds)

def probability(logr, x): # Probability of the days being good
    log_odds = logr.coef_ * x + logr.intercept_
    odds = np.exp(log_odds)
    probability = odds / (1 + odds)
    return np.round(probability*100).astype(int)

#print(predict(5))
#print(odds())
#print(probability(logr, x))