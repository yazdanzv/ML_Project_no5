# Imports
import numpy as np
import pandas as pd

# Scikit-Learn imports
from sklearn.naive_bayes import GaussianNB, CategoricalNB



class NewBayesModel:
    def __init__(self, cat_indices):
        self.cat_indices = cat_indices
        self.num_indices = None
        self.gnb = GaussianNB()
        self.cnb = CategoricalNB()

    def fit(self, X, y):
        self.num_indices = [i for i in range(X.shape[1]) if i not in self.cat_indices]

        # Separate the data
        X_num = X[:, self.num_indices]
        X_cat = X[:, self.cat_indices]

        # Fit models
        self.gnb.fit(X_num, y)
        self.cnb.fit(X_cat, y)

    def predict(self, X):
        X_num = X[:, self.num_indices]
        X_cat = X[:, self.cat_indices]

        # Get probabilities
        num_probs = self.gnb.predict_proba(X_num)
        cat_probs = self.cnb.predict_proba(X_cat)

        # Combine by multiplying probabilities and predict class with highest probability
        combined_probs = num_probs * cat_probs
        predictions = np.argmax(combined_probs, axis=1)
        return predictions