# Gaussian Naive Bayes
from Dataset.data_loader import data_loader
import pandas as pd
import numpy as np
import math

class NaiveBayesManual:
    def __init__(self):
        self.means = {}
        self.priors = {}
        self.vars = {}
        x, y = data_loader()

    def fit(self, x, y):
        self.classes = list(set(y))
        totalSamples = len(y)

        for c in self.classes:
            self.means[c] = {}
            self.vars[c] = {}

            # Calculating the prior
            
            classCount = 0
            for label in y:
                if label == c:
                    classCount += 1

            self.priors[c] = classCount / totalSamples

            for column in x.columns:
            
                # Calculating te mean
            
                colSum = 0.0
                for val, label in zip(x[column], y):
                    if label == c:
                        colSum += val

                featureMean = colSum / classCount
                self.means[c][column] = featureMean

                # Calculating the variance
                squaredDiffSum = 0.0
                for val, label in zip(x[column], y):
                    if label == c:
                        squaredDiffSum += (val - featureMean) ** 2

                featureVar = squaredDiffSum / classCount
                self.vars[c][column] = featureVar
        
        return self

    def predict(self, x):
        predictions = []

        for index, row in x.iterrows():
            bestClass = None
            highestProb = float('-inf')

            for c in self.classes:
                totalLogProb = math.log(self.priors[c])

                for column in x.columns:
                    featureValue = row[column]
                    mean = self.means[c][column]
                    var = self.vars[c][column]

                    epsilon = 1e-9
                    var += epsilon

                    exponent = math.exp(-((featureValue - mean) ** 2) / (2 * var))
                    likelihood = (1 / math.sqrt(2 * math.pi * var)) * exponent

                    likelihood += epsilon

                    totalLogProb += math.log(likelihood)

                if totalLogProb > highestProb:
                    highestProb = totalLogProb
                    bestClass = c

            predictions.append(bestClass)

        return predictions

            