import pandas as pd
import numpy as np
import math
from collections import defaultdict, deque

class TANManual:
    def __init__(self, q_bins=5):
        self.priors = {}
        self.cpts = {}       
        self.treeEdges = {} 
        
        self.q_bins = q_bins
        self.bins = {}      
        
    def fit(self, x, y):
        self.classes = list(set(y))
        self.features = list(x.columns)
        
        xDiscrete = x.copy()
        for col in self.features:
            xDiscrete[col], binEdges = pd.qcut(x[col], q=self.q_bins, retbins=True, labels=False, duplicates='drop')
            
            binEdges[0] = float('-inf')
            binEdges[-1] = float('inf')
            self.bins[col] = binEdges
            
        cmiWeights = []
        for i in range(len(self.features)):
            for j in range(i + 1, len(self.features)):
                featX = self.features[i]
                featY = self.features[j]
                
                cmi = self._calculate_cmi(xDiscrete[featX], xDiscrete[featY], y)
                cmiWeights.append((cmi, featX, featY))
                
        cmiWeights.sort(key=lambda item: item[0], reverse=True)
        
        parentUF = {feat: feat for feat in self.features}
        
        def find(i):
            if parentUF[i] == i:
                return i
            return find(parentUF[i])
            
        def union(i, j):
            rootI = find(i)
            rootJ = find(j)
            if rootI != rootJ:
                parentUF[rootI] = rootJ
                return True
            return False

        mstEdges = defaultdict(list)
        for weight, u, v in cmiWeights:
            if union(u, v):
                mstEdges[u].append(v)
                mstEdges[v].append(u)

        root = self.features[0]
        self.treeEdges = {} 
        
        visited = set([root])
        queue = deque([root])
        
        while queue:
            current = queue.popleft()
            for neighbor in mstEdges[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    self.treeEdges[neighbor] = current
                    queue.append(neighbor)
                    
        self._calculate_probabilities(xDiscrete, y)
        return self

    def _calculate_cmi(self, X_col, Y_col, y):
        df = pd.DataFrame({'X': X_col, 'Y': Y_col, 'C': y})
        cmi = 0.0
        
        prob_C = df['C'].value_counts(normalize=True).to_dict()
        
        for c in self.classes:
            df_c = df[df['C'] == c]
            if len(df_c) == 0: continue
            
            joint_counts = df_c.groupby(['X', 'Y']).size() / len(df_c)
            p_x_c = df_c['X'].value_counts(normalize=True).to_dict()
            p_y_c = df_c['Y'].value_counts(normalize=True).to_dict()
            
            for (x_val, y_val), p_xy_c in joint_counts.items():
                p_xyc = p_xy_c * prob_C[c]
                
                denominator = p_x_c.get(x_val, 1e-9) * p_y_c.get(y_val, 1e-9)
                if p_xy_c > 0 and denominator > 0:
                    cmi += p_xyc * math.log(p_xy_c / denominator)
                    
        return cmi

    def _calculate_probabilities(self, x, y):
        df = x.copy()
        df['Class'] = y
        totalSamples = len(y)
        
        for c in self.classes:
            df_c = df[df['Class'] == c]
            self.priors[c] = len(df_c) / totalSamples
            self.cpts[c] = {}
            
            for feature in self.features:
                self.cpts[c][feature] = {}
                parent = self.treeEdges.get(feature, None)
                
                if parent is None:
                    counts = df_c[feature].value_counts(normalize=True).to_dict()
                    self.cpts[c][feature] = {'no_parent': counts}
                else:
                    grouped = df_c.groupby(parent)[feature].value_counts(normalize=True)
                    for parent_val in df_c[parent].unique():
                        if parent_val in grouped:
                            self.cpts[c][feature][parent_val] = grouped[parent_val].to_dict()

    def predict(self, x):
        predictions = []
        epsilon = 1e-9
        
        xDiscrete = x.copy()
        for col in self.features:
            xDiscrete[col] = pd.cut(x[col], bins=self.bins[col], labels=False, include_lowest=True)
            
        for index, row in xDiscrete.iterrows():
            best_class = None
            highest_prob = float('-inf')
            
            for c in self.classes:
                total_log_prob = math.log(self.priors[c] + epsilon)
                
                for feature in self.features:
                    feature_val = row[feature]
                    parent = self.treeEdges.get(feature, None)
                    
                    if parent is None:
                        prob_dict = self.cpts[c][feature].get('no_parent', {})
                        likelihood = prob_dict.get(feature_val, epsilon)
                    else:
                        parent_val = row[parent]
                        prob_dict = self.cpts[c][feature].get(parent_val, {})
                        likelihood = prob_dict.get(feature_val, epsilon)
                        
                    total_log_prob += math.log(likelihood)
                    
                if total_log_prob > highest_prob:
                    highest_prob = total_log_prob
                    best_class = c
                    
            predictions.append(best_class)
            
        return predictions