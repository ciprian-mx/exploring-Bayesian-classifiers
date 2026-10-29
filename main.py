from Dataset.data_loader import data_loader
from src import *

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix
import time

def main():
    x, y = data_loader()
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    
    models = {
        "Scikit-Learn Baseline": GaussianNB(),
        "Manual Naive Bayes": NaiveBayesManual()
        # TANManual() 
        # BBNManual()
    }
    
    for name, model in models.items():
        print(f"--- Evaluating {name} ---")
        
        start_time = time.time()
        model.fit(x_train, y_train)
        train_time = time.time() - start_time
        
        predictions = model.predict(x_test)
        
        accuracy = accuracy_score(y_test, predictions)
        cm = confusion_matrix(y_test, predictions)
        
        # Display results
        print(f"Accuracy: {accuracy * 100:.2f}%")
        print(f"Training Time: {train_time:.4f} seconds")
        print("Confusion Matrix:")
        print(f"                 Predicted Non-Viral(0)   Predicted Viral(1)")
        print(f"Actual Non-Viral(0)      {cm[0][0]:<22} {cm[0][1]}")
        print(f"Actual Viral(1)          {cm[1][0]:<22} {cm[1][1]}")
        print("\n")

if __name__ == "__main__":
    main()