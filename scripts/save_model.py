import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression

# ==========================================
# 4. SAVE THE MODEL IN MODEL REGISTRY
# ==========================================
def save_model(trained_model_path, model_registry_path): 
    print(f"Executing: save_model -> Registry path: {model_registry_path}") 
    # Put your code here !!!!     
    print(f"Model saved to model registry: {model_registry_path}") 

if __name__ == "__main__":
    import sys
    save_model(sys.argv[1], sys.argv[2])

