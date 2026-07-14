import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression

# ==============================================================
# 2. READ TRAINING DATA AND TRAIN A LOGISTIC REGRESSION MODEL
# ==============================================================
def train_model(training_dataset_path, model_path):

    # Reads training data from the dataset...
    df_train = pd.read_csv(training_dataset_path)
   
    # Separate features and target
    X_train = df_train.drop(columns=["target"])
    y_train = df_train["target"]

    # Train the model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # serialize and dump the model into a file 
    joblib.dump(model, model_path)

if __name__ == "__main__":
    import sys
    train_model(sys.argv[1], sys.argv[2])

