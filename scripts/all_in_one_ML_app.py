import sys
import joblib
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# ====================================================================
# 1. PREPARE THE TRAINING AND TEST DATA FOR PREDICTIVE MODEL
# ====================================================================
def prepare_data(training_output_path, test_output_path):
    # Generates synthetic classification data and splits it into train/test CSVs
    print(f"Executing: prepare_data -> {training_output_path}, {test_output_path}")
    X, y = make_classification(
        n_samples=1000,
        n_features=4,
        random_state=42
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,      
        random_state=42,    
        stratify=y          
    )
   
    df_train = pd.DataFrame(X_train, columns=["feature_1", "feature_2", "feature_3", "feature_4"])
    df_train["target"] = y_train

    df_test = pd.DataFrame(X_test, columns=["feature_1", "feature_2", "feature_3", "feature_4"])
    df_test["target"] = y_test
   
    df_train.to_csv(training_output_path, index=False)
    df_test.to_csv(test_output_path, index=False)
    print("Data preparation completed successfully.")


# ==============================================================
# 2. READ TRAINING DATA AND TRAIN A LOGISTIC REGRESSION MODEL
# ==============================================================
def train_model(training_dataset_path, model_path):
    # Reads training data from a CSV path and serializes a trained Logistic Regression model
    print(f"Executing: train_model -> Reading: {training_dataset_path} | Saving to: {model_path}")
    df_train = pd.read_csv(training_dataset_path)
   
    X_train = df_train.drop(columns=["target"])
    y_train = df_train["target"]

    model = LogisticRegression()
    model.fit(X_train, y_train)

    joblib.dump(model, model_path)
    print("Model training completed successfully.")


# ==========================================
# 3. EVALUATE THE MODEL
# ==========================================
def evaluate_model(test_dataset_path, model_path, accuracy_path):
    # Evaluates the trained model against test data and saves the final accuracy metric score.
    print(f"Executing: evaluate_model -> Test Data: {test_dataset_path} | Model: {model_path}")
    df_test = pd.read_csv(test_dataset_path)
    X_test = df_test.drop(columns=["target"])
    y_test = df_test["target"]

    model = joblib.load(model_path)
    
    y_predicted = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_predicted)
    report = classification_report(y_test, y_predicted)

    print("Accuracy: ", accuracy)
    print(report)

    with open(accuracy_path, "w") as f:
        f.write(str(accuracy))
    
    print(f"Accuracy written to path file location: {accuracy_path}")
    return accuracy


# ====================================================================
# COMMAND LINE ROUTING INTERFACE
# ====================================================================
if __name__ == "__main__":
    # Check if a specific function command action keyword was provided
    if len(sys.argv) < 2:
        print("Usage: python pipeline_tasks.py [prepare|train|evaluate] [args...]")
        sys.exit(1)

    action = sys.argv[1].lower()

    if action == "prepare":
        if len(sys.argv) < 4:
            print("Usage: python pipeline_tasks.py prepare <train_output_path> <test_output_path>")
            sys.exit(1)
        prepare_data(sys.argv[2], sys.argv[3])

    elif action == "train":
        if len(sys.argv) < 4:
            print("Usage: python pipeline_tasks.py train <training_dataset_path> <model_path>")
            sys.exit(1)
        train_model(sys.argv[2], sys.argv[3])

    elif action == "evaluate":
        if len(sys.argv) < 5:
            print("Usage: python pipeline_tasks.py evaluate <test_dataset_path> <model_path> <accuracy_path>")
            sys.exit(1)
        evaluate_model(sys.argv[2], sys.argv[3], sys.argv[4])

    else:
        print(f"Unknown action: '{action}'. Choose from: 'prepare', 'train', or 'evaluate'.")
        sys.exit(1)
