import pandas as pd
import joblib
import json
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

# ==========================================
# 3. EVALUATE THE MODEL
# ==========================================
def evaluate_model(test_dataset_path, model_path, accuracy_path):

    # Reads test data from the dataset...
    df_test = pd.read_csv(test_dataset_path)
    # Separate features and target
    X_test = df_test.drop(columns=["target"])
    y_test = df_test["target"]

    # load the model
    model = joblib.load(model_path)
    
    # test the model and print the accuracy and the full accuracy metrics report
    y_predicted = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_predicted)
    report = classification_report(y_test, y_predicted)

    print("Accuracy: ", accuracy)
    print(report)

    # Write the accuracy in a file because the python module output is not always available to Kubeflow
    with open(accuracy_path, "w") as f:
        f.write(str(accuracy))
    
    return accuracy

if __name__ == "__main__":
    import sys
    evaluate_model(sys.argv[1], sys.argv[2], sys.argv[3])




