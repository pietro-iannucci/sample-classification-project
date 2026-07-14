import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# ===================================================
# 1. PREPARE THE TRAINING AND TEST DATA FOR PREDICTIVE MODEL
# ===================================================
def prepare_data(training_output_path, test_output_path):

    # Generate synthetic classification data
    X, y = make_classification(
        n_samples=1000,
        n_features=4,
        random_state=42
    )

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,      # 20% of the data for testing
        random_state=42,    # Ensures reproducibility
        stratify=y          # Maintains the class distribution in both sets
    )
   
    # Create a DataFrame for the training dataset
    df_train = pd.DataFrame(X_train, columns=[
        "feature_1",
        "feature_2",
        "feature_3",
        "feature_4"
    ])
    # Add the target column
    df_train["target"] = y_train

    # Create a DataFrame for the test dataset
    df_test = pd.DataFrame(X_test, columns=[
        "feature_1",
        "feature_2",
        "feature_3",
        "feature_4"
    ])
    # Add the target column
    df_test["target"] = y_test
   
    # Save to CSV
    df_train.to_csv(training_output_path, index=False)
    df_test.to_csv(test_output_path, index=False)

if __name__ == "__main__":
    import sys
    prepare_data(sys.argv[1], sys.argv[2])
