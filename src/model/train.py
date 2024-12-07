# Import libraries
import argparse
import glob
import os
import pandas as pd
import logging
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import mlflow
import mlflow.sklearn


# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")


# Define functions
def main(args):
    # Enable autologging
    mlflow.sklearn.autolog()

    logging.info("Starting the training script...")

    # Read data
    logging.info(f"Reading training data from: {args.training_data}")
    df = get_csvs_df(args.training_data)

    # Split data
    logging.info("Splitting data into train and test sets.")
    X_train, X_test, y_train, y_test = split_data(df)

    # Train model
    logging.info("Training the model.")
    train_model(args.reg_rate, X_train, X_test, y_train, y_test)

    logging.info("Training script completed.")


def get_csvs_df(path):
    if not os.path.exists(path):
        raise RuntimeError(f"Cannot use non-existent path provided: {path}")
    csv_files = glob.glob(f"{path}/*.csv")
    if not csv_files:
        raise RuntimeError(f"No CSV files found in provided data path: {path}")
    return pd.concat((pd.read_csv(f) for f in csv_files), sort=False)


# Function to split data
def split_data(df):
    """
    Splits data into training and testing sets.
    
    Args:
        df (pd.DataFrame): The input data containing features and target.

    Returns:
        X_train, X_test, y_train, y_test: Split features and target sets.
    """
    #if "target" not in df.columns:
     #   raise RuntimeError("Dataframe does not contain a 'target' column.")

    X = df.drop("Diabetic", axis=1)
    y = df["Diabetic"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test


def train_model(reg_rate, X_train, X_test, y_train, y_test):
    """
    Trains a Logistic Regression model.

    Args:
        reg_rate (float): Regularization rate.
        X_train (pd.DataFrame): Training features.
        X_test (pd.DataFrame): Testing features.
        y_train (pd.Series): Training target.
        y_test (pd.Series): Testing target.
    """
    with mlflow.start_run():
        # Train model
        model = LogisticRegression(C=1/reg_rate, solver="liblinear").fit(X_train, y_train)

        # Log model performance
        accuracy = model.score(X_test, y_test)
        logging.info(f"Model Accuracy: {accuracy:.4f}")

        # Log metrics
        mlflow.log_metric("accuracy", accuracy)


def parse_args():
    # Set up arg parser
    parser = argparse.ArgumentParser()

    # Add arguments
    parser.add_argument("--training_data", dest='training_data', type=str)
    parser.add_argument("--reg_rate", dest='reg_rate', type=float, default=0.01)

    # Parse args
    args = parser.parse_args()

    # Return args
    return args


# Run script
if __name__ == "__main__":
    # Add space in logs
    print("\n\n")
    print("*" * 60)

    # Parse args
    args = parse_args()

    # Run main function
    main(args)

    # Add space in logs
    print("*" * 60)
    print("\n\n")
