# src/train.py

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib
import os

dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, '../model/logistic_regression_model.pkl')


def train():
    # Load the iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=0.2, random_state=42)

    # Create and train the model
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)

    # Evaluate the model
    accuracy = model.score(X_test, y_test)
    print(f"Model accuracy: {accuracy:.2f}")

    # Save the trained model
    os.makedirs("model", exist_ok=True)
    joblib.dump(model, filename)

    print("Model training complete and saved")


if __name__ == "__main__":
    train()
