# tests/test_train.py

import sys
import os
import pytest

# Redirect stderr to avoid fileno issues
sys.stderr = open(os.devnull, 'w')

# Add src directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from train import train

def test_train():
    # Ensure the model directory does not exist before training
    if os.path.exists("model/logistic_regression_model.pkl"):
        os.remove("model/logistic_regression_model.pkl")
    
    # Run the training function
    train()
    
    # Check that the model file is created
    assert os.path.exists("model/logistic_regression_model.pkl"), "Model file not created"

if __name__ == "__main__":
    pytest.main(["-v", "tests/test_train.py"])
