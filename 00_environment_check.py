"""Verify the Ubuntu Python environment for the coursework project."""

import sys

import matplotlib
import numpy as np
import openpyxl
import pandas as pd
import sklearn
from sklearn.linear_model import LinearRegression


data = pd.DataFrame({"input": [1.0, 2.0, 3.0], "output": [2.0, 4.0, 6.0]})
model = LinearRegression().fit(data[["input"]], data["output"])

print("Python:", sys.version.split()[0])
print("Executable:", sys.executable)
print("pandas:", pd.__version__)
print("NumPy:", np.__version__)
print("Matplotlib:", matplotlib.__version__)
print("scikit-learn:", sklearn.__version__)
print("openpyxl:", openpyxl.__version__)
print("Test prediction:", model.predict(pd.DataFrame({"input": [4.0]}))[0])
print("Environment check: OK")
