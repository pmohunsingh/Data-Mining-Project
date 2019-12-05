# Import libraries 
import numpy as np
import pandas as pd
from time import time
from IPython.display import display 

# Visualization code visuals.py
import visuals as vs

# Pretty display for notebooks
%matplotlib inline

# Load the Census dataset
data = pd.read_csv("train_mode_file.csv")

# Success - Display the first record
display(data.head(n=1))
