# Import libraries 
import numpy as np
import pandas as pd
from time import time
from IPython.display import display 

# Visualization code 
import visuals as vs

# Display
%matplotlib inline

# Load Census dataset
data = pd.read_csv("train_mode_file.csv")

# Display first record
display(data.head(n=1))
