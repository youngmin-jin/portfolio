# based on Google colab

# --------------------------------------------------------------
# import modules
# --------------------------------------------------------------
# analysis
import numpy as np
import pandas as pd

# file upload
from google.colab import files
import io

# visualization
import matplotlib.pyplot as plt
import missingno 
from missingno.missingno import heatmap
from pandas.plotting import boxplot

# --------------------------------------------------------------
# 1. EDA & feature engineering
# --------------------------------------------------------------
# read data
uploaded = files.upload()
df_analysis = pd.read_csv(io.BytesIO(uploaded['acoustic_analysis.csv']), encoding='utf-8')
df_details = pd.read_csv(io.BytesIO(uploaded['details.csv']), encoding='utf-8')
