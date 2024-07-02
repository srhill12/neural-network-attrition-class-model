# README for Attrition and Department Classification Model

## Overview
This project aims to develop a neural network model to predict employee attrition and department based on various features. The dataset used is the attrition dataset from [BC-EDX](https://static.bc-edx.com/ai/ail-v-1-0/m19/lms/datasets/attrition.csv). The project includes data preprocessing, model creation, training, and evaluation.

## Instructions

### Open In Colab
You can run this project in Google Colab for an interactive experience.

### Part 1: Preprocessing

#### Import Dependencies
```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras import layers

# Load the dataset
attrition_df = pd.read_csv('https://static.bc-edx.com/ai/ail-v-1-0/m19/lms/datasets/attrition.csv')

# Display first few rows
attrition_df.head()

# Determine unique values in each column
attrition_df.nunique()
