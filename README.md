# Attrition and Department Classification Model

This project involves building a neural network model to predict employee attrition and department based on various features. The model uses data from a company to classify whether an employee will leave the company and which department they belong to.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Project Steps](#project-steps)
  - [Data Preparation](#data-preparation)
  - [Model Building](#model-building)
  - [Model Evaluation](#model-evaluation)
  - [Saving and Loading the Model](#saving-and-loading-the-model)
  - [Prediction](#prediction)
- [Results](#results)
- [Discussion](#discussion)
- [Sources](#sources)
- [Conclusion](#conclusion)

## Installation

Install the required packages:
- pandas
- tensorflow
- sklearn

```bash
pip install pandas tensorflow scikit-learn

```
## Usage

To run the project, follow the steps below:

### Prepare the data:

Download the dataset from [here](https://static.bc-edx.com/ai/ail-v-1-0/m19/lms/datasets/attrition.csv)

### Run the script:

Execute the script to train and evaluate the model.

```bash
python attrition_department_classification.py
```

## Project Steps

### Data Preparation

#### Read the Data
Load the attrition data into a Pandas DataFrame.

```python
attrition_df = pd.read_csv('https://static.bc-edx.com/ai/ail-v-1-0/m19/lms/datasets/attrition.csv')
attrition_df.head()
```

#### Feature and Target Selection
Define features (X) and target (y) datasets.

```python
# Define y variables
y_attrition = attrition_df[['Attrition']]
y_department = attrition_df[['Department']]

# Define X variables
X_df = attrition_df[['Education', 'Age', 'DistanceFromHome', 'JobSatisfaction', 'OverTime', 'StockOptionLevel', 'WorkLifeBalance', 'YearsAtCompany', 'YearsSinceLastPromotion', 'NumCompaniesWorked']]
```

#### Data Splitting
Split the data into training and testing sets.

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train_attrition, y_test_attrition, y_train_department, y_test_department = train_test_split(X_df, y_attrition, y_department, test_size=0.2, random_state=42)
```

#### Convert Categorical Data
Convert categorical data to numeric data types.

```python
from sklearn.preprocessing import LabelEncoder

# Initialize LabelEncoder
le = LabelEncoder()

# Transform "OverTime" column
X_train["OverTime"] = le.fit_transform(X_train["OverTime"])
X_test["OverTime"] = le.transform(X_test["OverTime"])
```

#### Feature Scaling
Scale the feature data using StandardScaler.

```python
from sklearn.preprocessing import StandardScaler

# Create a StandardScaler
scaler = StandardScaler()

# Fit and scale the data
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

#### Encode Target Variables
One-hot encode the target variables.

```python
from sklearn.preprocessing import OneHotEncoder

# OneHotEncode Department
department_encoder = OneHotEncoder(sparse_output=False)
department_encoder.fit(y_train_department)
y_train_department_encoded = department_encoder.transform(y_train_department)
y_test_department_encoded = department_encoder.transform(y_test_department)

# OneHotEncode Attrition
attrition_encoder = OneHotEncoder(sparse_output=False)
attrition_encoder.fit(y_train_attrition)
y_train_attrition_encoded = attrition_encoder.transform(y_train_attrition)
y_test_attrition_encoded = attrition_encoder.transform(y_test_attrition)
```

### Model Building

#### Define the Model
Create a neural network model with TensorFlow.

```python
from tensorflow.keras.models import Model
from tensorflow.keras import layers

# Input layer
input_shape = X_train_scaled.shape[1]
input_layer = layers.Input(shape=(input_shape,), name="input")

# Shared layers
shared_layer1 = layers.Dense(64, activation="relu", name="shared1")(input_layer)
shared_layer2 = layers.Dense(128, activation="relu", name="shared2")(shared_layer1)

# Department branch
department_hidden = layers.Dense(32, activation="relu", name="department_hidden")(shared_layer2)
department_output = layers.Dense(3, activation="softmax", name="department_output")(department_hidden)

# Attrition branch
attrition_hidden = layers.Dense(32, activation="relu", name="attrition_hidden")(shared_layer2)
attrition_output = layers.Dense(2, activation="softmax", name="attrition_output")(attrition_hidden)

# Model creation
model = Model(inputs=input_layer, outputs=[department_output, attrition_output])
```

#### Compile the Model
Compile the model using categorical cross-entropy loss and the Adam optimizer.

```python
model.compile(optimizer="adam",
              loss={"department_output": "categorical_crossentropy", "attrition_output": "categorical_crossentropy"},
              metrics={"department_output": "accuracy", "attrition_output": "accuracy"})
```

#### Train the Model
Fit the model with the training data.

```python
history = model.fit(X_train_scaled, [y_train_department_encoded, y_train_attrition_encoded],
                    epochs=100, batch_size=32, validation_split=0.2)
```

### Model Evaluation

#### Evaluate the Model
Evaluate the model using test data.

```python
evaluation = model.evaluate(X_test_scaled, [y_test_department_encoded, y_test_attrition_encoded])

# Print accuracy
print(f"Department predictions accuracy: {evaluation[3]}")
print(f"Attrition predictions accuracy: {evaluation[4]}")
```

### Saving and Loading the Model

#### Save the Model
Save the model to a Keras file.

```python
model.save("attrition_department_model.keras")
```

#### Load the Model
Load the model from the saved file.

```python
from tensorflow.keras.models import load_model
model = load_model("attrition_department_model.keras")
```

### Prediction

#### Make Predictions
Use the model to make predictions on the test data.

```python
predictions = model.predict(X_test_scaled)
```

### Results
The model achieved an accuracy of approximately 48.6% for department predictions and 81% for attrition predictions on the test data.

### Discussion

#### Is accuracy the best metric to use on this data? Why or why not?
Accuracy is typically used for classification but may not always be the best when the dataset is imbalanced. Some other metrics that may be useful in these cases would be Precision and Recall or using an F-1 Score.

#### What activation functions did you choose for your output layers, and why?
For the Department output layer, I chose softmax because it ensured that the sum of probabilities would be 1. Softmax works for multi-class classification problems. For the Attrition output layer, softmax was used again. Sigmoid could have been used if the output for Attrition were a binary classification to output a single value for the class that identified as a positive one. Yes/No.

#### Can you name a few ways that this model might be improved?
1. **Hyperparameter Tuning**: Experiment with different learning rates, batch sizes, and epochs.
2. **Feature Engineering**: Add more relevant features or create interaction terms.
3. **Data Augmentation and Balancing**: Address class imbalance using techniques like SMOTE.
4. **Regularization Techniques**: Add dropout layers or L2 regularization to prevent overfitting.
5. **Model Architecture**: Experiment with different numbers of layers and neurons.
6. **Advanced Models**: Use more advanced models like Gradient Boosting, XGBoost, or neural network architectures like LSTM or GRU if applicable.

### Sources
- Activities from Modules 18 and 19, particularly Day 3 Ex 04. Xpert Learning Assistant for README formatting suggestions.

### Conclusion
This project demonstrates the application of deep learning for predicting employee attrition and department classification, providing a foundation for further improvements and applications in HR analytics.
```
