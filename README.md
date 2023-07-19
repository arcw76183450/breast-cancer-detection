# Tumor Classification with Keras

This Jupyter Notebook demonstrates how to use Keras, a popular deep learning library, to build a model that can classify tumors as benign or malignant. The dataset used for this task is the Breast Cancer Wisconsin (Diagnostic) Dataset, which contains various features extracted from digitized images of breast mass.

## Table of Contents

1. [Installation](#installation)
2. [Data Preparation](#data-preparation)
3. [Model Training](#model-training)
4. [Model Evaluation](#model-evaluation)
5. [Conclusion](#conclusion)

## Installation<a id="installation"></a>

To run this notebook, you will need the following libraries:

- Keras
- NumPy
- Pandas
- Scikit-learn
- Matplotlib

You can install these dependencies using pip by running the following command:

```
pip install keras numpy pandas scikit-learn matplotlib
```

Additionally, you will need to have Jupyter Notebook installed. You can install Jupyter Notebook using pip:

```
pip install jupyter
```

Once you have the required libraries installed, you can proceed with running the notebook.

## Data Preparation<a id="data-preparation"></a>

1. Start by importing the necessary libraries:

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
```

2. Load the dataset using `pd.read_csv()`:

```python
dataset=pd.read_csv('breastcancer.csv')
dataset.head()
```

3. Separate the features from the target variable:

```python
X = data.drop(["id", "diagnosis"], axis=1)
y = data["diagnosis"]
```

4. Split the dataset into training and testing sets:

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

5. Perform feature scaling using `StandardScaler`:

```python
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

## Model Training<a id="model-training"></a>

1. Import the necessary Keras modules:

```python
from keras.models import Sequential
from keras.layers import Dense
```

2. Build the model architecture:

```python
model = Sequential()
model.add(Dense(16, activation="relu", input_shape=(X_train.shape[1],)))
model.add(Dense(16, activation="relu"))
model.add(Dense(1, activation="sigmoid"))
```

3. Compile the model:

```python
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
```

4. Train the model:

```python
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2)
```

## Model Evaluation<a id="model-evaluation"></a>

1. Plot the training and validation accuracy:

```python
import matplotlib.pyplot as plt

plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(["train", "validation"], loc="upper left")
plt.show()
```

2. Evaluate the model on the test set:

```python
_, accuracy = model.evaluate(X_test, y_test)
print("Test Accuracy:", accuracy)
```

## Conclusion<a id="conclusion"></a>

In this notebook, we have demonstrated how to build a tumor classification model using Keras. By training a neural network on the Breast Cancer Wisconsin dataset, we achieved a certain accuracy on the test set. This model can be further improved by experimenting with different architectures, hyperparameters, and training strategies.