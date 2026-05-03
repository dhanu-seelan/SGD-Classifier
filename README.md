# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the dataset and convert the placement status into binary values (Placed = 1, Not Placed = 0).
2. Select the input features (ssc_p, mba_p) and normalize the data using StandardScaler.
3. Train the Logistic Regression model using Sigmoid function and Gradient Descent to update theta values.
4. Predict the output, calculate accuracy, and plot the cost function graph to visualize model learning.



## Program:
```
/*
Program to implement the prediction of iris species using SGD Classifier.
Developed by: Danaseelan G
RegisterNumber:  212225040053
*/
```

```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Load dataset
data = pd.read_csv("Placement_Data.csv")

# Convert target values into 0 and 1
data['status'] = data['status'].map({
    'Placed': 1,
    'Not Placed': 0
})

# Independent variables
X = data[['ssc_p', 'mba_p']].values

# Dependent variable
y = data['status'].values

# Feature Scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Add bias column
m = len(y)
X = np.c_[np.ones((m, 1)), X]

# Sigmoid Function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Cost Function
def cost_function(X, y, theta):
    h = sigmoid(X @ theta)
    
    # Avoid log(0) error
    h = np.clip(h, 1e-10, 1 - 1e-10)
    
    cost = (-1 / m) * np.sum(
        y * np.log(h) + (1 - y) * np.log(1 - h)
    )
    
    return cost

# Gradient Descent
theta = np.zeros(X.shape[1])

alpha = 0.1
iterations = 500

cost_history = []

for i in range(iterations):

    z = X @ theta
    h = sigmoid(z)

    gradient = (1 / m) * (X.T @ (h - y))

    theta = theta - alpha * gradient

    cost = cost_function(X, y, theta)

    cost_history.append(cost)

# Prediction
y_pred = (sigmoid(X @ theta) >= 0.5).astype(int)

# Accuracy
accuracy = np.mean(y_pred == y) * 100

print("Accuracy =", round(accuracy, 2), "%")

#Cost Function
plt.plot(range(iterations), cost_history)

plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Cost Function Reduction")

plt.show()
```
## Output:
<img width="919" height="719" alt="image" src="https://github.com/user-attachments/assets/53a28260-37a8-4be0-8dc5-897ee873d162" />


## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
