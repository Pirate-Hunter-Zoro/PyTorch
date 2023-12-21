# 1) Design model (input size, output size, forward pass)
# 2) Construct loss and optimizer
# 3) Training loop
#   - forward pass: compute prediction and loss
#   - backward pass: gradients
#   - update weights

import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import (
    StandardScaler,
)  # scaling of data - whatever the fuck that means
from sklearn.model_selection import (
    train_test_split,
)  # separation of training and testing data

# 0) prepare data
bc = (
    datasets.load_breast_cancer()
)  # binary classification problem where we predict cancer based on input features
X, Y = bc.data, bc.target

n_samples, n_features = X.shape

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=1234
)

# scale
sc = StandardScaler()  # zero mean, unit variance
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# convert all the numpy arrays into tensors
X_train = torch.from_numpy(X_train.astype(np.float32))
Y_train = torch.from_numpy(Y_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
Y_test = torch.from_numpy(Y_test.astype(np.float32))

# reshape the output
Y_train = Y_train.view(Y_train.shape[0], 1)  # turn into a column vector of 0's and 1's
Y_test = Y_test.view(Y_test.shape[0], 1)


# 1) model
# f = wx + b, sigmoid at the end
class LogisticRegression(nn.Module):
    def __init__(self, n_input_features):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(n_input_features, 1)

    def forward(self, x):
        y_predicted = torch.sigmoid(self.linear(x))
        return y_predicted


model = LogisticRegression(n_features)

# 2) loss and optimizer
learning_rate = 0.01
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 3) training loop
num_epochs = 300
for epoch in range(num_epochs):
    # forward pass and loss
    Y_predicted = model(X_train)
    loss = criterion(Y_predicted, Y_train)

    # backward pass
    loss.backward()

    # updates
    optimizer.step()
    
    # zero gradients
    optimizer.zero_grad()

    if (epoch+1) % 10 == 0:
        print(f'epoch: {epoch+1}, loss = {loss.item():.4f}')
        
with torch.no_grad(): # do not track the gradients here
    Y_predicted = model(X_test) # classification predictions - between 0 and 1
    Y_predicted_cls = Y_predicted.round()
    # create a vector of 0's and 1's based on if each prediction was accurate - sum those up and divide by the total
    acc = Y_predicted_cls.eq(Y_test).sum() / float(Y_test.shape[0])
    print(f'accuracy = {acc:.4f}')