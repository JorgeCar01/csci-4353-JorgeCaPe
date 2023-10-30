# import pandas as pd
# from sklearn.preprocessing import LabelEncoder, StandardScaler
# from sklearn.model_selection import train_test_split
# import torch
# import torch.nn as nn
# import torch.optim as optim


# # Check if GPU is available and if not, fall back to CPU
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# # Load the dataset
# train_data = pd.read_csv(
#     r"C:\school\class repos\csci-4353-JorgeCaPe\labs\lab15\data\train.csv"
# )

# # Handle missing values
# train_data["Age"].fillna(train_data["Age"].mean(), inplace=True)
# train_data["Embarked"].fillna(train_data["Embarked"].mode()[0], inplace=True)

# # Encode categorical variables
# le_sex = LabelEncoder()
# train_data["Sex"] = le_sex.fit_transform(train_data["Sex"])

# le_embarked = LabelEncoder()
# train_data["Embarked"] = le_embarked.fit_transform(train_data["Embarked"])

# # Normalize numerical features
# scaler = StandardScaler()
# train_data[["Age", "Fare"]] = scaler.fit_transform(train_data[["Age", "Fare"]])

# # Split the training data into train and validation
# features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
# X = train_data[features].values
# y = train_data["Survived"].values

# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# class TitanicNet(nn.Module):
#     def __init__(self, dropout_rate=0.5):
#         super(TitanicNet, self).__init__()
#         self.fc1 = nn.Linear(len(features), 64)
#         self.dropout1 = nn.Dropout(p=dropout_rate)
#         self.fc2 = nn.Linear(64, 32)
#         self.dropout2 = nn.Dropout(p=dropout_rate)
#         self.fc3 = nn.Linear(32, 1)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         x = torch.relu(self.fc1(x))
#         x = self.dropout1(x)
#         x = torch.relu(self.fc2(x))
#         x = self.dropout2(x)
#         x = self.sigmoid(self.fc3(x))
#         return x


# model = TitanicNet().to(device)

# X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
# y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
# X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
# y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1).to(device)
# # Hyperparameters & other setup
# learning_rate = 0.001
# num_epochs = 1000  # increased, as early stopping may halt training prematurely
# criterion = nn.BCELoss()
# optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# # Early stopping parameters
# patience = 20
# best_val_loss = float("inf")
# counter = 0


# def calculate_accuracy(y_pred, y_true):
#     # Convert probabilities to binary predictions (0 or 1)
#     y_pred = (y_pred > 0.5).float()

#     # Calculate the correct predictions
#     correct_predictions = (y_pred == y_true).sum().item()

#     return correct_predictions / y_pred.numel()


# # Training loop with early stopping
# for epoch in range(num_epochs):
#     model.train()
#     optimizer.zero_grad()
#     outputs = model(X_train_tensor)
#     loss = criterion(outputs, y_train_tensor)
#     loss.backward()
#     optimizer.step()

#     train_acc = calculate_accuracy(outputs, y_train_tensor)
#     # Validation phase
#     model.eval()
#     with torch.no_grad():
#         val_outputs = model(X_val_tensor)
#         val_loss = criterion(val_outputs, y_val_tensor)

#     val_acc = calculate_accuracy(val_outputs, y_val_tensor)
#     # Print training progress
#     if (epoch + 1) % 10 == 0:
#         print(
#             f"Epoch [{epoch+1}/{num_epochs}]\n Train Loss: {loss.item():.4f}, Train Accuracy: {train_acc:.4f}\n Val Loss: {val_loss.item():.4f}, Val Accuracy: {val_acc:.4f}"
#         )

#     # Early stopping logic
#     if val_loss < best_val_loss:
#         best_val_loss = val_loss
#         counter = 0  # Reset the counter
#     else:
#         counter += 1

#     if counter == patience:
#         print("Early stopping...")
#         break


# def preprocess_data(df):
#     # Your preprocessing steps here...
#     # For example:
#     df = df.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1)
#     df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
#     df["Embarked"] = df["Embarked"].map({"S": 0, "C": 1, "Q": 2})
#     df = df.fillna(0)  # Handle missing values as an example
#     # ... Add more preprocessing if necessary

#     return df


# # Load training data and preprocess
# # train_data = pd.read_csv("train.csv")
# # train_data = preprocess_data(train_data)
# # y_train = train_data["Survived"].values
# # X_train = train_data.drop("Survived", axis=1).values

# # Load test data and preprocess
# test_data = pd.read_csv(
#     r"C:\school\class repos\csci-4353-JorgeCaPe\labs\lab15\data\test.csv"
# )
# passenger_ids = test_data["PassengerId"].values
# test_data = preprocess_data(test_data)
# X_test = test_data.values  # Assuming test.csv doesn't have the "Survived" column

# # Convert test data to tensor and move to device
# X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)

# # Predict
# model.eval()  # Set the model to evaluation mode
# with torch.no_grad():
#     test_outputs = model(X_test_tensor)
#     print(test_outputs)
#     test_predictions = (
#         (test_outputs > 0.5).float().numpy()
#     )  # Convert predictions to 0 or 1
# # print("PREDS\n", test_predictions)
# output = pd.DataFrame(
#     {"PassengerId": passenger_ids, "Survived": test_predictions.reshape(-1)}
# )
# output.to_csv("predictions.csv", index=False)

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    StackingClassifier,
)
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression

# Load data
train = pd.read_csv(
    r"C:\school\class repos\csci-4353-JorgeCaPe\labs\lab15\data\train.csv"
)
test = pd.read_csv(
    r"C:\school\class repos\csci-4353-JorgeCaPe\labs\lab15\data\test.csv"
)

# Handle missing values
train["Age"].fillna(train["Age"].mean(), inplace=True)
train["Embarked"].fillna(train["Embarked"].mode()[0], inplace=True)
test["Age"].fillna(test["Age"].mean(), inplace=True)
test["Fare"].fillna(test["Fare"].mean(), inplace=True)
test["Embarked"].fillna(test["Embarked"].mode()[0], inplace=True)

# Convert categorical variables to numerical
train = pd.get_dummies(train, columns=["Sex", "Embarked"])
test = pd.get_dummies(test, columns=["Sex", "Embarked"])

# Feature Engineering

# Extract Titles from Names
train["Title"] = train["Name"].apply(lambda x: x.split(",")[1].split(".")[0].strip())
test["Title"] = test["Name"].apply(lambda x: x.split(",")[1].split(".")[0].strip())
common_titles = ["Mr", "Miss", "Mrs", "Master"]
train["Title"] = train["Title"].apply(lambda x: x if x in common_titles else "Other")
test["Title"] = test["Title"].apply(lambda x: x if x in common_titles else "Other")
train = pd.get_dummies(train, columns=["Title"])
test = pd.get_dummies(test, columns=["Title"])

# Family Size
train["FamilySize"] = train["SibSp"] + train["Parch"] + 1
test["FamilySize"] = test["SibSp"] + test["Parch"] + 1

# Updated feature list
features = [
    "Pclass",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "Sex_female",
    "Sex_male",
    "Embarked_C",
    "Embarked_Q",
    "Embarked_S",
    "FamilySize",
    "Title_Mr",
    "Title_Miss",
    "Title_Mrs",
    "Title_Master",
    "Title_Other",
]

X_train = train[features]
y_train = train["Survived"]
X_test = test[features]

# Model training with Random Forest
# clf = RandomForestClassifier(n_estimators=100)
# clf.fit(X_train, y_train)
# predictions = clf.predict(X_test)

# Hyperparameter Tuning using GridSearchCV
param_grid = {
    "n_estimators": [50, 100, 200],
    "max_features": ["auto", "sqrt"],
    "max_depth": [None, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
}
# grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, n_jobs=-1)
# grid_search.fit(X_train, y_train)
# best_clf = grid_search.best_estimator_
# grid_predictions = best_clf.predict(X_test)

# # Cross-Validation
# scores = cross_val_score(RandomForestClassifier(), X_train, y_train, cv=5)
# print("Cross-validation scores:", scores)

# Trying Different Models

# Gradient Boosted Trees (commented out for now)
gbc = GradientBoostingClassifier()
gbc.fit(X_train, y_train)
gbc_predictions = gbc.predict(X_test)

# Neural Networks (commented out for now)
# nn = MLPClassifier(hidden_layer_sizes=(100,50), max_iter=1000)
# nn.fit(X_train, y_train)
# nn_predictions = nn.predict(X_test)

# Advanced Techniques

# Stacking (commented out for now)
# estimators = [
#    ('rf', RandomForestClassifier(n_estimators=100)),
#    ('gbc', GradientBoostingClassifier())
# ]
# stack_clf = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
# stack_clf.fit(X_train, y_train)
# stack_predictions = stack_clf.predict(X_test)

# Output predictions using the best model
output = pd.DataFrame({"PassengerId": test.PassengerId, "Survived": gbc_predictions})
output.to_csv("predictions2.csv", index=False)
