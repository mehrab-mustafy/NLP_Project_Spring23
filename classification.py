import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

df = pd.read_csv('data.csv')

# Step 1: Prepare the Data
X = df[['sentence_count', 'spelling_mistakes', 'agreement', 'verbs', 'c3']]
y = df['grade'].map({'high': 1, 'low': 0})  # Convert labels to binary

# Step 2: Split Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Define and Train the Model
# MLP:
mlp_classifier = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', max_iter=1000)
mlp_classifier.fit(X_train, y_train)
y_pred = mlp_classifier.predict(X_test)
accuracy1 = accuracy_score(y_test, y_pred)
precision1 = precision_score(y_test, y_pred)
recall1 = recall_score(y_test, y_pred)
f1_1 = f1_score(y_test, y_pred)
print('MLP:')
print(f'Accuracy: {accuracy1}, Precision: {precision1}, Recall: {recall1}, F1 Score: {f1_1}')
print()

# Logistic Regression
logistic_regression = LogisticRegression()
logistic_regression.fit(X_train, y_train)
y_pred = logistic_regression.predict(X_test)
accuracy2 = accuracy_score(y_test, y_pred)
precision2 = precision_score(y_test, y_pred)
recall2 = recall_score(y_test, y_pred)
f1_2 = f1_score(y_test, y_pred)
print('Logistic Regression:')
print(f'Accuracy: {accuracy2}, Precision: {precision2}, Recall: {recall2}, F1 Score: {f1_2}')
print()

# Naive Bayes
naive_bayes = GaussianNB()
naive_bayes.fit(X_train, y_train)
y_pred = naive_bayes.predict(X_test)
accuracy3 = accuracy_score(y_test, y_pred)
precision3 = precision_score(y_test, y_pred)
recall3 = recall_score(y_test, y_pred)
f1_3 = f1_score(y_test, y_pred)
print('Naive Bayes')
print(f'Accuracy: {accuracy3}, Precision: {precision3}, Recall: {recall3}, F1 Score: {f1_3}')
print()