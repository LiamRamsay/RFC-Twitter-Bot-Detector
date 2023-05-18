# Import pandas to read .csv dataset + remove string from float
import pandas as pd
data = pd.read_csv("twibot-22.csv")
data['id'] = data['id'].str[1:]

# Import Numpy to split labels and features in the dataset
import numpy as np
data = np.array(data)
features = data[0: , :-1]
labels = data[0: ,-1]

# Create encoder then encode labels
from sklearn.preprocessing import LabelEncoder
myEncoder = LabelEncoder()
encLabels = myEncoder.fit_transform(labels)

# Split features and labels into training and test
from sklearn.model_selection import train_test_split
train_features, test_features, train_labels, test_labels = train_test_split(features, encLabels, test_size=0.2) 

# Import RFC algorithm and create model
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(train_features, train_labels)

# Check the training accuracy
trainAcc = model.score(train_features, train_labels)
print("Training Accuracy: " + str(trainAcc*100))

# Calculating accuracy score by getting predictions first
from sklearn.metrics import accuracy_score
trainPred = model.predict(train_features)
trainAcc2 = accuracy_score(train_labels, trainPred)
print("Training Accuracy Prediction: " + str(trainAcc2*100))

# Check the test accuracy
testAcc = model.score(test_features, test_labels)
print("Testing Accuracy: " + str(testAcc*100))

# Calculating accuracy score by getting predictions first
testPred = model.predict(test_features)
testAcc2 = accuracy_score(test_labels, testPred)
print("Testing Accuracy Prediction: " + str(testAcc2*100))

# Calculate precision score
from sklearn.metrics import precision_score
prec = precision_score(test_labels, testPred)
print("Precision Score: " + str(prec*100))

# Calculate recall score
from sklearn.metrics import recall_score
rec = recall_score(test_labels, testPred)
print("Recall Score: " + str(rec*100))

# Calculate f1 score
from sklearn.metrics import f1_score
f1 = f1_score(test_labels, testPred)
print("f1 Score: " + str(f1*100))

# Calculate class report
from sklearn.metrics import classification_report
print("Class report is : ")
print(classification_report(test_labels, testPred))

# Create and display confusion matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
ConfusionMatrixDisplay.from_predictions(test_labels, testPred)
plt.show()

# Create and dsiplay ROC curve
from sklearn.metrics import RocCurveDisplay
ax = plt.gca()
rfc_disp = RocCurveDisplay.from_estimator(model, test_features, test_labels, ax=ax, alpha=0.8)
plt.show()
