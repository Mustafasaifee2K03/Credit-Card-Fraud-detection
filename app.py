import pandas as pd
from utils.preprocessing import preprocess_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import pickle
df=pd.read_csv("E:\codeclause DS internship\credit card fraud proj\dataset\dataset.csv")
df=preprocess_data(df)
print(df.shape)
#Separate features and targets
# features
X = df.drop('fraud', axis=1)  # Features
# target
y = df['fraud']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

# Create a Random Forest classifier with the desired number of trees (n_estimators)
# You can also set other hyperparameters such as max_depth, min_samples_split, etc.
max_depth_=[10,20,30,None]
min_samples_split_=[50,100,150,200]
for i in max_depth_:
    rf_classifier = RandomForestClassifier(n_estimators=110, random_state=42,max_depth=i)
    rf_classifier.fit(X_train, y_train)
    y_pred = rf_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    # Print the evaluation metrics
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print("-----------------------------------")
#according to the output we see that among the max_depth_ provided in the matrix, max_depth=10 is performing well on the data according to metrics
# now repeat the same for the hyperparameter min_samples_split
for j in min_samples_split_:
    rf_classifier = RandomForestClassifier(n_estimators=110, random_state=42,max_depth=10,min_samples_split=j)
    rf_classifier.fit(X_train, y_train)
    y_pred = rf_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    # Print the evaluation metrics
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print("-----------------------------------")
# in case of min_samples_split, the value min_samples_split=50 performs pretty well

# now it's time to train model on these 2 perfect hyperparameters
rf_classifier = RandomForestClassifier(n_estimators=110, random_state=42,max_depth=10,min_samples_split=50)
rf_classifier.fit(X_train, y_train)
y_pred = rf_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
# Print the evaluation metrics
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
# Print the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)
print("as we have seen in the confusion metrics, only ",conf_matrix[0,1]," are False Negatives and ",conf_matrix[1,0]," are False Negatives")
pickle.dump(rf_classifier,open("model.pkl","wb"))