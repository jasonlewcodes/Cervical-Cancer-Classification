import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import ADASYN

# This function displays statistics for the model's performance.
def print_statistics(predictions, y_test, model):
  class_report = classification_report(y_test, predictions)
  print("SKLearn Classification Report")
  print(class_report)
  print("{} Model Accuracy:".format(model), str((accuracy_score(predictions, y_test) * 100).round(1)) + '%')

  ax = plt.subplot()
  con_mat = confusion_matrix(y_test, predictions)
  sns.heatmap(con_mat, annot = True, fmt = 'g', ax=ax)
  ax.set_title('Confusion Matrix')
  ax.set_ylabel('True label')
  ax.set_xlabel('Predicted label')
  labels = ['No cancer', 'Cancer']
  ax.xaxis.set_ticklabels(labels)
  ax.yaxis.set_ticklabels(labels)
  plt.show()

# Logistic Regression model using sklearn.
def logistic_regression(X, y):
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25) # 75% training size, 25% test size
  model = LogisticRegression(max_iter=1000)
  model.fit(X_train, y_train)
  predictions = model.predict(X_test)
  print_statistics(predictions, y_test, "Logistic Regression")

# Gaussian Naive Bayes model using sklearn.
def gaussian_naive_bayes(X, y):
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25) # 75% training size, 25% test size
  model = GaussianNB()
  model.fit(X_train, y_train)
  predictions = model.predict(X_test)
  print_statistics(predictions, y_test, "Gaussian Naive Bayes")

class_df = pd.read_csv("cervical_cancer_risk_factors.csv")
drop_list = ['STDs: Time since last diagnosis', 'STDs: Time since first diagnosis'] # Drop these columns because significant number of data points are missing.
class_df = class_df.drop(columns = drop_list)
class_df = class_df.replace('?', np.nan) # In this data set, '?' denotes missing data. Replace '?' with NaN.
class_df = class_df.apply(pd.to_numeric) # Make all columns numeric.

# The code below displays the correlation matrix for this data set.

# corr_matrix = class_df.corr()
# plt.figure(figsize=(34,34))
# sns.heatmap(corr_matrix, annot = True)
# plt.show()

col_means = class_df.mean() # Calculate mean of each column.
class_df = class_df.fillna(col_means) # Replace NaN values with the mean of the column.

y = class_df['Dx:Cancer'] # These are the classifications: 0 = no cancer, 1 = cancer
X = class_df.drop(columns = ['Dx:Cancer'])

# There are very few instances where patients have cancer in this dataset. ADASYN generates synthetic data to supplement the existing data.
adasyn = ADASYN(random_state=42) 
x_adasyn, y_adasyn = adasyn.fit_resample(X,y)

logistic_regression(x_adasyn, y_adasyn)
gaussian_naive_bayes(x_adasyn, y_adasyn) # The Gaussian Naive Bayes model performs poor due to variable correlation.