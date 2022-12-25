# Cervical-Cancer-Classification
Machine learning model (Logistic Regression &amp; Gaussian Naive Bayes) in Python that classifies cervical cancer based on risk factors.

The data set can be [found here](https://www.kaggle.com/datasets/loveall/cervical-cancer-risk-classification) on Kaggle. According to the authors, the data set is obtained from UCI Repository. This dataset contains 30+ risk factors for cervical cancer with the result in column `Dx:Cancer`: 0 = no cancer and 1 = cancer. I use a Logistic Regression model &amp; a Gaussian Naive Bayes model to classify whether or not a set of risk factors results in a cervical cancer diagnosis.

The highest observed accuracy using Logistic Regression was 100%. Using Gaussian Naive Bayes, the highest observed accuracy was 65%, likely due to the lack of independence between risk factors.

Libraries used: Numpy, Pandas, Seaborn, Pyplot, SKLearn, and Imblearn.

See Python file for more detail and annotations.
