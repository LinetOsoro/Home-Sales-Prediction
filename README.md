# Home-Sales-Prediction

 
# Authors: Jennifer Savage, Madeline Riley, Linet Osoro


# Overview
This project aimed to predict housing prices using Zillow information and machine learning techniques. We sought to determine if data from Zillow could be effectively processed through various machine learning algorithms to predict home prices accurately. Random Forest, Decision Tree, Support Vector Machine, and Linear Regression models were utilized for this analysis.


# Methodology
Data Loading and Preprocessing
Read the CSV file into a Spark DataFrame
Dropped non-beneficial columns and handled missing values
Converted data types and encoded categorical variables
Converted Spark DataFrame into a pandas DataFrame
Feature Selection and Target Definition: Our Targer was Price, Features were State, Bedroom, Bathroom, Area, ConvertedLot. 
Defined features (X) and the target variable (y) for the machine learning models.
Train-Test Split the dataset into training and testing sets
Feature StandardScaler to ensure mean 0 and variance 1
Model Training and Prediction 
Model Testing. 

# Analysis
The data was cleaned, and different machine learning models were trained and tested to predict home prices. We evaluated the performance of each model and documented their accuracy percentages. The analysis provided insights into the effectiveness of various machine learning methods in predicting housing prices.

### Random Forest Regression Model
This model seems to have moderate performance. With the R^2 score of approximately 0.65 explains approximately the variance in the target variable. The model suggests moderate prediction.

### Decision Tree Regression Model  
R^2 score being negative (50%) suggests that the model performs worse than a horizontal line, which is not uncommon in some scenarios but generally indicates a poor fit.

### Linear Regression Model 
R^2 score for training was 41% and test was 30%,  a high MSE value indicates that the model's predictions are far from the actual values on average.The model performed poorly. 

### Support Vector Machine(SVM)
A lower MSE indicates better model performance. A lower RMSE values indicate better model performance. An R^2 score of 0.46 suggests that the model explains approximately 46% of the variance in the target variable. Std measures the dispersion of the actual target values around the mean. It provides context for interpreting the MSE and RMSE values.

## Collaboration
This project was a collaborative effort among team members with the assistance of tutors. The code is original and was developed using a combination of class assignments, tutor guidance, and instructor support.

## Sources used are as follows: 

[Forecasting the Zillow Home Value Index Using Three ML Techniques](https://nycdatascience.com/blog/student-works/capstone/forecasting-the-zillow-home-value-index-using-three-ml-techniques) 

[End To End Machine Learning Project To Predict Housing Prices In California](https://lukeclarke12.medium.com/end-to-end-machine-learning-project-to-predict-housing-prices-in-california-e58cb10b2005)

[10 Real Estate Data Science Projects](https://www.interviewquery.com/p/real-estate-data-science-projects)

[Zillow Economic Data](https://www.kaggle.com/datasets/zillow/zecon)

[Housing Data](https://www.zillow.com/research/data)

[Zillow House Price](https://www.kaggle.com/datasets/paultimothymooney/zillow-house-price-data)

[A Practical Approach to Linear Regression in Machine Learning](https://towardsdatascience.com/linear-regression-5100fe32993a)

[Keras Applications](https://keras.io/api/applications/#usage-examples-for-image-classification-models)

[VGG16 and VGG19](https://keras.io/api/applications/vgg/#vgg19-function)

[Treemaps](https://plotly.com/python/treemaps/)

[graph](https://plotly.com/python/setting-graph-size/)

[Census for region](https://github.com/cphalpert/census-regions/blob/master/us%20census%20bureau%20regions%20and%20divisions.csv)

[Deep Neural](https://towardsdatascience.com/deep-neural-networks-for-regression-problems-81321897ca33)

[Geeksforgeeks](https://www.geeksforgeeks.org/random-forest-algorithm-in-machine-learning/)

[Decision Tree](https://www.geeksforgeeks.org/decision-tree/)

[An Intuitive Explanation of Convolutional Neural Networks](https://ujjwalkarn.me/2016/08/11/intuitive-explanation-convnets/)

[House Price Prediction](https://www.geeksforgeeks.org/house-price-prediction-using-machine-learning-in-python/)

[House Price Prediction using Machine Learning in Python](https://www.geeksforgeeks.org/house-price-prediction-using-machine-learning-in-python/)


## Repository
The code and files for this project are available on GitHub. https://github.com/LinetOsoro/Home-Sales-Prediction 


