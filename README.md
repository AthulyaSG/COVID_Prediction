# COVID_Prediction

The key aim of this endeavor is to employ Machine Learning (ML) for determining whether a patient has contracted COVID or not. The data source for this research was procured from the Kaggle website. In order to optimize the model's accuracy, feature selection was conducted by picking relevant variables, namely, blood and virus components. This was followed by various data processing techniques, including encoding, imputation, PCA, and Synthetic Minority Over-sampling Technique (SMOTE). Finally, the most effective model was selected, and its performance was evaluated.

## Implementation

The implementation process began with importing the necessary libraries and loading the dataset. The distribution of positive and negative exam results was checked and plotted using a bar chart (Figure 1). The percentage of missing values in each column was calculated using isna() and sum() functions. Columns with missing data between 86% and 90% were selected and assigned to a list called "blood". A heatmap was created to visualize the correlation between columns in the "blood" list (Figure 2), and histograms were plotted for each column (Figure 3). Columns with missing values greater than 70% and less than 80% were selected and assigned to a list called "virus". Another heatmap was created to visualize the relationship between columns in the "virus" list and the target variable (Figure 4).

Figure 1. Bar chart to find the distribution of positive and negative exam results.

![image](https://github.com/user-attachments/assets/9cdec19b-5bcf-470b-acc8-12daf30cbc85)


Figure 2. Heatmap to visualize the correlation between blood columns.

![image](https://github.com/user-attachments/assets/8abad7d9-1a3a-4d17-836e-52fb6a2983ea)


Figure 3: Histogram for hematocrit.

![image](https://github.com/user-attachments/assets/5ef8a93c-af63-4abc-9206-dbae35a01796)


Figure 4: Heatmap to visualize the correlation between viral columns.

![image](https://github.com/user-attachments/assets/5dc0f836-3718-4400-9614-8cdedc574ece)


### Preprocessing

In Method 1, categorical variables were encoded using the encode function for the combined blood and virus components. Missing values were imputed using the median value of each column, and two functions were chained. The data was then split into training and testing sets, scaled using StandardScaler function, and PCA was applied to reduce the dimensionality of the data.

In Method 2, the same steps were implemented as in Method 1, except for the utilization of StandardScaler() and PCA(). SMOTE analysis was used instead to balance the imbalanced data.

Method 3 uses the same steps as Method 2, with the sole difference being that KNNImputer() is used to impute missing values instead of using fillna() with the median.

In this Method 4, only blood components are taken into consideration, in contrast to other methods where both blood and virus are considered. Rather than filling in missing values, dropna() is used to eliminate the null values.


### Models

The following models have been used (Parashar, 2022) for all four methods: K-Nearest Neighbors (KNN) is an algorithm used for classification, which determines the class of new data points by measuring their distance from the closest classified points. A decision tree is a hierarchical structure used for classification, consisting of a set of conditions that guide the path of a sample through the tree until it reaches the bottom. Support Vector Machines (SVMs) aim to find a hyperplane in an n-dimensional space, where n represents the number of features, which can separate data points into different classes. Logistic Regression is a classification technique that aims to find the best-fitted curve for data. It uses a linear model for binary classification, which distinguishes between positive and negative classes. Random Forest is an ensemble technique composed of multiple decision trees. Each tree in a random forest is trained on a different subset of data, and the prediction with the most votes is chosen as the random forest prediction. Gradient Boosting constructs multiple decision trees in a step-by-step manner, where each tree learns from the mistakes of the previous trees.


### Evaluation of model performance

To assess a model's performance (Beheshti, 2022), 10-fold cross-validation is employed in Methods 1 and 4, by dividing the data into ten parts or "folds," with nine of them used for training and the remaining one for testing. GridSearchCV is utilized in Method 2 to fine-tune a model's hyperparameters by thoroughly exploring a predefined set of values and selecting the best combination that yields the highest performance. Method 4 did not involve the evaluation of model performance since none of the models showed satisfactory results.


## Results

In this work, various ML techniques have been implemented to diagnose patients with COVID.

### Method 1

Logistic Regression was the best model with an accuracy of 90.16%, F1 score of 85.50%, recall score of 90.16%, and precision of 81.30% (Table 1). The learning curves of all models were plotted (Figure 5). The performance of Logistic Regression was evaluated using 10-fold cross-validation (average cross-validation score: 0.90). The logistic regression model’s confusion matrix revealed that it correctly predicted all instances in the negative class but failed to predict any instance in the positive class (Figure 6), indicating that Logistic Regression is not the best model to predict COVID-positive patients.

Table 1: Accuracy, F1 score, Recall, and precision of all six models.

![image](https://github.com/user-attachments/assets/af7fdbf8-6a4b-4895-8777-265666e68c1e)

Figure 5: Learning curve of Logistic Regression.

![image](https://github.com/user-attachments/assets/cd0c65a4-eb82-417d-a5df-b1a7f871a799)


Figure 6: Confusion matrix for Logistic Regression.

![image](https://github.com/user-attachments/assets/35586d8d-4245-474c-a1d0-95baef9160bb)


### Method 2

In addition to the previous findings, the SMOTE analysis is used to address the issue of imbalanced data, and Random Forest outperforms other models with an accuracy of 0.598, F1 score of 0.527, recall of 0.598, and precision score of 0.738 (Table 2). The Random Forest model's confusion matrix indicates that there were 250 true negatives, 21 false negatives, 1212 true positives, and 958 false positives. This means that the model correctly identified 250 negative cases and 1212 positive cases but misclassified 21 negative cases as positive and 958 positive cases as negative (Figure 7). Hyperparameter tuning was performed using GridSearchCV() to find the best hyperparameters, and the best parameters are passed to the Random Forest model, resulting in an accuracy score of 27% for the best model.


Table 2: Accuracy, F1 score, Recall, and precision of all six models using SMOTE analysis.

![image](https://github.com/user-attachments/assets/1cd9e734-6bbb-48c5-8670-d0b3bbca9ce6)


Figure 7: Confusion matrix of Random Forest.

![image](https://github.com/user-attachments/assets/caba97cf-0f64-4888-8141-ca6fbeaa37b0)


### Method 3

Therefore, in this method, the KNNImputer is employed to impute missing values. The SMOTE analysis was conducted after the imputation stage, followed by the training and evaluation of six models. The Logistic Regression model has a high accuracy of (0.803), but low F1 score (0.165), recall (0.198), and precision (0.141) (Table 3) indicate that it is not the best model to diagnose COVID patients. None of the models performed well in this method, and hence their performance was not evaluated.

Table 3: Accuracy, confusion matrix, F1 score, recall, and precision for six models with KNNImputer() and SMOTE analysis.

![image](https://github.com/user-attachments/assets/6762c4b9-9479-4942-947d-2fb16563ad4c)


### Method 4

Lastly, the dropna() function is used to remove all missing values. Gradient Boosting model appears to be the best model as its accuracy, F1 score, recall, and precision values are 88.28%, 86.85%, 88.28%, and 86.80%, respectively (Table 4). Confusion matrix correctly identified 92 instances as positive and 6 instances as negative, while making 10 false-negative predictions and 3 false-positive predictions (Figure 8). To evaluate the performance of Gradient Boosting, 10-fold cross-validation is performed, where the average cross-validation was 87%.

Table 4: Accuracy, F1 score, Recall, and precision of all six models in which all null values are removed.

![image](https://github.com/user-attachments/assets/6d20540d-2692-413b-b1de-fed01b93512d)


Figure 8: Confusion matrix for Gradient Boosting.

![image](https://github.com/user-attachments/assets/9b586bdb-54bd-49a2-bb00-2ff8190cd380)


## Conclusions

The objective of the study was to evaluate the diagnostic accuracy of six predictive models, namely Logistic Regression, K Nearest Neighbors, Decision Tree, Random Forest, AdaBoost, and Gradient Boosting, for identifying patients with COVID. The data preprocessing involved two approaches, namely imputation and elimination of rows and columns containing missing values. The performance of the models was evaluated using 10-fold cross-validation and GridSearchCV(). Imputation of missing values resulted in low accuracy, precision, recall, and F1 score or confusion matrix, even after balancing the dataset with SMOTE analysis. However, the models performed well when all the missing values were removed from the dataset. Specifically, Gradient Boosting was the most effective model, achieving an accuracy rate of 88.28%. Overall, the study showed that different machine learning models performed differently in diagnosing COVID patients. Further optimization and improvement of the models are needed to increase their accuracy and performance.

This study has a few limitations that need to be acknowledged. Firstly, the dataset used in this study only includes blood and virus components, and therefore, the results may not be generalizable to other parameters. Furthermore, imputing missing values can result in erroneous values, which can affect the overall accuracy of the results. Similarly, deleting missing values can leave the dataset with insufficient data to process, which can limit the statistical power of the study.


## References
Beheshti, N. (2022). Cross validation and Grid Search: using sklearn’s GridSearchCV on random forest model. https://towardsdatascience.com/cross-validation-and-grid-search-efa64b127c1b
Parashar, A. (2022). Explaining each machine learning model in brief. https://levelup.gitconnected.com/explaining-each-machine-learning-model-in-brief-92f82b41ba71
