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


