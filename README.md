# ML-projects
This repository contains three machine learning projects, including regression analysis, classification models, and unsupervised learning methods. The projects implement linear models, tree-based algorithms, feature engineering, and model evaluation techniques using Python and sci-kit.

## Overview

This portfolio showcases my skills in:
- Data preprocessing and exploratory data analysis
- Linear and regularized regression models
- Classification using traditional and tree-based methods
- Unsupervised learning methods, including clustering
- Model evaluation and hyperparameter tuning
- Feature engineering and selection

## Project Structure 

ML-projects/
│
├── Project-1.ipynb          # Regression Analysis on Seoul Bike-Sharing Data
├── data_Project-1.csv       # Seoul Bike-Sharing dataset
├── variables_Project-1.txt  # Variable descriptions for Project 1
│
├── Project-2.ipynb          # Classification of Online News Popularity
├── data_Project-2.csv       # Online News Popularity dataset
├── variables_Project-2      # Variable descriptions for Project 2
│
├── Project-3.ipynb          # Hotel Review Analysis
├── variables_Project-3.pdf  # Variable descriptions for Project 3
│
└── README.md                # Repository documentation

## Data Files

- Each project has its corresponding data file(s) and variable descriptions
- For the Hotel Review dataset, download from: https://surfdrive.surf.nl/files/index.php/s/cy3NzaikRxHOXy2?path=%2FHotel%20Review%20Project#editor

## Projects

## Assignment 1: Regression Analysis on Seoul Bike-Sharing Data

### Overview

This project implements and compares various regression models to predict hourly bike rentals in the Seoul Bike-Sharing system. The analysis includes comprehensive data preprocessing, exploratory data analysis with visualizations, and hyperparameter tuning.

### Dataset

The dataset contains weather conditions and temporal information to predict the number of bikes rented each hour:

- **Date**: Year-month-day
- **Hour**: Hour of the day (0-23)
- **Temperature**: Temperature in Celsius
- **Humidity**: Humidity percentage
- **Wind speed**: Wind speed in m/s
- **Visibility**: Visibility in meters
- **Dew point temperature**: Dew point temperature in Celsius
- **Solar radiation**: Solar radiation in MJ/m2
- **Rainfall**: Rainfall in mm
- **Snowfall**: Snowfall in cm
- **Target Variable**: Rented Bike Count

### Models Implemented

1. **Baseline Mean Predictor**: A simple model that predicts the mean rental count for all instances
2. **Ordinary Least-Squares (OLS) Linear Regression**: Basic linear regression model
3. **Ridge Regression**: Linear regression with L2 regularization
4. **Lasso Regression**: Linear regression with L1 regularization

### Key Findings

- Temperature emerged as the strongest predictor with a coefficient of 319.03
- Hour of the day showed a strong positive influence (coefficient: 195.46)
- Humidity had a strong negative impact (coefficient: -172.30)
- The OLS model achieved an R² score of 0.465, explaining about 46.5% of the variance
- Ridge and Lasso models performed similarly to OLS, suggesting minimal multicollinearity in the dataset
- Adding temporal features (month, day of week, weekend indicator, seasonal effects) improved model performance, increasing R² to 0.51

![Unknown-2](https://github.com/user-attachments/assets/fab8ac62-b174-4d18-8fc0-a8e687aaf1d7)
![Unknown-3](https://github.com/user-attachments/assets/5df10103-889f-475a-8d87-46ce28e7a8b3)

### Feature Engineering

Additional temporal features were extracted to enhance model performance:
- Month (1-12): Captures monthly seasonality patterns
- Day of Week (0-6): Captures weekly patterns
- Is Weekend (0/1): Binary indicator for weekend days
- Seasonal Effects: One-hot encoded seasons (Winter, Spring, Summer, Fall)

![Unknown-1](https://github.com/user-attachments/assets/8d3fa77a-6cec-4ed8-8120-e5cc16b851d6)

### Interaction Effects

Interaction terms were created to capture complex relationships between features:
- Temperature × Humidity
- Temperature × Rainfall
- Temperature × Hour
- Hour × Weekend
- Temperature × Season

### Results

| Model | Original R² | Enhanced R² (with temporal features) | Original MSE | Enhanced MSE |
|-------|-------------|-------------------------------------|--------------|--------------|
| OLS   | 0.465       | 0.511                               | 222,880.72   | 203,644.03   |
| Ridge | 0.465       | 0.511                               | 222,880.87   | 203,688.98   |
| Lasso | 0.465       | 0.510                               | 222,931.41   | 204,024.40   |

### Conclusion

The analysis revealed that while linear models captured significant patterns in bike rental behavior, there might be non-linear relationships that could be explored through more sophisticated modeling approaches. The addition of temporal features and interaction terms significantly improved model performance.

## Assignment 2: Classification of Online News Popularity

This assignment focused on analyzing and predicting online news article popularity by comprehensively exploring various classification techniques. Using a dataset of 39,000 news articles with 58 features, including metrics like word count, number of links, and content polarity, we developed models to classify articles as popular or unpopular based on a 1,400-shares threshold.

### Dataset

The dataset contains various features related to online news articles:
- Word-based features (e.g., number of words in title, content)
- Link-based features (e.g., number of links, images)
- Digital media features (e.g., number of videos, images)
- Metadata features (e.g., day of week, data channel)
- Sentiment-related features (e.g., polarity, subjectivity)
- Target variable: Article shares (transformed into a binary popularity indicator)

### Part A: Traditional Classifiers

This part examines linear classifiers (Logistic Regression, linear SVC, and K-Nearest Neighbors), emphasising feature scaling impacts in predicting online news popularity.

#### Models Implemented

1. **Logistic Regression**: A linear model for binary classification
2. **Linear SVC**: Support Vector Classification with a linear kernel
3. **K-Nearest Neighbors**: A non-parametric method for classification

#### Feature Scaling Impact

Each model was trained and evaluated both with and without feature scaling to understand the impact of standardization on model performance.

##### Results Summary

| Model | Without Scaling (Accuracy) | With Scaling (Accuracy) | Improvement |
|-------|----------------------------|-------------------------|-------------|
| Logistic Regression | 60.84% | 65.42% | +4.58% |
| Linear SVC | 60.61% | 65.33% | +4.72% |
| KNN | 57.76% | 62.95% | +5.19% |

#### Key Findings

- Feature scaling consistently improved performance across all models
- Logistic Regression and Linear SVC showed the best results (accuracy ~65.4% and ~65.3% respectively)
- KNN, while benefiting from scaling, performed slightly lower at 62.95% accuracy
- Without scaling, model performance dropped significantly, particularly for KNN
- Logistic Regression emerged as the most effective classifier, combining good performance with computational efficiency

#### Hyperparameter Optimization

- **Logistic Regression**: Optimal C=0.1 (with and without scaling)
- **Linear SVC**: Optimal C=0.001 (without scaling), C=0.01 (with scaling)
- **KNN**: Optimal n_neighbors=15 (with and without scaling)

#### Conclusion

The results highlight the importance of proper feature preprocessing for these algorithms. Logistic Regression emerged as the most effective classifier for news popularity prediction, combining good performance with computational efficiency.

### Part B: Tree-based Classifiers

This part investigates tree-based methods (Decision Trees and Random Forests), including robustness analysis under varying noise conditions for predicting online news popularity.

#### Models Implemented

1. **Decision Tree**: A non-parametric supervised learning method
2. **Random Forest**: An ensemble learning method that operates by constructing multiple decision trees

#### Hyperparameter Optimization

##### Decision Tree
- Optimal max_depth: 7
- Optimal max_features: None (use all features)
- Overfitting begins at depth: 20

##### Random Forest
- Optimal parameters: {'max_depth': None, 'max_features': 'log2', 'n_estimators': 300}
- Best cross-validation accuracy: 0.6724

#### Model Performance

| Model | Accuracy | Macro Precision | Macro Recall | F1 Score |
|-------|----------|----------------|-------------|----------|
| Decision Tree | 0.6383 | 0.6368 | 0.6326 | 0.6355 |
| Random Forest | 0.6640 | 0.6626 | 0.6597 | 0.6625 |

The Random Forest model showed a 2.57% improvement over the Decision Tree model.

#### Feature Importance

Both models identified similar important features, with the Random Forest providing more stable feature importance estimates.

_Feature Importance of Decision Tree_
![Unknown-1](https://github.com/user-attachments/assets/5581e7aa-a499-49a1-b971-a0de45bf0e88)

_Feature Importance of Random Forest_
![Unknown-2](https://github.com/user-attachments/assets/ede755bf-d1c3-412d-93d2-2d4d7b78dd2e)

#### Robustness Testing

Models were tested with increasing levels of Gaussian noise (standard deviations of 0, 0.1, 0.2, 0.5, and 1.0) to evaluate their robustness.

##### Results

| Noise Level | Decision Tree Accuracy | Random Forest Accuracy |
|-------------|------------------------|------------------------|
| 0.0 | 0.6383 | 0.6640 |
| 0.1 | 0.6348 | 0.6519 |
| 0.2 | 0.6239 | 0.6426 |
| 0.5 | 0.5809 | 0.5900 |
| 1.0 | 0.5500 | 0.5737 |

Performance Degradation:
- Decision Tree: 13.83% decrease from noise=0 to noise=1.0
- Random Forest: 13.60% decrease from noise=0 to noise=1.0

#### ROC Curve Analysis

The Random Forest model achieved a higher AUC score compared to the Decision Tree model, indicating better overall performance.

![Unknown](https://github.com/user-attachments/assets/c54f39ac-74c6-446c-8c27-3a1b699cb019)

#### Conclusion

The Random Forest model demonstrated higher accuracy, stability, and robustness against noise compared to the Decision Tree model. While both models experienced performance degradation with increasing noise, Random Forest remained more stable, making it a more reliable choice for this classification task.

## Final Project: Hotel Review Analysis

### Overview

This project analyzes a dataset of 515,000 hotel reviews from across Europe to extract insights about user behavior and hotel performance. The analysis includes exploratory data analysis, classification, regression, and clustering techniques.

### Dataset

The dataset contains reviews of hotels across Europe with the following information:
- Hotel details (name, address, average score)
- Reviewer information (nationality, number of reviews)
- Review content (positive and negative reviews, word counts)
- Review metadata (date, tags, reviewer score)
- Geographical information (latitude, longitude)

Please download the dataset using the following link: https://surfdrive.surf.nl/files/index.php/s/cy3NzaikRxHOXy2?path=%2FHotel%20Review%20Project#editor

### Exploratory Data Analysis

The analysis revealed several interesting patterns:
- Distribution of reviewer scores is right-skewed with most ratings being positive
- Clear differences in rating patterns across nationalities
- Correlation between review length and sentiment
- Geographical patterns in hotel ratings

### Classification Task

A Random Forest Classifier was trained to predict whether a review would have a higher or lower score than average.

#### Results
- Accuracy: 0.74
- F1 Score: 0.74
- Precision: 0.74
- Recall: 0.74

#### Feature Importance
The most important features for predicting high scores were:
1. Review length
2. Positive word count
3. Negative word count
4. Trip type (business vs. leisure)
5. Reviewer nationality

### Hyperparameter Tuning

Grid search was used to find the optimal hyperparameters for the Random Forest model:
- n_estimators: 200
- max_depth: 20
- min_samples_split: 5
- min_samples_leaf: 2
- bootstrap: True

### Regression Analysis

A Random Forest Regressor was used to predict the exact reviewer score based on review characteristics and metadata.

### Geographical Analysis

Hotels were clustered based on their geographical location, revealing interesting patterns in the distribution of highly-rated hotels across Europe.

![Unknown-3](https://github.com/user-attachments/assets/cab27e2a-d03e-4b0a-a2be-311d3ded8d1d)

![Unknown-5](https://github.com/user-attachments/assets/37d66b65-d791-4581-a23e-aedfec88102a)

### Conclusion

The analysis provides valuable insights for both travelers and hotel managers:
- Certain nationalities tend to give consistently higher or lower ratings
- Specific tags and trip types correlate strongly with higher ratings
- Review length and sentiment are strong predictors of the overall score
- There are geographical clusters of highly-rated hotels in specific regions

These findings can help hotels improve their services and help travelers make more informed decisions when booking accommodations.

## Skills Demonstrated

- **Programming Languages:** Python
- **Libraries & Frameworks:** scikit-learn, pandas, numpy, matplotlib, seaborn
- **Machine Learning Algorithms:** 
  - Linear Regression, Ridge, Lasso
  - Logistic Regression, SVM, KNN
  - Decision Trees, Random Forests
- **Data Analysis:** 
  - Exploratory Data Analysis
  - Feature Engineering
  - Feature Selection
- **Model Evaluation:** 
  - Cross-validation
  - Hyperparameter Tuning
  - Performance Metrics

## Getting Started

### Prerequisites
```python
# Required packages
pandas
numpy
matplotlib
seaborn
scikit-learn 
