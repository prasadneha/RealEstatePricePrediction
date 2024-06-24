# Real Estate Price Prediction Project
## Project Overview
This project aims to predict the median value of owner-occupied homes in various suburbs of Boston. The dataset includes various features such as crime rate, number of rooms, and property tax rate, which are used to train different machine learning models to make accurate price predictions.

## Dataset
The dataset used in this project is stored in the file Real Estate Price Prediction.csv. It contains the following columns:

- **CRIM**: Per capita crime rate by town.
- **ZN**: Proportion of residential land zoned for lots over 25,000 sq. ft.
- **INDUS**: Proportion of non-retail business acres per town.
- **CHAS**: Charles River dummy variable (1 if tract bounds river; 0 otherwise).
- **NOX**: Nitrogen oxides concentration (parts per 10 million).
- **RM**: Average number of rooms per dwelling.
- **AGE**: Proportion of owner-occupied units built before 1940.
- **DIS**: Weighted distances to five Boston employment centers.
- **RAD**: Index of accessibility to radial highways.
- **TAX**: Full-value property tax rate per $10,000.
- **PTRATIO**: Pupil-teacher ratio by town.
- **B**: 1000(Bk - 0.63)^2 where Bk is the proportion of Black residents by town.
- **LSTAT**: Percentage of lower status of the population.
- **MEDV**: Median value of owner-occupied homes in $1000s.
## Project Structure
The project consists of the following key components:
### Jupyter Notebooks
1. Model Usage.ipynb:
  - Loads a pre-trained model (Dragon.joblib) and demonstrates how to make predictions using a sample feature array.
2. Dragon Real Estates.ipynb:
  - Comprehensive notebook that includes data loading, exploration, preprocessing, model training, evaluation, and saving/loading models.
### Output Metrics
The performance of different models is evaluated and stored in Real Estate Models Outputs.txt:

- Decision Tree:
  - Mean: 4.1895
  - Standard Deviation: 0.8481
- Linear Regression:
  - Mean: 4.2219
  - Standard Deviation: 0.7520
- Random Forest Regression:
  - Mean: 3.4947
  - Standard Deviation: 0.7620
### Data Preprocessing
The data preprocessing pipeline handles missing values and scales features. Key components include:

- Imputation: Using `SimpleImputer` to fill missing values.
- Scaling: Using `StandardScaler` to normalize features.
### Model Training and Evaluation
The following models are trained and evaluated:

- Linear Regression
- Decision Tree Regression
- Random Forest Regression
Evaluation metrics used:

- Root Mean Squared Error (RMSE)
- Cross-Validation Scores
### Model Saving and Loading
Models are saved using `joblib` and can be loaded for making predictions. An example of how to save and load models is included.

## Usage
To use the notebooks and run the project:

1. Clone the repository.
2. Ensure all necessary libraries are installed (e.g., pandas, scikit-learn, joblib, matplotlib, numpy).
3. Open the Jupyter notebooks and execute the cells to see the full workflow from data loading to model evaluation.


## Conclusion
This project demonstrates the complete workflow for building and evaluating machine learning models to predict real estate prices. It includes data preprocessing, feature engineering, model training, evaluation, and deployment steps. The Random Forest Regression model showed the best performance based on the evaluation metrics.

