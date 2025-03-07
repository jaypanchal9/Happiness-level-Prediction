# Happiness Level Prediction - Kaggle Regression Contest 24S1

## Overview
**Happiness Level Prediction** is a predictive analytics project designed to estimate an individual's happiness level based on responses from a life and well-being survey. The model was developed for the **Kaggle Regression Contest 24S1**, focusing on accurately predicting happiness scores using advanced machine learning techniques.

The project follows a stacked ensemble learning approach to achieve high accuracy, evaluated by the **Root Mean Squared Error (RMSE)** metric.

## Features
- **Stacked Ensemble Learning**: Combines predictions from multiple models for enhanced accuracy.
- **Comprehensive Data Preprocessing**: Includes handling missing values, encoding categorical features, and normalization.
- **Cross-Validation**: Implements 10-fold cross-validation to ensure robust model evaluation.

## Key Questions Addressed
- How accurately can individual happiness scores be predicted from survey data?
- Which machine learning models perform best on this task?
- How does stacking models improve prediction accuracy?

## Motivation
The project aims to explore advanced predictive modeling techniques to effectively analyze subjective survey data, providing insights into factors influencing personal well-being and happiness.

## Intended Audience
- **Data Scientists**: For exploring stacked ensemble techniques in predictive modeling.
- **Researchers**: Interested in analyzing survey data to understand subjective measures like happiness.
- **Students and Educators**: To use as a reference for learning ensemble methods and data preprocessing.

## Technology Stack
- **Programming Language**: R
- **Libraries Used**:
  - `caret`, `randomForest`, `gbm`, `xgboost`, `nnet` for model training and predictions.
  - `data.table` for data manipulation.

## Data Sources
- Training and test datasets (`regression_train.csv`, `regression_test.csv`) included in the repository under the `data/` folder.

## Installation
To run this project locally, ensure you have R installed along with the following libraries:

```R
install.packages(c("data.table", "caret", "randomForest", "gbm", "xgboost", "nnet"))
```

### Running the Project
1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/happiness-prediction.git
   ```
2. Execute the R script to build models and generate predictions:
   ```R
   source("scripts/33564000_final_model.R")
   ```

## Usage
Upon running the script, the following processes occur:
- **Data Preprocessing**: Cleans and transforms data for modeling.
- **Model Training**: Trains individual machine learning models and a stacking meta-model.
- **Prediction Generation**: Outputs predictions to `RegressionPredictLabel.csv`.

## Contributing
Contributions are welcome! Feel free to fork the repository, make changes, and submit a pull request.

## License
This project is licensed under the GNU General Public License v3.0.

## Contact
For any queries or suggestions, feel free to open an issue on this repository!

