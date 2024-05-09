Copper Sales Analysis and Prediction

Overview

This comprehensive project encompasses both data analysis and predictive modeling for copper sales. It includes a Jupyter Notebook for exploratory data analysis and machine learning model training, as well as a Streamlit application for predicting selling prices and status.

 Files

- `Copper_model.ipynb`: Jupyter Notebook for data analysis and model training.
- `Copper.py`: Python script for the Streamlit application to predict outcomes.

 Data Analysis

 Requirements

- pandas
- numpy
- seaborn
- matplotlib
- scikit-learn
- pickle

 Installation

Install the required libraries with:


pip install pandas numpy seaborn matplotlib scikit-learn pickle
Usage
•	Run the Jupyter Notebook:
jupyter notebook Copper_model.ipynb
•	Execute the Python script:
python Copper.py
Data Cleaning
The dataset undergoes several cleaning steps, including conversion of dates to datetime format, numerical type corrections, and handling missing values.
Visualization
Insights are drawn from the data using various visualizations like heatmaps and boxplots.
Machine Learning
Two models are trained:
•	Decision Tree Regressor for selling_price.
•	Decision Tree Classifier for status.
Models are fine-tuned with GridSearchCV and saved for later use.
Predictive Application
Requirements
•	streamlit
•	streamlit_option_menu
•	numpy
•	pickle
•	scikit-learn
Installation
Install the required libraries with:
pip install streamlit streamlit_option_menu numpy pickle scikit-learn
Running the Application
Navigate to the directory containing Copper.py and run:
streamlit run Copper.py
Features
•	About Project: Overview of the project’s purpose and technologies used.
•	Predictions: Interface for users to input data and receive predictions.
Predictions
•	Predict Selling Price: Decision Tree Regressor predicts the selling price based on input features.
•	Predict Status: Decision Tree Classifier predicts the status (Won/Lost) based on input features.
Models
The application utilizes pre-trained .pkl machine learning models for predictions.
Conclusion
This project provides a full workflow for copper sales data analysis and a user-friendly application for predicting selling prices and status, showcasing the power of machine learning in the manufacturing domain.


