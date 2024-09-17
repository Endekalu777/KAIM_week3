import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import shap
from datetime import datetime

class ModelAnalysis:
    def __init__(self, file_path):
        self.df = pd.read_csv(file_path)
        self.cat_cols = None
        self.num_cols = None
        self.X_train = None
        self.X_test = None 
        self.y_train = None 
        self.y_test = None
        self.lr_model = None
        self.rf_model = None  
        self.xgb_model = None 

    def handle_missing_values(self):
        relevant_columns = [
            'SumInsured', 'CalculatedPremiumPerTerm', 'CapitalOutstanding',
            'VehicleType', 'make', 'Model', 'bodytype', 'Province', 'AlarmImmobiliser', 'TrackingDevice', 'NewVehicle', 'Rebuilt', 'Converted', 'CoverCategory', 'CoverType', 'CoverGroup', 'TotalPremium', 'TotalClaims',
            'VehicleIntroDate', 'RegistrationYear' 
        ]

        self.df = self.df[relevant_columns]

        num_imputer = SimpleImputer(strategy="median")
        cat_imputer = SimpleImputer(strategy="most_frequent")

        self.num_cols = self.df.select_dtypes(include=["float64", "int64"]).columns
        self.cat_cols = self.df.select_dtypes(include=["object", "bool"]).columns

        self.df[self.num_cols] = num_imputer.fit_transform(self.df[self.num_cols])
        self.df[self.cat_cols] = cat_imputer.fit_transform(self.df[self.cat_cols])

    def feature_engineering(self):
        # Calculate VehicleAge using RegistrationYear
        current_year = datetime.now().year
        self.df['VehicleAge'] = current_year - self.df['RegistrationYear']

        # Create a feature for presence of Alarm or Tracking Device
        self.df['HasAlarmOrTracker'] = np.where((self.df['AlarmImmobiliser'] == 'Yes') | (self.df['TrackingDevice'] == 'Yes'), 1, 0)
        self.num_cols = list(self.num_cols) + ['VehicleAge', 'HasAlarmOrTracker']
    def encoder(self):
        encoder = OneHotEncoder(drop='first', sparse_output=False)
        encoded_cat_data = encoder.fit_transform(self.df[self.cat_cols])
        encoded_cat_df = pd.DataFrame(encoded_cat_data, columns=encoder.get_feature_names_out(self.cat_cols))
    
        # Include VehicleAge and other numeric columns
        numeric_df = self.df[self.num_cols]
    
        self.df = pd.concat([numeric_df, encoded_cat_df], axis=1)

    def train_test_split(self):
        X = self.df.drop(['TotalPremium'], axis=1) 
        y = self.df['TotalPremium']  

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    def linear_model(self):
        self.lr_model = LinearRegression()  
        self.lr_model.fit(self.X_train, self.y_train)
        lr_preds = self.lr_model.predict(self.X_test)
        lr_mse = mean_squared_error(self.y_test, lr_preds)
        lr_r2 = r2_score(self.y_test, lr_preds)
        print(f"Linear Regression MSE: {lr_mse}")
        print(f"Linear Regression R2 Score: {lr_r2}")

    def randomf_model(self):
        self.rf_model = RandomForestRegressor(n_estimators=25, max_depth=10, random_state=42)  
        self.rf_model.fit(self.X_train, self.y_train)
        rf_preds = self.rf_model.predict(self.X_test)
        rf_mse = mean_squared_error(self.y_test, rf_preds)
        rf_r2 = r2_score(self.y_test, rf_preds)
        print(f"Random Forest MSE: {rf_mse}")
        print(f"Random Forest R2 Score: {rf_r2}")

    def xg_model(self):
        self.xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)  
        self.xgb_model.fit(self.X_train, self.y_train)
        xgb_preds = self.xgb_model.predict(self.X_test)
        xgb_mse = mean_squared_error(self.y_test, xgb_preds)
        xgb_r2 = r2_score(self.y_test, xgb_preds)
        print(f"XGBoost MSE: {xgb_mse}")
        print(f"XGBoost R2 Score: {xgb_r2}")

    def shap_analysis(self):
        if self.xgb_model is None:
            self.xg_model()
        explainer = shap.TreeExplainer(self.xgb_model)
        shap_values = explainer.shap_values(self.X_test)
        shap.summary_plot(shap_values, self.X_test)
        shap.dependence_plot("VehicleAge", shap_values, self.X_test)

        if self.lr_model is None:
            self.linear_model()
        explainer = shap.LinearExplainer(self.lr_model, self.X_train)  
        shap_values = explainer.shap_values(self.X_test)
        shap.summary_plot(shap_values, self.X_test)
        shap.dependence_plot("VehicleAge", shap_values, self.X_test)

        if self.rf_model is None:
            self.randomf_model()  

        explainer = shap.TreeExplainer(self.rf_model)
        shap_values = explainer.shap_values(self.X_test)
        shap.summary_plot(shap_values, self.X_test)
        shap.dependence_plot("VehicleAge", shap_values, self.X_test)