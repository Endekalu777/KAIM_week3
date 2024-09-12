import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import importlib
from IPython.display import display


class DataAnalysis():
    def __init__(self, filepath):
        self.df = pd.read_csv(filepath, delimiter = '|', low_memory = False)

    def data_display(self):
        display(self.df.head())
        display(self.df.shape)
        
    def data_description(self):
        numerical_columns = self.df.select_dtypes(include = 'number').columns
        categorical_columns = self.df.select_dtypes(exclude = 'number').columns
        print("Numerical columns:")
        display(numerical_columns)
        print("\nCategorical columns")
        display(categorical_columns)        
        display(self.df[numerical_columns].describe())

    def data_type(self):
        print("Original data types:")
        display(self.df.dtypes)
        
        for col in self.df.select_dtypes(include=['object']).columns:
            sample = self.df[col].dropna().sample(min(100, len(self.df[col].dropna())))
            if sample.apply(lambda x: pd.to_datetime(x, errors='coerce')).notna().all():
                self.df[col] = pd.to_datetime(self.df[col], errors='coerce')
                print(f"Converted {col} to datetime")

    def Data_quality(self):
        display(self.df.isnull().sum())

    def Univariate_analysis(self):
        numerical_columns = self.df.select_dtypes(include = 'number').columns
        categorical_columns = self.df.select_dtypes(exclude = 'number').columns
        n_rows = 5
        n_cols = int(np.ceil(len(numerical_columns) / n_rows))
        plt.figure(figsize = (15, n_rows * 5))
        for i, col in enumerate(numerical_columns):
            plt.subplot(n_rows, n_cols, i + 1)
            sns.histplot(self.df[col], bins = 30, kde = True)
            plt.title(f"Distribution of {col}")
            plt.xlabel(col)
            plt.ylabel("frequency")
        plt.tight_layout()
        plt.show()

        for col in categorical_columns:
            plt.figure(figsize = (10, 5))
            sns.countplot(data = self.df, x = col)
            plt.title(f"Distribution of {col}")
            plt.xticks(rotation = 60)
            plt.xlabel(col)
            plt.ylabel('Count')
            plt.show()

    def monthly_changes(self):
        self.df['ZipCode'] = self.df['ZipCode'].astype('category')
        self.df['TransactionMonth'] = pd.to_datetime(self.df['TranscationMonth'])
        self.df['Month'] = self.df['TransactionMonth'].dt.to_period('M')
        # Calculate monthly changes for TotalPremium and TotalClaims
        self.monthly_changes = self.df.groupby(['ZipCode', 'Month']).agg(
            TotalPremium_change=('TotalPremium', lambda x: x.pct_change().fillna(0)),
            TotalClaims_change=('TotalClaims', lambda x: x.pct_change().fillna(0))
        ).reset_index()

    def correlation_analysis(self):
        # Calculate correlations between monthly changes
        corr_matrix = self.monthly_changes[['TotalPremium_change', 'TotalClaims_change']].corr()
        display(corr_matrix)
        
        # Plot heatmap for correlation matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title("Correlation Matrix of Monthly Changes")
        plt.show()
        
        # Scatter plots for each ZipCode
        zipcodes = self.monthly_changes['ZipCode'].unique()
        for zipcode in zipcodes:
            zipcode_data = self.monthly_changes[self.monthly_changes['ZipCode'] == zipcode]
            
            plt.figure(figsize=(8, 6))
            sns.scatterplot(data=zipcode_data, x='TotalPremium_change', y='TotalClaims_change', marker='o')
            plt.title(f"Scatter Plot of Monthly Changes in TotalPremium vs TotalClaims for ZipCode {zipcode}")
            plt.xlabel('Monthly Change in TotalPremium')
            plt.ylabel('Monthly Change in TotalClaims')
            plt.grid(True)
            plt.show()

    def trends_over_geography(self):
        # Group by ZipCode and calculate mean or count for comparison
        insurance_cover_trends = self.df.groupby('ZipCode')['InsuranceCoverType'].value_counts(normalize=True).unstack().fillna(0)
        premium_trends = self.df.groupby('ZipCode')['TotalPremium'].mean()
        auto_make_trends = self.df.groupby('ZipCode')['AutoMake'].value_counts(normalize=True).unstack().fillna(0)

        # Visualize trends
        plt.figure(figsize=(12, 6))
        insurance_cover_trends.plot(kind='bar', stacked=True, colormap='tab20', figsize=(15, 8))
        plt.title('Trends in Insurance Cover Type Over Geography (ZipCode)')
        plt.ylabel('Proportion of Insurance Cover Type')
        plt.xlabel('ZipCode')
        plt.xticks(rotation=45)
        plt.legend(title='Insurance Cover Type', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(12, 6))
        premium_trends.plot(kind='line', marker='o')
        plt.title('Average Premium Over Geography (ZipCode)')
        plt.ylabel('Average Premium')
        plt.xlabel('ZipCode')
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.show()

        plt.figure(figsize=(12, 6))
        auto_make_trends.plot(kind='bar', stacked=True, colormap='tab20', figsize=(15, 8))
        plt.title('Trends in Auto Make Over Geography (ZipCode)')
        plt.ylabel('Proportion of Auto Make')
        plt.xlabel('ZipCode')
        plt.xticks(rotation=45)
        plt.legend(title='Auto Make', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()

    def detect_outliers(self):
        numerical_columns = self.df.select_dtypes(include='number').columns

        # Create box plots for each numerical column to detect outliers
        plt.figure(figsize=(15, len(numerical_columns) * 5))
        for i, col in enumerate(numerical_columns, 1):
            plt.subplot(len(numerical_columns), 1, i)
            sns.boxplot(x=self.df[col], color='skyblue')
            plt.title(f'Box Plot of {col}')
            plt.xlabel(col)
        plt.tight_layout()
        plt.show()

    def creative_visualizations(self):
        # Pair Plot for key numerical variables
        sns.pairplot(self.df[['TotalPremium', 'TotalClaims', 'OtherVariable1', 'OtherVariable2']])
        plt.suptitle('Pair Plot of Key Variables', y=1.02)
        plt.show()

        # Heatmap of correlation matrix
        plt.figure(figsize=(10, 8))
        correlation_matrix = self.df.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
        plt.title('Heatmap of Correlation Matrix')
        plt.show()

        # Violin Plot for distribution analysis
        plt.figure(figsize=(12, 6))
        sns.violinplot(x='ZipCode', y='TotalPremium', data=self.df, palette='Set2')
        plt.title('Violin Plot of Total Premium Distribution by ZipCode')
        plt.xticks(rotation=45)
        plt.show()


    def run_analysis(self): 
        self.data_display()
        self.data_description()
        self.Data_quality()
        self.data_type()
        self.Univariate_analysis()
        self.monthly_changes()
        self.correlation_analysis()

