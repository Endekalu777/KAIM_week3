import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import importlib
from IPython.display import display
import warnings
warnings.filterwarnings(
    "ignore", category = FutureWarning
)
warnings.filterwarnings(

    "ignore", category  = UserWarning
)


class DataAnalysis():
    def __init__(self, filepath):
        self.df = pd.read_csv(filepath, delimiter = '|', low_memory = False)

    def data_display(self):
        display(self.df.head())
        print(f"DataFrame shape: {self.df.shape}")
        
    def data_description(self):
        numerical_columns = self.df.select_dtypes(include='number').columns
        categorical_columns = self.df.select_dtypes(exclude='number').columns
        
        print("Numerical columns:")
        display(numerical_columns)
        print("\nCategorical columns:")
        display(categorical_columns)
        
        desc = self.df[numerical_columns].describe()
        desc.loc['variability'] = desc.loc['std'] / desc.loc['mean']
        display(desc)

    def handle_missing_values(self):
        # Categorical Columns
        categorical_columns = [
            'Bank', 'AccountType', 'MaritalStatus', 'Gender', 'Province', 'PostalCode', 
            'MainCrestaZone', 'SubCrestaZone', 'ItemType', 'make', 'Model', 'bodytype'
        ]
        
        for col in categorical_columns:
            self.df[col].fillna('Unknown', inplace=True)  # Replace missing values with 'Unknown'
        
        # Columns where a default or frequent value can be imputed
        self.df['VehicleType'].fillna('Unknown', inplace=True) 
        self.df['mmcode'].fillna(self.df['mmcode'].mode()[0], inplace=True) 
        
        # Columns where zero can be a suitable replacement
        self.df['CustomValueEstimate'].fillna(0, inplace=True)
        self.df['CapitalOutstanding'].fillna(0, inplace=True)
        self.df.drop('NumberOfVehiclesInFleet', axis=1, inplace=True)
        
        # Binary columns where missing values could imply 'No'
        binary_columns = ['NewVehicle', 'WrittenOff', 'Rebuilt', 'Converted', 'CrossBorder']
        for col in binary_columns:
            self.df[col].fillna(0, inplace=True)  
        
        numerical_columns = ['Cylinders', 'cubiccapacity', 'kilowatts', 'NumberOfDoors', 'VehicleIntroDate']

        for col in numerical_columns:
            # Check if the column is numeric before attempting to fill missing values
            if pd.api.types.is_numeric_dtype(self.df[col]):
                self.df[col].fillna(self.df[col].median(), inplace=True)
            else:
                # If the column is not numeric, handle it differently (e.g., leave as NaN or fill with the mode)
                print(f"Column '{col}' is not numeric, skipping median imputation.")
                # For date columns, you might want to convert to datetime and handle accordingly
                if 'Date' in col:
                    self.df[col] = pd.to_datetime(self.df[col], errors='coerce')
                    self.df[col].fillna(self.df[col].mode()[0], inplace=True)

        # Check for remaining missing values after imputation
        remaining_missing = self.df.isnull().sum()
        print(f"Remaining missing values after imputation: \n{remaining_missing}")

    def data_type(self):
        print("Original data types:")
        display(self.df.dtypes)
        
        for col in self.df.select_dtypes(include=['object']).columns:
            sample = self.df[col].dropna().sample(min(100, len(self.df[col].dropna())))
            if sample.apply(lambda x: pd.to_datetime(x, errors='coerce')).notna().all():
                self.df[col] = pd.to_datetime(self.df[col], errors='coerce')
                print(f"Converted {col} to datetime")

    def data_quality(self):
        missing_values = self.df.isnull().sum()
        missing_percentages = 100 * missing_values / len(self.df)
        missing_table = pd.concat([missing_values, missing_percentages], axis=1, keys=['Missing Values', 'Percentage'])
        display(missing_table.sort_values('Percentage', ascending=False))


    def Univariate_analysis(self):
        # Identify numerical and categorical columns
        numerical_columns = self.df.select_dtypes(include='number').columns
        categorical_columns = self.df.select_dtypes(exclude='number').columns
        
        # Plot for numerical columns
        n_cols = 3
        n_rows = int(np.ceil(len(numerical_columns) / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
        axes = axes.flatten()
        
        for i, col in enumerate(numerical_columns):
            ax = axes[i]
            sns.histplot(self.df[col], bins=30, kde=True, ax=ax)
            ax.set_title(f"Distribution of {col}")
            ax.set_xlabel(col)
            ax.set_ylabel("Frequency")
        
        # Remove extra subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
        
        plt.tight_layout()
        plt.show()

        # Plot for categorical columns
        if len(categorical_columns) > 0:
            n_cols = 4
            n_rows = int(np.ceil(len(categorical_columns) / n_cols))
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
            axes = axes.flatten()
            
            for i, col in enumerate(categorical_columns):
                ax = axes[i]
                value_counts = self.df[col].value_counts()
                top_n = 15  # Show top 15 categories
                
                if len(value_counts) > top_n:
                    other = pd.Series({'Other': value_counts.iloc[top_n:].sum()})
                    value_counts = pd.concat([value_counts.iloc[:top_n], other])
                
                sns.barplot(x=value_counts.index, y=value_counts.values, ax=ax)
                ax.set_title(f"Distribution of {col}")
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
                ax.set_ylabel('Count')
                
                # Add value labels on top of each bar
                for p in ax.patches:
                    ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                                ha='center', va='center', xytext=(0, 10), textcoords='offset points')

            # Remove extra subplots
            for j in range(i + 1, len(axes)):
                fig.delaxes(axes[j])
        
        plt.tight_layout()
        plt.show()

    def monthly_changes(self):
        # Convert 'PostalCode' to category type and 'TransactionMonth' to datetime type
        self.df['PostalCode'] = self.df['PostalCode'].astype('category')
        self.df['TransactionMonth'] = pd.to_datetime(self.df['TransactionMonth'])
        
        # Extract 'Month' from 'TransactionMonth'
        self.df['Month'] = self.df['TransactionMonth'].dt.to_period('M')
        
        # Sort the DataFrame by 'PostalCode' and 'TransactionMonth' to ensure correct percentage change calculation
        self.df.sort_values(['PostalCode', 'TransactionMonth'], inplace=True)
        
        # Calculate monthly percentage changes for TotalPremium and TotalClaims within each PostalCode group
        self.df['TotalPremium_change'] = self.df.groupby('PostalCode')['TotalPremium'].pct_change().fillna(0)
        self.df['TotalClaims_change'] = self.df.groupby('PostalCode')['TotalClaims'].pct_change().fillna(0)

    def correlation_analysis(self):
        # Calculate correlations between monthly changes
        corr_matrix = self.df[['TotalPremium_change', 'TotalClaims_change']].corr()
        display(corr_matrix)
        
        # Plot heatmap for correlation matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".5f")
        plt.title("Correlation Matrix of Monthly Changes")
        plt.show()
        
        # aggregate the data by PostalCode
        agg_data = self.df.groupby('PostalCode').agg({
        'TotalPremium_change': ['mean', 'median'],
        'TotalClaims_change': ['mean', 'median']
        })

        agg_data.columns = ['Premium_change_mean', 'Premium_change_median', 'Claims_change_mean', 'Claims_change_median']
        
        plt.figure(figsize=(10, 8))
        sns.scatterplot(data=agg_data, x='Premium_change_mean', y='Claims_change_mean')
        plt.title("Aggregated Monthly Changes by Postal Code")
        plt.xlabel('Mean Monthly Change in TotalPremium')
        plt.ylabel('Mean Monthly Change in TotalClaims')
        plt.grid(True)
        plt.show()

    def trends_over_geography(self):
        # Group by PostalCode and calculate mean or count for comparison
        insurance_cover_trends = self.df.groupby('PostalCode')['CoverType'].value_counts(normalize=True).unstack().fillna(0)
        premium_trends = self.df.groupby('PostalCode')['TotalPremium'].mean()
        auto_make_trends = self.df.groupby('PostalCode')['make'].value_counts(normalize=True).unstack().fillna(0)

        # Visualize trends
        plt.figure(figsize=(12, 6))
        insurance_cover_trends.plot(kind='bar', stacked=True, colormap='tab20', figsize=(15, 8))
        plt.title('Trends in Insurance Cover Type Over Geography (PostalCode)')
        plt.ylabel('Proportion of Insurance Cover Type')
        plt.xlabel('PostalCode')
        plt.xticks(rotation=45)
        plt.legend(title='Insurance Cover Type', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(12, 6))
        premium_trends.plot(kind='line', marker='o')
        plt.title('Average Premium Over Geography (PostalCode)')
        plt.ylabel('Average Premium')
        plt.xlabel('PostalCode')
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.show()

        plt.figure(figsize=(12, 6))
        auto_make_trends.plot(kind='bar', stacked=True, colormap='tab20', figsize=(15, 8))
        plt.title('Trends in Auto Make Over Geography (ZipCode)')
        plt.ylabel('Proportion of Auto Make')
        plt.xlabel('PostalCode')
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
            sns.boxplot(y=self.df[col], color='skyblue')  # Changed x to y for vertical orientation
            plt.title(f'Box Plot of {col}')
            plt.ylabel(col)  
            plt.tight_layout()

        plt.show()

    def creative_visualizations(self):
        # Pair Plot for key numerical variables
        sns.pairplot(self.df[['TotalPremium', 'TotalClaims']])
        plt.suptitle('Pair Plot of Key Variables', y=1.02)
        plt.show()

        # Before creating a heatmap, ensure only numeric data is used
        plt.figure(figsize=(10, 8))
        
        correlation_columns = self.df[['Cylinders', 'cubiccapacity', 'kilowatts', 'NumberOfDoors', 'CustomValueEstimate', 'CapitalOutstanding', 'SumInsured', 'CalculatedPremiumPerTerm', 'TotalPremium', 'TotalClaims']]
        correlation_columns_numeric = correlation_columns.apply(pd.to_numeric, errors = 'coerce')
        correlation_columns_numeric = correlation_columns_numeric.dropna(axis = 1, how ='all')

        # Calculate the correlation matrix
        correlation_matrix = correlation_columns_numeric.corr()

        # Generate the heatmap
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
        plt.title('Heatmap of Correlation Matrix')
        plt.show()

        # Violin Plot for distribution analysis
        plt.figure(figsize=(12, 6))
        sns.violinplot(x='PostalCode', y='TotalPremium', data=self.df, palette='Set2')
        plt.title('Violin Plot of Total Premium Distribution by ZipCode')
        plt.xticks(rotation=45)
        plt.show()

    def run_analysis(self): 
        self.data_display()
        self.data_quality()
        self.data_description()
        self.handle_missing_values()
        self.data_type()
        self.Univariate_analysis()
        self.monthly_changes()
        self.correlation_analysis()
        self.trends_over_geography()
        self.detect_outliers()
        self.creative_visualizations()

