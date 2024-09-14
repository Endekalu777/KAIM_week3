import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings
from IPython.display import display

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


class DataAnalysis:
    def __init__(self, filepath):
        self.df = pd.read_csv(filepath, delimiter='|', low_memory=False)

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
        # Categorical Columns - Replace missing values with 'Unknown'
        categorical_columns = [
            'Bank', 'AccountType', 'MaritalStatus', 'Gender', 'Province', 'PostalCode',
            'MainCrestaZone', 'SubCrestaZone', 'ItemType', 'make', 'Model', 'bodytype'
        ]

        for col in categorical_columns:
            self.df[col].fillna('Unknown', inplace=True)

        # For specific columns, replace missing values with mode or a suitable default value
        self.df['VehicleType'].fillna('Unknown', inplace=True)
        self.df['mmcode'].fillna(self.df['mmcode'].mode()[0], inplace=True)

        # Numerical Columns - Replace missing values with median or 0 where appropriate
        numerical_columns = ['Cylinders', 'cubiccapacity', 'kilowatts', 'NumberOfDoors', 'VehicleIntroDate']

        for col in numerical_columns:
            if pd.api.types.is_numeric_dtype(self.df[col]):
                self.df[col].fillna(self.df[col].median(), inplace=True)
            else:
                print(f"Column '{col}' is not numeric, skipping median imputation.")

        # Additional columns for special handling
        self.df['CustomValueEstimate'].fillna(0, inplace=True)
        self.df['CapitalOutstanding'].fillna(0, inplace=True)
        self.df.drop('NumberOfVehiclesInFleet', axis=1, inplace=True)

        # Binary columns where missing values could imply 'No'
        binary_columns = ['NewVehicle', 'WrittenOff', 'Rebuilt', 'Converted', 'CrossBorder']
        for col in binary_columns:
            self.df[col].fillna(0, inplace=True)

        # Check for remaining missing values
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
            sample = self.df[col].sample(n=min(100000, len(self.df)))
            sns.histplot(sample, bins=30, kde=True, ax=ax)
            ax.set_title(f"Distribution of {col}")
            ax.set_xlabel(col)
            ax.set_ylabel("Frequency")
            ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

        # Remove extra subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.show()

        # Plot for categorical columns
        if len(categorical_columns) > 0:
            n_cols = 3
            n_rows = int(np.ceil(len(categorical_columns) / n_cols))
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
            axes = axes.flatten()

            for i, col in enumerate(categorical_columns):
                ax = axes[i]
                value_counts = self.df[col].value_counts()
                top_n = 10  # Show top 10 categories
                if len(value_counts) > top_n:
                    other = pd.Series({'Other': value_counts.iloc[top_n:].sum()})
                    value_counts = pd.concat([value_counts.iloc[:top_n], other])

                sns.barplot(x=value_counts.index, y=value_counts.values, ax=ax)
                ax.set_title(f"Distribution of {col}")
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
                ax.set_ylabel('Count')
                ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

                # Add value labels on top of each bar
                for p in ax.patches:
                    ax.annotate(f'{int(p.get_height()):,}',
                                (p.get_x() + p.get_width() / 2., p.get_height()),
                                ha='center', va='center', xytext=(0, 10),
                                textcoords='offset points', fontsize=8)

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

        # Aggregate the data by PostalCode
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

        # Get the top 15 PostalCodes based on average TotalPremium
        top_15_postal_codes = premium_trends.nlargest(15).index

        # Filter the data for these top 15 PostalCodes
        insurance_cover_trends = insurance_cover_trends.loc[top_15_postal_codes]
        premium_trends = premium_trends.loc[top_15_postal_codes]
        auto_make_trends = auto_make_trends.loc[top_15_postal_codes]

        # Visualize trends
        plt.figure(figsize=(12, 6))
        insurance_cover_trends.plot(kind='bar', stacked=True, colormap='tab20', figsize=(15, 8))
        plt.title('Trends in Insurance Cover Type Over Top 15 Postal Codes')
        plt.ylabel('Proportion of Insurance Cover Type')
        plt.xlabel('PostalCode')
        plt.xticks(rotation=45)
        plt.legend(title='Insurance Cover Type', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(12, 6))
        premium_trends.plot(kind='line', marker='o')
        plt.title('Average Premium Over Top 15 Postal Codes')
        plt.ylabel('Average Premium')
        plt.xlabel('PostalCode')
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.show()

        plt.figure(figsize=(12, 6))
        auto_make_trends.plot(kind='bar', stacked=True, colormap='tab20', figsize=(15, 8))
        plt.title('Trends in Auto Make Over Top 15 Postal Codes')
        plt.ylabel('Proportion of Auto Make')
        plt.xlabel('PostalCode')
        plt.xticks(rotation=45)
        plt.legend(title='Auto Make', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()

    def customer_segmentation_analysis(df, n_clusters=5):
        """
        Perform a creative analysis on car insurance data, including customer segmentation, risk profiling,
        and claim pattern analysis.

        Parameters:
        - df: DataFrame containing the insurance data.
        - n_clusters: The number of customer segments to create using KMeans clustering.

        Returns:
        - cluster_summary: DataFrame summarizing each cluster's risk characteristics.
        - cluster_analysis_plot: Visual plot showing the cluster distribution.
        """
        # Select key features for segmentation (you can adjust this based on your dataset)
        features = ['total_premium', 'total_claims', 'vehicle_age', 'annual_mileage', 'claim_frequency']

        # Drop missing values in selected features
        df_segmentation = df[features].dropna()

        # Standardize the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df_segmentation)

        # Perform KMeans clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(scaled_data)
        df_segmentation['Cluster'] = clusters

        # Add cluster labels back to the original dataset
        df['Cluster'] = clusters

        # Analyze clusters for risk profiling
        cluster_summary = df_segmentation.groupby('Cluster').mean()

        # Add cluster size for context
        cluster_summary['Cluster Size'] = df_segmentation['Cluster'].value_counts().sort_index()

        # Visualize cluster distribution using a pairplot
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=df['total_premium'], y=df['total_claims'], hue=df['Cluster'], palette='viridis', s=100)
        plt.title('Customer Segmentation Based on Premiums and Claims')
        plt.xlabel('Total Premium')
        plt.ylabel('Total Claims')
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
        # Identify the top 15 postal codes based on TotalPremium
        top_15_postal_codes = self.df.groupby('PostalCode')['TotalPremium'].mean().nlargest(15).index

        # Filter the DataFrame to include only the top 15 postal codes
        df_top_15 = self.df[self.df['PostalCode'].isin(top_15_postal_codes)]

        # Pair Plot for key numerical variables
        sns.pairplot(df_top_15[['TotalPremium', 'TotalClaims']])
        plt.suptitle('Pair Plot of Key Variables (Top 15 Postal Codes)', y=1.02)
        plt.show()

        # Before creating a heatmap, ensure only numeric data is used
        plt.figure(figsize=(10, 8))

        correlation_columns = df_top_15[['Cylinders', 'cubiccapacity', 'kilowatts', 'CustomValueEstimate', 'CapitalOutstanding', 'SumInsured', 'CalculatedPremiumPerTerm', 'TotalPremium', 'TotalClaims']]
        correlation_columns_numeric = correlation_columns.apply(pd.to_numeric, errors='coerce')
        correlation_columns_numeric = correlation_columns_numeric.dropna(axis=1, how='all')

        # Calculate the correlation matrix
        correlation_matrix = correlation_columns_numeric.corr()

        # Generate the heatmap
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
        plt.title('Heatmap of Correlation Matrix (Top 15 Postal Codes)')
        plt.show()

        # Violin Plot for distribution analysis
        plt.figure(figsize=(16, 6))  # Increase the figure size to make it wider
        sns.violinplot(x='PostalCode', y='TotalPremium', data=df_top_15, palette='Set2')
        plt.title('Violin Plot of Total Premium Distribution by PostalCode (Top 15)')

        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right')

        # Adjust layout to fit the labels
        plt.tight_layout()
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
        self.customer_segmentation_analysis
        self.detect_outliers()
        self.creative_visualizations()
