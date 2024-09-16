import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

class ABHypothesisTesting:
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path)
        self.results = {}
        self.prepare_data()

    def prepare_data(self):
        self.data['Margin'] = self.data['TotalPremium'] - self.data['TotalClaims']
        
        le = LabelEncoder()
        categorical_cols = ['Province', 'PostalCode', 'Gender', 'StatutoryRiskType']
        for col in categorical_cols:
            self.data[col] = le.fit_transform(self.data[col].astype(str))
            
    def segment_data(self, feature, group_a_value, group_b_value):
        group_a = self.data[self.data[feature] == group_a_value]
        group_b = self.data[self.data[feature] == group_b_value]
        return group_a, group_b

    def check_group_equivalence(self, group_a, group_b, features_to_check):
        for feature in features_to_check:
            if self.data[feature].dtype in ['int64', 'float64']:
                _, p_value = stats.ttest_ind(group_a[feature], group_b[feature])
            else:
                observed = pd.crosstab(self.data[feature], columns='count')
                _, p_value, _, _ = stats.chi2_contingency(observed)
            
            if p_value < 0.05:
                print(f"Groups differ for {feature} (p={p_value:.4f})")
            else:
                print(f"Groups are equivalent for {feature} (p={p_value:.4f})")

    def test_risk_differences(self, feature):
        grouped = self.data.groupby(feature)['StatutoryRiskType'].value_counts().unstack()
        chi2, p_value, _, _ = stats.chi2_contingency(grouped.fillna(0))
        self.results[f'{feature} Risk Difference'] = p_value
        return chi2, p_value

    def test_margin_difference(self, feature, group_a_value, group_b_value):
        group_a, group_b = self.segment_data(feature, group_a_value, group_b_value)
        t_stat, p_value = stats.ttest_ind(group_a['Margin'], group_b['Margin'])
        self.results[f'{feature} Margin Difference'] = p_value
        return t_stat, p_value

    def analyze_results(self, alpha=0.05):
        for test, p_value in self.results.items():
            if p_value < alpha:
                print(f"{test}: Reject null hypothesis (p = {p_value:.5f})")
            else:
                print(f"{test}: Fail to reject null hypothesis (p = {p_value:.5f})")
            print()

    def visualize_results(self):
        plt.figure(figsize=(12, 6))
        sns.barplot(x=list(self.results.keys()), y=list(self.results.values()))
        plt.title('P-values for Different Tests')
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('P-value')
        plt.tight_layout()
        plt.show('hypothesis_test_results.png')
        plt.close()

        def additional_analysis(self):
            avg_margin_by_province = self.data.groupby('Province')['Margin'].mean().sort_values(ascending=False)
            print("Average Margin by Province:")
            print(avg_margin_by_province)

            risk_by_gender = self.data.groupby('Gender')['StatutoryRiskType'].value_counts(normalize=True).unstack()
            print("\nRisk Distribution by Gender:")
            print(risk_by_gender)

            correlation = self.data['TotalPremium'].corr(self.data['TotalClaims'])
            print(f"\nCorrelation between Total Premium and Total Claims: {correlation:.4f}")


    def run_analysis(self):
        print("1. Testing risk differences across provinces")
        self.test_risk_differences('Province')

        print("\n2. Testing risk differences between postal codes")
        self.test_risk_differences('PostalCode')

        print("\n3. Testing margin differences between postal codes")
        postal_codes = self.data['PostalCode'].unique()[:2]
        self.test_margin_difference('PostalCode', postal_codes[0], postal_codes[1])

        print("\n4. Testing risk differences between genders")
        self.test_risk_differences('Gender')

        print("\nAnalyzing Results:")
        self.analyze_results()

        print("\nVisualizing Results:")
        self.visualize_results()

        print("\nAdditional Analysis:")
        self.additional_analysis()

