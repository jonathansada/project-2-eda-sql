import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# Expected to be used with the parameter na_values of the method pd.read_csv
def na_values(exclude=[]):
    # Default values defined in doc: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html
    na_values = [" ", "#N/A", "#N/A N/A", "#NA", "-1.#IND", "-1.#QNAN", "-NaN", "-nan", "1.#IND", "1.#QNAN", "<NA>", "N/A", "NA", "NULL", "NaN", "None", "n/a", "nan", "null"]
    for v in exclude:
        na_values.remove(v)
    return na_values

def cleanColNames(df, ncol={}):
    df.columns = [ncol[column] if (column in ncol) else (column.lower().replace(" ", "_")) for column in df.columns]
    return df

def getNullValues(df, onlyEmpty=False):
    tmp_df = df.isna().sum()
    return tmp_df[tmp_df>0] if onlyEmpty else tmp_df
    
def cleanNullRows(df):
    return df.dropna(axis=0, how="all")

def getDuplicated(df):
    return df.duplicated().sum()

def cleanDuplicated(df, keep="first"):
    df.drop_duplicates(keep=keep, inplace=True)
    df.reset_index(inplace=True)
    return df

def getFrequencyTable(df, column):
    return pd.concat([df[column].value_counts(), df[column].value_counts(normalize=True).round(2)], axis=1)

# TODO: Call show in the notebook, function only calculate and return values
def analyze_cat_values(df, column, chart_size=False, ax_labels = False, ay_labels = False):
    print(f"Proportion table for {column}: ")
    frequency_table = getFrequencyTable(df, column)
    display(frequency_table)

    if len(df[column].unique()) <= 2: # If it is binary draw a pie plot
        chart_size = chart_size if chart_size else (6,6)
        plt.figure(figsize=chart_size)
        labels = ax_labels if ax_labels else frequency_table.index
        plt.pie(frequency_table["proportion"], labels=labels, autopct='%1.1f%%')
        plt.title(f'Proportion of values for the feature {column}')
    else: # else draw a barchart
        chart_size = chart_size if chart_size else (6,3)
        plt.figure(figsize=chart_size)
        ax = sns.barplot(y=frequency_table["count"], x=frequency_table.index, data=frequency_table, legend=False);
        plt.title(f'Proportion of values for the feature {column}')
        if ax_labels:
            ax.set_xticklabels(ax_labels, rotation=0)
        if ay_labels:
            ax.set_xticklabels(ay_labels, rotation=0)

    plt.show();

# TODO: Call print and show in the notebook, function only calculate and return values
def analyze_num_values(df, column, ax_labels = False, ay_labels = False):
    print(f"Statistic values for {column}:")
    print("Count:\t\t", df[column].count())
    print("AVG:\t\t", df[column].mean())
    print("Min:\t\t", df[column].min())
    print("Quantile 25:\t", df[column].quantile(0.25))
    print("Quantile 50:\t", df[column].quantile(0.5))
    print("Quantile 75:\t", df[column].quantile(0.75))
    print("Max:\t\t", df[column].max())
    print("Mode:\t\t", list(df[column].mode()))
    print("Variance:\t", df[column].var())
    print("STD:\t\t", df[column].std())
    print("Skewness:\t", df[column].skew())
    print("Kurtosis:\t", df[column].kurt())

    plt.figure(figsize=(10,5))
    plot = sns.histplot(df[column], bins=100)
    if ax_labels:
        ax.set_xticklabels(ax_labels, rotation=0)
    if ay_labels:
        ax.set_xticklabels(ay_labels, rotation=0)
    #plot.set(yscale='log')

    plt.show();

def isChisquareStrong(chi2_pvalue):
    return float(chi2_pvalue) < 0.05

def getCramervRelation(cramer_v):
    return ["Negligible", "Weak", "Moderate", "Strong", "Very Strong"][int(cramer_v*10)]