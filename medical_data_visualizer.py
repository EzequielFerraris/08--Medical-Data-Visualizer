import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Import data
dataset = pd.read_csv('medical_examination.csv', sep = ",", skip_blank_lines=True) 

df = pd.DataFrame(dataset)

ow = pd.DataFrame()
ow['overweight'] = df['weight'] / ((df['height'] / 100) ** 2)
ow['overweight'] = (ow['overweight'] > 25).astype(int)

# Add 'overweight' column

df['overweight'] = ow

# Normalize data by making 0 always good and 1 always bad. If the value of 'cholesterol' or 'gluc' is 1, make the value 0. If the value is more than 1, make the value 1.

df['age'] = (df['age'] / 365).round(0).astype(int)
df['cholesterol'] = (df['cholesterol'] > 1).astype(int)
df['gluc'] = (df['gluc'] > 1).astype(int)


def draw_cat_plot():

# Create DataFrame for cat plot using `pd.melt` using just the values from 'cholesterol', 'gluc', 'smoke', 'alco', 'active', and 'overweight'.

    df_cat = df.melt(id_vars='cardio', value_vars=['cholesterol', 'gluc', 'alco', 'active', 'smoke', 'overweight'])

# Group and reformat the data to split it by 'cardio'. Show the counts of each feature. You will have to rename one of the columns for the catplot to work correctly.

    df_cat = [rows for _, rows in df_cat.groupby('cardio', as_index=False)]

    df_cat_0 = df_cat[0]
    df_cat_0 = df_cat_0.value_counts()
    df_cat_0 = df_cat_0.reset_index()
    df_cat_0 = df_cat_0.rename(columns={0:'total'}) 
    df_cat[0] = df_cat_0
  
    df_cat_1 = df_cat[1]
    df_cat_1 = df_cat_1.value_counts()
    df_cat_1 = df_cat_1.reset_index()
    df_cat_1 = df_cat_1.rename(columns={0:'total'})
    df_cat[1] = df_cat_1 
 
    df_cat = pd.concat(df_cat)
  
# Draw the catplot with 'sns.catplot()'

    sns.set_theme(style="darkgrid")
    fig = sns.catplot(x="variable", y="total", hue="value", col='cardio', kind="bar", data = df_cat, order=['active', 'alco', 'cholesterol', 'gluc', 'overweight', 'smoke']).fig
  
# Do not modify the next two lines
    fig.savefig('catplot.png')
    return fig


# Draw Heat Map
def draw_heat_map():
  # Clean the datas
    df1 = df.fillna(0)
  
    df2 = df1[(df1['height'] >= df1['height'].quantile(0.025)) & (df1['height'] <= df1['height'].quantile(0.975)) & (df1['weight'] >= df1['weight'].quantile(0.025)) & (df1['weight'] <= df1['weight'].quantile(0.975)) & (df1['ap_lo'] <= df1['ap_hi'])]
  
    df_heat = df2

    # Calculate the correlation matrix
    corr = df_heat.corr()

    # Generate a mask for the upper triangle

    mask = np.triu(np.ones_like(corr, dtype='bool'))
  
    # Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(11, 9))
  
    # Draw the heatmap with 'sns.heatmap()'
    sns.heatmap(corr, mask=mask, annot=True, fmt='.1f', cmap='Blues')
  
    # Do not modify the next two lines
    fig.savefig('heatmap.png')
    return fig
