import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

#*************************************************
# the purpose of this code is to: 
# 1) load the data; 
# 2) use VIF to remove collinear features; 
# 3) fill NaN values with their mean value;
# 4) store cleaned data into csv file
#*************************************************

colors=['#9edbf9', '#ff7171']

#dataset related definitions
colums_to_be_modified = [1,2,3,4,5,6,7]
target_column = 8
columnNames = ["n_preg", "glu", "blood_pres", "skin_thickness", "isulin", "BMI", "DPF", "age", "Class"]

def calculate_VIF(X):
    thresh = 1000.0
    variables = [i for i in range(X.shape[1])]
    print(variables)
    dropped=True
    print("**************************************")
    print("************ VIF cleaning ************")
    while (dropped==True):
        dropped=False
        vif = [variance_inflation_factor(X[variables].values, ix) for ix in range(X[variables].shape[1])]
        max_vif = max(vif)
        maxloc = vif.index(max_vif)
        if max_vif > thresh:
            print('dropping column %i with VIF = %.2f' %(variables[maxloc], max_vif))
            del variables[maxloc]
            dropped=True
    print('Remaining variables: ', variables)
    print("************ VIF complete ************")
    print("**************************************")
    return variables


df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data', header=None)

#first look at df
#print(df.head(20))

# mark zero values as missing or NaN
df[colums_to_be_modified] = df[colums_to_be_modified].replace(0, np.NaN)

#check again to see NaN are placed correctly
#print(df.head(20))

#temporarily remove rows with NaN for VIF calculation
df_for_VIF = df.dropna()
df_for_VIF.drop(df_for_VIF.columns[[target_column]], axis=1, inplace=True)
#print(df_for_VIF.head(20))

#plot scatter and pie plots
df_for_scatter_plots = df.copy()
df_for_scatter_plots.columns = columnNames
# df_for_scatter_plots.drop(df_for_scatter_plots.columns[[target_column]], axis=1, inplace=True)
features = columnNames[:target_column]
pd.tools.plotting.scatter_matrix(df_for_scatter_plots[features], alpha=0.2, figsize=(10, 10), diagonal='kde', c=df_for_scatter_plots.Class.apply(lambda x:colors[x]))
plt.savefig("/Users/zmao/M-Data/School/scipy/raw_scatter_plot.png")
plt.figure()  # New window
df_for_scatter_plots.Class.value_counts(sort=False).plot.pie(figsize=(6, 6), colors=colors)
plt.savefig("/Users/zmao/M-Data/School/scipy/class_pie_plot.png")
plt.figure()  # New window
corr = df_for_scatter_plots[features].corr()
fig, ax = plt.subplots(figsize=(10, 8))
cax = ax.matshow(corr, vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,len(features),1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
locs, labels = plt.xticks()
plt.setp(labels, rotation=45)

plt.xticks(range(len(corr.columns)), corr.columns);
plt.yticks(range(len(corr.columns)), corr.columns);
plt.savefig("/Users/zmao/M-Data/School/scipy/corr_matrix.png")
plt.figure()  # New window

#remove low variance features?

#perform VIF calculation
good_features = calculate_VIF(df_for_VIF)
output_df = df[good_features + [target_column]]

#fill NaN values
output_df.fillna(output_df.median(), inplace=True)

#set column names
output_df.columns = [columnNames[iCol] for iCol in (good_features+[target_column])]

#shuffle rows to ease next step
output_df = output_df.sample(frac=1).reset_index(drop=True)

#plot new scatter plot
features = output_df.columns[:-1]
pd.tools.plotting.scatter_matrix(output_df[features], alpha=0.2, figsize=(10, 10), diagonal='kde', c=output_df.Class.apply(lambda x:colors[x]))
plt.savefig("/Users/zmao/M-Data/School/scipy/cleaned_scatter_plot.png")


#save to new csv
output_df.to_csv("/Users/zmao/M-Data/School/scipy/cleaned_data.csv", index = False)
