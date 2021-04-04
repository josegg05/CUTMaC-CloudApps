import miceforest as mf
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data and introduce missing values
iris = pd.concat(load_iris(as_frame=True,return_X_y=True),axis=1)
iris.drop(columns=['target'], inplace=True)
#iris['target'] = iris['target'].astype('category')
iris_amp = mf.ampute_data(iris,perc=0.25,random_state=1991)

#%%
print(iris.columns)
print(iris.head())
print(iris_amp.head())

#%% Create kernel.
kds = mf.KernelDataSet(
  iris_amp,
  save_all_iterations=True,
  random_state=1991
)

# Run the MICE algorithm for 3 iterations
kds.mice(3)

# Return the completed kernel data
completed_data = kds.complete_data()

print(completed_data.head())

#%% Create kernel.
kernel = mf.MultipleImputedKernel(
  iris_amp,
  datasets=4,
  save_all_iterations=True,
  random_state=1991
)

# Run the MICE algorithm for 3 iterations on each of the datasets
kernel.mice(3)

print(kernel)

#%%
# Run the MICE algorithm for 2 more iterations on the kernel,
kernel.mice(2,n_jobs=2)

#%% Creating a Custom Imputation Schema
var_sch = {
    'sepal width (cm)': ['target','petal width (cm)'],
    'petal width (cm)': ['target','sepal length (cm)']
}
var_mmc = {
    'sepal width (cm)': 5,
    'petal width (cm)': 0
}

cust_kernel = mf.MultipleImputedKernel(
    iris_amp,
    datasets=3,
    variable_schema=var_sch,
    mean_match_candidates=var_mmc
)
cust_kernel.mice(2)

#%% Imputing New Data with Existing Models
# Our 'new data' is just the first 15 rows of iris_amp
new_data = iris_amp.iloc[range(15)]
new_data_imputed = kernel.impute_new_data(new_data=new_data)
print(new_data_imputed)
print(new_data_imputed.complete_data(1))  # 0,1,2,3
#%% Distribution of Imputed-Values
kernel.plot_imputed_distributions(wspace=0.3,hspace=0.3)
plt.show()

#%% Convergence of Correlation
kernel.plot_correlations()
plt.show()

#%% Variable Importance
kernel.plot_feature_importance(annot=True,cmap="YlGnBu",vmin=0, vmax=1)
plt.show()

#%% Mean Convergence
kernel.plot_mean_convergence(wspace=0.3, hspace=0.4)
plt.show()

#%%
acclist = []
for iteration in range(kernel.iteration_count() + 1):
    target_na_count = kernel.na_counts['petal width (cm)']
    compdat = kernel.complete_data(dataset=0, iteration=iteration)

    # Record the accuract of the imputations of target.
    acclist.append(
        round(1 - sum(compdat['petal width (cm)'] != iris['petal width (cm)']) / target_na_count, 2)
    )

# acclist shows the accuracy of the imputations
# over the iterations.
print(acclist)
