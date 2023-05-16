# This is the Githib repo for the FynnLabs internship where the code which is in the 'R' language of Mcdonals case should be converted to Python language 
import pandas as pd

mcdonalds = pd.read_csv('https://raw.githubusercontent.com/terrytangyuan/MSA/master/data/mcdonalds.csv')
print(list(mcdonalds.columns))
print(mcdonalds.shape)

mcdonalds = pd.read_csv('mcdonalds.csv')
print(mcdonalds.head(3))

MD_x = np.array(mcdonalds.iloc[:, 0:11])
MD_x = (MD_x == "Yes").astype(int)
np.round(np.mean(MD_x, axis=0), 2)

from sklearn.decomposition import PCA
import numpy as np

MD_pca = PCA().fit(MD_x)
print("Importance of components:")
print(np.round(MD_pca.explained_variance_ratio_, 4))

import flexclust
import matplotlib.pyplot as plt

MD_pca = flexclust.PCA(MD)
plt.scatter(MD_pca.predict(), color="grey")
MD_pca.projAxes()
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

np.random.seed(1234)

# Assuming MD.x is your data in Python

# Step 1: Finding the optimal number of clusters
silhouette_scores = []
k_values = range(2, 9)

for k in k_values:
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=1234)
    kmeans.fit(MD.x)
    labels = kmeans.labels_
    score = silhouette_score(MD.x, labels)
    silhouette_scores.append(score)

best_k = k_values[np.argmax(silhouette_scores)]
print("Best number of clusters:", best_k)

# Step 2: Clustering with the best number of clusters
kmeans = KMeans(n_clusters=best_k, n_init=10, random_state=1234)
kmeans.fit(MD.x)
labels = kmeans.labels_
MD_km28 = labels

import matplotlib.pyplot as plt

# Assuming MD_km28 contains the cluster labels from the previous code

# Plotting the clusters
plt.plot(k_values, silhouette_scores, marker='o')
plt.xlabel("Number of segments")
plt.ylabel("Silhouette score")
plt.title("Silhouette score vs. Number of segments")
plt.show()

import numpy as np
from sklearn.cluster import KMeans
from sklearn.utils import resample

np.random.seed(1234)

# Assuming MD.x is your data in Python

# Step 1: Bootstrapping with different number of clusters
n_rep = 10
n_boot = 100
k_values = range(2, 9)
bootstrapped_results = []

for k in k_values:
    cluster_results = []
    
    for _ in range(n_rep):
        bootstrap_samples = []
        
        for _ in range(n_boot):
            bootstrap_sample = resample(MD.x, random_state=1234)
            bootstrap_samples.append(bootstrap_sample)
        
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=1234)
        kmeans.fit(bootstrap_samples[-1])  # Using the last bootstrap sample
        labels = kmeans.labels_
        cluster_results.append(labels)
    
    bootstrapped_results.append(cluster_results)

# Now you have a 3D list: bootstrapped_results[i][j] contains the cluster labels
# for the j-th replication with i+2 number of clusters.

# Accessing the results for a specific number of clusters and replication:
# Example: MD_b28 = bootstrapped_results[6][0]  # 8 clusters, first replication

import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score

# Assuming bootstrapped_results contains the bootstrapped cluster labels

# Plotting the adjusted Rand index
plt.figure(figsize=(10, 6))
for i, k in enumerate(k_values):
    rands = []
    for j in range(n_rep):
        true_labels = bootstrapped_results[i][j]  # True labels from bootstrapping
        rand_index = adjusted_rand_score(MD_km28, true_labels)
        rands.append(rand_index)
    plt.plot([k] * n_rep, rands, "o", color="blue", alpha=0.5)
plt.xlabel("Number of segments")
plt.ylabel("Adjusted Rand index")
plt.title("Adjusted Rand index vs. Number of segments")
plt.show()

import matplotlib.pyplot as plt

# Assuming MD_km28 contains the cluster labels from the clustering step
# Assuming MD.x is your data in Python

# Filter data points belonging to cluster "4"
cluster_4_indices = np.where(MD_km28 == 3)  # Adjust index based on 0-based or 1-based labeling
cluster_4_data = MD.x[cluster_4_indices]

# Plotting the histogram
plt.hist(cluster_4_data, bins=20, range=[0, 1], edgecolor='black')
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.title("Histogram of Cluster 4")
plt.xlim(0, 1)
plt.show()

# Assuming MD_km28 contains the cluster labels from the clustering step

# Extracting data points belonging to cluster "4"
MD_k4 = MD.x[MD_km28 == 3]  # Adjust index based on 0-based or 1-based labeling

import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri

# Convert MD_x to R dataframe
pandas2ri.activate()
r_data = pandas2ri.py2rpy_dataframe(MD_x)

# Convert MD_k4 to R vector
r_cluster_labels = robjects.vectors.IntVector(MD_k4)

# Load required packages
robjects.r('library(flexmix)')

# Perform supervised learning with FlexMix
robjects.r.assign('data', r_data)
robjects.r.assign('cluster_labels', r_cluster_labels)
robjects.r('MD_r4 <- slswFlexclust(data, cluster_labels)')

# Convert MD_r4 back to a pandas DataFrame
r_data_output = robjects.r('MD_r4')
pandas_data_output = pandas2ri.rpy2py_dataframe(r_data_output)

import matplotlib.pyplot as plt

# Assuming pandas_data_output contains the supervised learning results

# Extracting the segment stability values
segment_numbers = pandas_data_output['segment number']
segment_stability = pandas_data_output['segment stability']

# Plotting the segment stability
plt.plot(segment_numbers, segment_stability)
plt.xlabel("Segment number")
plt.ylabel("Segment stability")
plt.title("Segment Stability")
plt.ylim(0, 1)
plt.show()

import rpy2.robjects as robjects

# Load the flexmix package
robjects.r('library(flexmix)')

# Convert MD_x to an R matrix
r_data = robjects.r.matrix(MD_x, nrow=MD_x.shape[0], ncol=MD_x.shape[1])

# Set the random seed
robjects.r('set.seed(1234)')

# Perform model-based clustering with stepFlexmix
robjects.r('MD_m28 <- stepFlexmix(MD.x ~ 1, model = FLXMCmvbinary(), k = 2:8, nrep = 10, verbose = FALSE)')

# Print the results
robjects.r('print(MD_m28)')

import matplotlib.pyplot as plt

# Assuming MD_m28 contains the model-based clustering results

# Extracting the information criteria values
aic_values = MD_m28.rx2('ICL')
bic_values = MD_m28.rx2('BIC')
icl_values = MD_m28.rx2('ICL')

# Plotting the information criteria values
k_values = range(2, 9)
plt.plot(k_values, aic_values, label='AIC', marker='o')
plt.plot(k_values, bic_values, label='BIC', marker='o')
plt.plot(k_values, icl_values, label='ICL', marker='o')
plt.xlabel("Number of clusters (k)")
plt.ylabel("Value of information criteria")
plt.title("Information Criteria")
plt.legend()
plt.show()

import pandas as pd

# Assuming MD_m28 and MD_k4 contain the model-based clustering and k-means clustering results, respectively

# Extract the model for cluster "4" from MD_m28
MD_m4 = MD_m28.rx2("models")[3]

# Get the cluster labels for k-means and MD_m4
kmeans_labels = clusters(MD_k4)
mixture_labels = clusters(MD_m4)

# Create a contingency table
contingency_table = pd.crosstab(kmeans_labels, mixture_labels, rownames=["kmeans"], colnames=["mixture"])

# Display the contingency table
print(contingency_table)

import pandas as pd
import rpy2.robjects as robjects

# Assuming MD_x and MD_k4 contain the data and k-means clustering labels, respectively

# Convert MD_x to an R dataframe
r_data = robjects.r['data.frame'](MD_x)

# Get the k-means clustering labels
r_kmeans_labels = robjects.vectors.IntVector(MD_k4)

# Fit FlexMix model using k-means labels
robjects.r('library(flexmix)')
robjects.r('MD_m4a <- flexmix(MD.x ~ 1, cluster = clusters(MD_k4), model = FLXMCmvbinary())')

# Get the cluster labels from the new FlexMix model
r_mixture_labels = robjects.r['clusters'](robjects.r['MD_m4a'])

# Convert cluster labels to Python
kmeans_labels = list(r_kmeans_labels)
mixture_labels = list(r_mixture_labels)

# Create a contingency table
contingency_table = pd.crosstab(kmeans_labels, mixture_labels, rownames=["kmeans"], colnames=["mixture"])

# Display the contingency table
print(contingency_table)

import rpy2.robjects as robjects

# Assuming MD_m4a and MD_m4 contain the FlexMix models

# Compute the log-likelihood for MD_m4a
loglik_m4a = robjects.r('logLik(MD_m4a)')[0]

# Compute the log-likelihood for MD_m4
loglik_m4 = robjects.r('logLik(MD_m4)')[0]

# Print the log-likelihood values
print("MD_m4a log-likelihood:", loglik_m4a)
print("MD_m4 log-likelihood:", loglik_m4)

import numpy as np

# Assuming mcdonalds is a pandas DataFrame with a column named 'Like'

# Get the frequency table of the 'Like' column
table = mcdonalds['Like'].value_counts()

# Reverse the order of the table
reversed_table = np.flip(table.values)

# Print the reversed table
print(reversed_table)

import pandas as pd

# Assuming mcdonalds is a pandas DataFrame with a column named 'Like'

# Create a new column 'Like.n' in the DataFrame
mcdonalds['Like.n'] = 6 - mcdonalds['Like'].astype(int)

# Compute the frequency table for the 'Like.n' column
frequency_table = mcdonalds['Like.n'].value_counts()

# Print the frequency table
print(frequency_table)

import pandas as pd
import statsmodels.formula.api as smf

# Assuming mcdonalds is a pandas DataFrame with the relevant columns

# Get the names of the first 11 columns and collapse them into a string
column_names = ' + '.join(mcdonalds.columns[0:11])

# Create the formula string
formula_string = 'Like.n ~ ' + column_names

# Create the formula object
formula = smf.ols(formula_string, data=mcdonalds)

# Print the formula
print(formula)

import numpy as np
import pandas as pd
from rpy2.robjects import r, pandas2ri

# Assuming mcdonalds is a pandas DataFrame with the relevant columns

# Set the seed for reproducibility
np.random.seed(1234)

# Convert the pandas DataFrame to an R dataframe
pandas2ri.activate()
r_mcdonalds = pandas2ri.py2ri(mcdonalds)

# Define the formula
formula = 'Like.n ~ yummy + convenient + spicy + fattening + greasy + fast + cheap + tasty + expensive + healthy + disgusting'

# Fit the FlexMix model with variable selection
r('library(flexmix)')
r('MD.reg2 <- stepFlexmix(' + formula + ', data = r_mcdonalds, k = 2, nrep = 10, verbose = FALSE)')

# Print the FlexMix model results
r('print(MD.reg2)')

import rpy2.robjects as robjects

# Refit the FlexMix model
robjects.r('MD.ref2 <- refit(MD.reg2)')

# Get the summary of the refitted model
summary = robjects.r('summary(MD.ref2)')

# Print the summary
print(summary)

import rpy2.robjects as robjects
import rpy2.robjects.packages as rpackages

# Install and import the "flexmix" R package
rpackages.importr('flexmix')

# Plot the refitted model with significance indications
robjects.r('plot(MD.ref2, significance = TRUE)')

import numpy as np
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage

# Assuming MD_x is a numpy array containing the data in MD.x

# Transpose the matrix MD_x
MD_x_transposed = np.transpose(MD_x)

# Compute the pairwise distance matrix
dist_matrix = pdist(MD_x_transposed)

# Perform hierarchical clustering
MD_vclust = linkage(dist_matrix)

import matplotlib.pyplot as plt
import numpy as np

# Assuming MD_k4 is a list or array containing the data in MD.k4
# Assuming MD_vclust is a hierarchical clustering object obtained using scipy

# Reverse the order of MD.vclust$order
reversed_order = np.flip(MD_vclust['leaves'])

# Create a bar chart with shading
plt.bar(range(len(MD_k4)), MD_k4[reversed_order], color='gray', alpha=0.5)

# Set the x-axis tick labels
plt.xticks(range(len(MD_k4)))

# Set the x-axis label
plt.xlabel('Segment number')

# Set the y-axis label
plt.ylabel('Frequency')

# Show the plot
plt.show()

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

# Assuming MD_k4 is a list or array containing the data in MD.k4
# Assuming MD_pca is a PCA object obtained using sklearn
# Assuming MD_x is a numpy array containing the data in MD.x

# Project the data onto the PCA components
projected_data = MD_pca.transform(MD_x)

# Create a scatter plot with projected data
plt.scatter(projected_data[:, 0], projected_data[:, 1], c=MD_k4)

# Disable convex hull and similarity lines
plt.gca().set_hull(None)
plt.gca().set_simlines(None)

# Set the x-axis and y-axis labels
plt.xlabel('principal component 1')
plt.ylabel('principal component 2')

# Show the plot
plt.show()

# Get the projection axes
proj_axes = MD_pca.components_.T

# Print the projection axes
print(proj_axes)

import matplotlib.pyplot as plt
import pandas as pd

# Assuming k4 is a list or array containing the data in MD.k4
# Assuming mcdonalds is a pandas DataFrame with the relevant columns

# Convert k4 and mcdonalds$Like to pandas Series
k4_series = pd.Series(k4)
like_series = mcdonalds['Like']

# Create a cross-tabulation table
cross_table = pd.crosstab(k4_series, like_series)

# Create a mosaic plot with shading
plt.rcParams["font.size"] = 10  # Adjust the font size if needed
cross_table.plot(kind='bar', stacked=True, cmap='gray', alpha=0.5)

# Set the title and x-axis label
plt.title('')
plt.xlabel('Segment number')

# Remove the y-axis label
plt.ylabel('')

# Show the plot
plt.show()

import matplotlib.pyplot as plt
import pandas as pd

# Assuming k4 is a list or array containing the data in MD.k4
# Assuming mcdonalds is a pandas DataFrame with the relevant columns

# Convert k4 and mcdonalds$Gender to pandas Series
k4_series = pd.Series(k4)
gender_series = mcdonalds['Gender']

# Create a cross-tabulation table
cross_table = pd.crosstab(k4_series, gender_series)

# Create a mosaic plot with shading
plt.rcParams["font.size"] = 10  # Adjust the font size if needed
cross_table.plot(kind='bar', stacked=True, cmap='gray', alpha=0.5)

# Set the title
plt.title('')

# Show the plot
plt.show()

import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn import tree

# Assuming mcdonalds is a pandas DataFrame with the relevant columns
# Assuming k4 is a list or array containing the data in MD.k4

# Create a copy of the mcdonalds DataFrame
data = mcdonalds.copy()

# Convert categorical variables to numeric using LabelEncoder
label_encoder = LabelEncoder()
for column in data.columns:
    if data[column].dtype == np.object:
        data[column] = label_encoder.fit_transform(data[column])

# Create input features X and target variable y
X = data[['Like.n', 'Age', 'VisitFrequency', 'Gender']]
y = (k4 == 3)

# Create a decision tree classifier
clf = DecisionTreeClassifier()
clf.fit(X, y)

# Plot the decision tree
fig, ax = plt.subplots(figsize=(12, 12))
tree.plot_tree(clf, feature_names=X.columns, class_names=['False', 'True'], filled=True, ax=ax)

# Show the plot
plt.show()

import numpy as np
import pandas as pd

# Assuming mcdonalds is a pandas DataFrame with the relevant columns
# Assuming k4 is a list or array containing the data in MD.k4

# Convert k4 and mcdonalds['VisitFrequency'] to pandas Series
k4_series = pd.Series(k4)
visit_frequency_series = mcdonalds['VisitFrequency']

# Calculate the mean of VisitFrequency for each segment (k4)
visit_mean = visit_frequency_series.groupby(k4_series).mean()

# Convert the result to a dictionary
visit_mean_dict = visit_mean.to_dict()

# Display the mean of VisitFrequency for each segment
for segment, mean_value in visit_mean_dict.items():
    print(f'Segment {segment}: {mean_value}')

import numpy as np
import pandas as pd

# Assuming mcdonalds is a pandas DataFrame with the relevant columns
# Assuming k4 is a list or array containing the data in MD.k4

# Convert k4 and mcdonalds['Like.n'] to pandas Series
k4_series = pd.Series(k4)
like_n_series = mcdonalds['Like.n']

# Calculate the mean of Like.n for each segment (k4)
like_mean = like_n_series.groupby(k4_series).mean()

# Convert the result to a dictionary
like_mean_dict = like_mean.to_dict()

# Display the mean of Like.n for each segment
for segment, mean_value in like_mean_dict.items():
    print(f'Segment {segment}: {mean_value}')

import numpy as np
import pandas as pd

# Assuming mcdonalds is a pandas DataFrame with the relevant columns
# Assuming k4 is a list or array containing the data in MD.k4

# Convert k4 and mcdonalds['Gender'] to pandas Series
k4_series = pd.Series(k4)
gender_series = mcdonalds['Gender']

# Convert gender to binary values (0 for "Male" and 1 for "Female")
gender_binary = (gender_series == "Female").astype(int)

# Calculate the proportion of females for each segment (k4)
female_proportion = gender_binary.groupby(k4_series).mean()

# Convert the result to a dictionary
female_proportion_dict = female_proportion.to_dict()

# Display the proportion of females for each segment
for segment, proportion_value in female_proportion_dict.items():
    print(f'Segment {segment}: {proportion_value}')

import matplotlib.pyplot as plt

# Assuming visit, like, and female are lists or arrays

# Set up the figure and axes
fig, ax = plt.subplots()

# Plot the scatter points with marker sizes proportional to female
scatter = ax.scatter(visit, like, c=range(1, 5), s=10 * female)

# Set the x-axis and y-axis limits
ax.set_xlim(2, 4.5)
ax.set_ylim(-3, 3)

# Add text labels for each segment
for i, txt in enumerate(range(1, 5)):
    ax.text(visit[i], like[i], txt)

# Add a colorbar legend
cbar = fig.colorbar(scatter, ticks=range(1, 5))
cbar.set_label('Segment')

# Add labels and title
ax.set_xlabel('Visit Frequency')
ax.set_ylabel('Like')
ax.set_title('Scatter Plot')

# Display the plot
plt.show()


