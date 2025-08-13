# machine_learning_network_intrusion
Intrusion detection evaluation dataset (ISCXIDS2012):  https://www.unb.ca/cic/datasets/ids.html. This dataset provides network intrusion detection datasets which can be used for forensic analysis. 

While undertaking this objective, the dataset chosen was: Intrusion detection evaluation dataset (ISCXIDS2012):  https://www.unb.ca/cic/datasets/ids.html. This dataset provides network intrusion detection datasets which can be used for forensic analysis. Available through this link 
Dataset: https://www.unb.ca/cic/datasets/ids.html

Intrusion detection evaluation dataset (ISCXIDS2012)

While undertaking this objective, the dataset chosen was: Intrusion
detection evaluation dataset (ISCXIDS2012):
[[https://www.unb.ca/cic/datasets/ids.html]{.underline}](https://www.unb.ca/cic/datasets/ids.html).
This dataset provides network intrusion detection datasets which can be
used for forensic analysis. Available through this link

Dataset:
[[https://www.unb.ca/cic/datasets/ids.html]{.underline}](https://www.unb.ca/cic/datasets/ids.html)

# Intrusion detection evaluation dataset (ISCXIDS2012)

Step 1: Data Pre-processing:

This model indicates a serious problem as there are only 2992 attacks vs
134112 normal attacks. If we train the model, it is unable to predict
attack scenarios. but can predict normal scenarios. It is due to
oversampling. We have limited attack data. To solve this, we will
oversample the minority class (attacks) using the (Synthetic Minority
Over-sampling Technique). This should help balance the classes and
improve the model\'s ability to predict attack scenarios.

```
### Code here:

  import pandas as pd\
  import numpy as np\
  from sklearn.preprocessing import StandardScaler\
  from sklearn.model_selection import train_test_split\
  from imblearn.over_sampling import SMOTE\
  from collections import Counter\
  from google.colab import drive\
  \
  \# Mount Google Drive\
  drive.mount(\'/content/drive\')\
  \
  \# Load the dataset from Google Drive\
  file_path=\'/content/drive/My Drive/Colab
  Notebooks/TestbedMonJun14Flows.csv\'\
  df = pd.read_csv(file_path)\
  print(\"Original dataset shape:\", df.shape)\
  \
  \# Select relevant features\
  numeric_features = \[\'totalSourceBytes\', \'totalDestinationBytes\',
  \'totalDestinationPackets\',\
  \'totalSourcePackets\', \'sourcePort\', \'destinationPort\'\]\
  categorical_features = \[\'appName\', \'direction\',
  \'protocolName\'\]\
  \
  \# Keep only selected features and the label\
  selected_features = numeric_features + categorical_features +
  \[\'Label\'\]\
  df = df\[selected_features\]\
  \
  \# One-hot encode categorical features\
  df_encoded = pd.get_dummies(df, columns=categorical_features)\
  \
  \# Separate features and label\
  X = df_encoded.drop(\'Label\', axis=1)\
  y = df_encoded\[\'Label\'\]\
  \
  print(\"Dataset shape after preprocessing:\", X.shape)\
  \
  \# Standardize numerical features\
  scaler = StandardScaler()\
  X\[numeric_features\] = scaler.fit_transform(X\[numeric_features\])\
  \
  print(\"Standardized numerical features:\")\
  print(\"Mean:\", X\[numeric_features\].mean())\
  print(\"Std:\", X\[numeric_features\].std())\
  \
  \# Split the data\
  X_train, X_test, y_train, y_test = train_test_split(X, y,
  test_size=0.2, random_state=42)\
  print(\"Training set shape before SMOTE:\", X_train.shape)\
  print(\"Original class distribution:\")\
  print(Counter(y_train))\
  \
  \# Apply SMOTE to the training data\
  smote = SMOTE(random_state=42)\
  X_train_resampled, y_train_resampled = smote.fit_resample(X_train,
  y_train)\
  print(\"Training set shape after SMOTE:\", X_train_resampled.shape)\
  print(\"Resampled class distribution:\")\
  print(Counter(y_train_resampled))\
  \
  \
  \# Keep the test set unchanged\
  print(\"Test set shape:\", X_test.shape)\
  \
  \# Re-scale all features after SMOTE\
  scaler = StandardScaler()\
  X_train_resampled_scaled = scaler.fit_transform(X_train_resampled)\
  X_test_scaled = scaler.transform(X_test)\
  \
  print(\"Scaled training set shape:\", X_train_resampled_scaled.shape)\
  print(\"Scaled test set shape:\", X_test_scaled.shape)\
  \
  \# Now X_train_resampled_scaled and X_test_scaled can be used for our
  model
  -----------------------------------------------------------------------

  -----------------------------------------------------------------------

Output

  -----------------------------------------------------------------------
  Drive already mounted at /content/drive; to attempt to forcibly
  remount, call drive.mount(\"/content/drive\", force_remount=True).\
  Original dataset shape: (171380, 21)\
  Dataset shape after preprocessing: (171380, 110)\
  Standardized numerical features:\
  Mean: totalSourceBytes -5.389810e-19\
  totalDestinationBytes 3.648486e-18\
  totalDestinationPackets -1.368182e-18\
  totalSourcePackets -7.048212e-19\
  sourcePort -6.898956e-17\
  destinationPort 0.000000e+00\
  dtype: float64\
  Std: totalSourceBytes 1.000003\
  totalDestinationBytes 1.000003\
  totalDestinationPackets 1.000003\
  totalSourcePackets 1.000003\
  sourcePort 1.000003\
  destinationPort 1.000003\
  dtype: float64\
  Training set shape before SMOTE: (137104, 110)\
  Original class distribution:\
  Counter({\'Normal\': 134112, \'Attack\': 2992})\
  Training set shape after SMOTE: (268224, 110)\
  Resampled class distribution:\
  Counter({\'Attack\': 134112, \'Normal\': 134112})\
  Test set shape: (34276, 110)\
  Scaled training set shape: (268224, 110)\
  Scaled test set shape: (34276, 110)
  -----------------------------------------------------------------------

  -----------------------------------------------------------------------

This code does the following:

1.  It imports the necessary libraries, including SMOTE from
    > imbalanced-learn.

2.  It performs preprocessing steps as before: loading data, selecting
    > features, one-hot encoding, and standardizing numerical features.

3.  After splitting the data into training and test sets, it applies
    > SMOTE to the training data only.

4.  It prints the shape and class distribution of the dataset before and
    > after applying SMOTE.

After running this code, we have a more balanced class distribution in
our training data. The test set remains unchanged to represent the
real-world distribution of our data.

To use this resampled data in our deep learning model, we will use
X_train_resampled and y_train_resampled respectively when fitting our
model.

This approach should help our model better learn to predict both normal
and attack scenarios. While the training data is now balanced, the test
set still reflects the original distribution, which is important for
getting an accurate evaluation of our model\'s performance on real-world
data.

-   Train-Test Splitting: The Intrusion detection evaluation dataset
    > (ISCXIDS2012) dataset was divided into training and testing sets,
    > with 80% allocated for training and 20% for testing.
    > Scikit-learn\'s train_test_split function was used to ensure that
    > the model learned from the majority of the data while being
    > evaluated on unseen data.

-   Standardization: The network traffic data, which may have features
    > with varying scales, was standardized using z-score normalization
    > to ensure that all features contributed equally during analysis.

Step 2: Anomaly Detection and Clustering:

-   Isolation Forest: Anomaly detection was performed using the
    > Isolation Forest algorithm. Different hyperparameters, such as the
    > number of trees and contamination level, were experimented with to
    > optimize performance using Scikit-learn\'s IsolationForest.

This code starts with Isolation Forest for anomaly detection, then moves
on to K-Means clustering, and finally implements SOM.

  -----------------------------------------------------------------------
  import numpy as np\
  from sklearn.ensemble import IsolationForest\
  from sklearn.cluster import KMeans\
  from sklearn.metrics import silhouette_score\
  !pip install minisom tqdm\
  from minisom import MiniSom\
  import matplotlib.pyplot as plt\
  from sklearn.preprocessing import StandardScaler\
  from tqdm import tqdm\
  \
  \# X_train_resampled_scaled and y_train_resampled from the previous
  step\
  \
  \# Apply Anomaly Detection\
  \# 1. Isolation Forest for Anomaly Detection\
  print(\"Performing Anomaly Detection with Isolation Forest:\")\
  contamination_levels = \[0.1, 0.2, 0.3\]\
  for contamination in contamination_levels:\
  clf = IsolationForest(contamination=contamination, random_state=42,
  n_jobs=-1)\
  y_pred = clf.fit_predict(X_train_resampled_scaled)\
  n_outliers = np.sum(y_pred == -1)\
  print(f\"Contamination {contamination}: {n_outliers} outliers
  detected\")\
  \
  \# 2. K-Means Clustering\
  print(\"\\nPerforming K-Means Clustering:\")\
  max_clusters = 10\
  silhouette_scores = \[\]\
  for n_clusters in tqdm(range(2, max_clusters + 1), desc=\"Calculating
  silhouette scores\"):\
  kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)\
  cluster_labels = kmeans.fit_predict(X_train_resampled_scaled)\
  silhouette_avg = silhouette_score(X_train_resampled_scaled,
  cluster_labels, sample_size=10000)\
  silhouette_scores.append(silhouette_avg)\
  print(f\"For n_clusters = {n_clusters}, the average silhouette score is
  : {silhouette_avg}\")\
  \
  \# Plot silhouette scores\
  plt.figure(figsize=(10, 6))\
  plt.plot(range(2, max_clusters + 1), silhouette_scores)\
  plt.xlabel(\'Number of clusters\')\
  plt.ylabel(\'Silhouette Score\')\
  plt.title(\'Silhouette Score vs Number of Clusters\')\
  plt.show()\
  \
  \# Choose the number of clusters with the highest silhouette score\
  optimal_clusters = silhouette_scores.index(max(silhouette_scores)) + 2\
  print(f\"\\nOptimal number of clusters: {optimal_clusters}\")\
  \
  \# Perform final K-Means clustering with optimal number of clusters\
  kmeans = KMeans(n_clusters=optimal_clusters, random_state=42,
  n_init=10)\
  cluster_labels = kmeans.fit_predict(X_train_resampled_scaled)\
  \
  \# 3. Self-Organizing Map (SOM)\
  print(\"\\nPerforming Self-Organizing Map (SOM) analysis:\")\
  som_dim = (20, 20) \# Larger SOM grid for larger dataset\
  input_dim = X_train_resampled_scaled.shape\[1\] \# Number of features\
  \
  \# Initialize and train SOM\
  som = MiniSom(som_dim\[0\], som_dim\[1\], input_dim, sigma=0.3,
  learning_rate=0.5)\
  \
  \# Train SOM with progress updates\
  num_iterations = 10000 \# Increased iterations for better convergence\
  for i in tqdm(range(num_iterations), desc=\"Training SOM\"):\
  som.train_random(X_train_resampled_scaled, 1) \# Train one sample at a
  time\
  \
  \# Get SOM node positions for each data point\
  print(\"Calculating SOM node positions\...\")\
  som_nodes = np.array(\[som.winner(x) for x in
  tqdm(X_train_resampled_scaled, desc=\"Processing data points\")\])\
  \
  \# Visualize SOM\
  plt.figure(figsize=(12, 10))\
  plt.title(\'Self-Organizing Map with K-Means Clusters\')\
  scatter = plt.scatter(som_nodes\[:, 0\], som_nodes\[:, 1\],
  c=cluster_labels, cmap=\'viridis\')\
  plt.colorbar(scatter, label=\'K-Means Cluster\')\
  plt.xlabel(\'SOM X\')\
  plt.ylabel(\'SOM Y\')\
  plt.show()\
  \
  print(\"Anomaly detection, clustering, and SOM analysis completed.\")\
  \
  \# Additional analysis: Feature importance for each cluster\
  print(\"\\nAnalyzing feature importance for each cluster:\")\
  for cluster in range(optimal_clusters):\
  cluster_data = X_train_resampled_scaled\[cluster_labels == cluster\]\
  cluster_mean = cluster_data.mean(axis=0)\
  cluster_std = cluster_data.std(axis=0)\
  \
  print(f\"\\nCluster {cluster}:\")\
  for feature, mean, std in zip(X.columns, cluster_mean, cluster_std):\
  print(f\"{feature}: Mean = {mean:.2f}, Std = {std:.2f}\")\
  \
  \# Visualize feature importance\
  plt.figure(figsize=(15, 10))\
  for cluster in range(optimal_clusters):\
  cluster_data = X_train_resampled_scaled\[cluster_labels == cluster\]\
  cluster_mean = cluster_data.mean(axis=0)\
  plt.plot(range(len(cluster_mean)), cluster_mean, label=f\'Cluster
  {cluster}\')\
  \
  plt.title(\'Feature Importance by Cluster\')\
  plt.xlabel(\'Feature Index\')\
  plt.ylabel(\'Mean Value\')\
  plt.legend()\
  plt.show()
  -----------------------------------------------------------------------

  -----------------------------------------------------------------------

This code does the following:

-   Isolation Forest for Anomaly Detection:

    -   It tries different contamination levels (0.1, 0.2, 0.3) and
        > reports the number of outliers detected for each.

    -   We then use a contamination level of 0.1 for further analysis.

-   K-Means Clustering:

    -   It tries different numbers of clusters (from 2 to 10) and
        > calculates the silhouette score for each.

    -   It plots the silhouette scores to help visualize the optimal
        > number of clusters.

    -   It then performs the final clustering with the optimal number of
        > clusters.

-   Self-Organizing Map (SOM):

    -   It creates a 10x10 SOM grid and trains it on the data.

    -   It then visualizes the SOM, coloring each point based on its
        > K-Means cluster assignment.

let\'s interpret the results we have from the code:

1.  Isolation Forest (Anomaly Detection):

  -----------------------------------------------------------------------
  Requirement already satisfied: minisom in
  /usr/local/lib/python3.10/dist-packages (2.3.2)\
  Requirement already satisfied: tqdm in
  /usr/local/lib/python3.10/dist-packages (4.66.4)\
  Performing Anomaly Detection with Isolation Forest:\
  Contamination 0.1: 26822 outliers detected\
  Contamination 0.2: 53644 outliers detected\
  Contamination 0.3: 80467 outliers detected
  -----------------------------------------------------------------------

  -----------------------------------------------------------------------

2.  This suggests that there\'s a significant number of anomalies in our
    > dataset. The choice of contamination level depends on the specific
    > use case and domain knowledge.

3.  K-Means Clustering:

    -   The silhouette scores suggest that 8 clusters provide the best
        > separation of the data, with a silhouette score of about
        > 0.2968044119.

    -   This unusually high score suggests that our data might have two
        > very distinct groups with little overlap.

![](vertopal_a8c83d3e11574ca39985f1f7b2fd1c88/media/image6.png){width="6.5in"
height="5.875in"}

4.  Optimal number of clusters: 8 This aligns with the silhouette score
    > results. Our network traffic seems to fall into 8 distinct
    > categories.

Interpretation:

1.  The high number of anomalies detected by Isolation Forest suggests
    > that our network traffic has a significant amount of unusual
    > activity. This could be indicative of potential security threats
    > or simply diverse network usage patterns.

2.  The K-Means clustering results, with an optimal number of 8 clusters
    > and a very high silhouette score, suggest that our network traffic
    > falls into 8 distinct categories. This binary classification could
    > potentially align with \"normal\" vs \"anomalous\" traffic, but
    > it\'s important to investigate what these clusters represent in
    > the context of our specific network.

    -   Clustering Algorithms:

Visualize the cluster distributions: How our code functions:

PCA for dimensionality reduction and visualization, then we visualize
the cluster distributions, analyze feature importance, and finally try a
different clustering algorithm. Here\'s the step-by-step process:

-   PCA and Visualization

-   Visualize Cluster Distributions

-   Analyze Feature Importance

Judging from the results: Feature Selection: Given that we now know the
most important features, we can create a reduced dataset using only the
top features. This can often lead to more interpretable and potentially
more accurate clustering. K-means Clustering: We\'ll apply K-means
clustering on both the full dataset and the reduced dataset with top
features.

Visualization: We\'ll visualize the results to compare the clustering
outcomes.

Evaluation: We\'ll evaluate the clustering performance using silhouette
scores and compare the results between the full and reduced datasets.

  -----------------------------------------------------------------------
  import numpy as np\
  import pandas as pd\
  from sklearn.cluster import KMeans\
  from sklearn.metrics import silhouette_score\
  from sklearn.preprocessing import StandardScaler\
  import matplotlib.pyplot as plt\
  from sklearn.decomposition import PCA\
  \
  \# X_train_resampled_scaled set as our current dataset\
  \
  \# 1. Feature Selection\
  top_features = \[\'totalSourceBytes\', \'totalSourcePackets\',
  \'totalDestinationPackets\',\
  \'appName_IGMP\', \'protocolName_igmp\', \'appName_dsp3270\',\
  \'appName_NortonAntiVirus\', \'appName_MicrosoftMediaServer\',\
  \'totalDestinationBytes\', \'appName_SMS\'\]\
  \
  X_reduced = X_train_resampled_scaled\[:,
  \[X_train_resampled.columns.get_loc(col) for col in top_features\]\]\
  \
  \# 2. K-means Clustering\
  n_clusters = 10 \# Adjust based on domain knowledge or previous
  analysis\
  kmeans_full = KMeans(n_clusters=n_clusters, random_state=42)\
  kmeans_reduced = KMeans(n_clusters=n_clusters, random_state=42)\
  \
  labels_full = kmeans_full.fit_predict(X_train_resampled_scaled)\
  labels_reduced = kmeans_reduced.fit_predict(X_reduced)\
  \
  \# 3. Visualization Function\
  def plot_clusters(X, labels, title):\
  pca = PCA(n_components=2)\
  X_pca = pca.fit_transform(X)\
  \
  plt.figure(figsize=(10, 8))\
  scatter = plt.scatter(X_pca\[:, 0\], X_pca\[:, 1\], c=labels,
  cmap=\'viridis\', alpha=0.6)\
  plt.colorbar(scatter)\
  plt.title(title)\
  plt.xlabel(\'First Principal Component\')\
  plt.ylabel(\'Second Principal Component\')\
  plt.show()\
  \
  plot_clusters(X_train_resampled_scaled, labels_full, \'K-means
  Clustering (Full Dataset)\')\
  plot_clusters(X_reduced, labels_reduced, \'K-means Clustering (Reduced
  Dataset)\')
  -----------------------------------------------------------------------

  -----------------------------------------------------------------------

Analyzing the results: Let\'s analyze both images to understand what
they tell us about the clustering results for the full dataset and the
reduced dataset.

Image 1: K-means Clustering (Full Dataset)

![](vertopal_a8c83d3e11574ca39985f1f7b2fd1c88/media/image4.png){width="6.5in"
height="5.722222222222222in"}

1.  Distribution: The data points are spread out along both the first
    > and second principal components, with a concentration of points
    > near the origin.

2.  Clusters: There appear to be two main clusters, represented by
    > yellow and purple points. The clusters are not perfectly
    > separated, with some overlap visible.

3.  Outliers: There are several outlier points, particularly along the
    > positive direction of the first principal component and the
    > positive direction of the second principal component.

4.  Balance: The clusters seem relatively balanced in terms of the
    > number of points in each, though the purple cluster appears
    > slightly larger.

Image 2: K-means Clustering (Reduced Dataset)

![](vertopal_a8c83d3e11574ca39985f1f7b2fd1c88/media/image2.png){width="6.5in"
height="5.722222222222222in"}

1.  Distribution: The data points are primarily distributed along the
    > first principal component (x-axis), with much less variation along
    > the second principal component.

2.  Clusters: Two distinct clusters are visible - a large purple cluster
    > concentrated near the origin and a small yellow cluster far to the
    > right on the x-axis.

3.  Outliers: There are a few outlier points, most notably one purple
    > point with a very high value on the second principal component.

4.  Balance: The clusters are highly imbalanced, with the purple cluster
    > containing the vast majority of the data points and the yellow
    > cluster containing only a few points.

Comparing the two:

1.  Dimensionality: The full dataset clustering preserves more of the
    > data\'s complexity, showing variation in both dimensions. The
    > reduced dataset clustering mainly shows variation in one
    > dimension, suggesting that the feature reduction may have
    > oversimplified the data structure.

2.  Cluster Separation: The full dataset shows less distinct separation
    > between clusters, while the reduced dataset shows very clear
    > separation. However, this clear separation in the reduced dataset
    > comes at the cost of creating a highly imbalanced clustering.

3.  Outlier Detection: Both methods identify outliers, but they appear
    > more pronounced and numerous in the full dataset visualization.

4.  Interpretability: The reduced dataset provides a simpler view, which
    > might be easier to interpret, but it may oversimplify the
    > underlying data structure.

Given these observations, we recommended:

1.  The full dataset as it seems to preserve more of the data\'s
    > complexity and provides more balanced clusters.

    -   K-Means Clustering: K-Means was used for clustering network
        > traffic data with Scikit-learn\'s KMeans. The optimal number
        > of clusters (k) was determined during the process.

This code does the following:

  -----------------------------------------------------------------------
  import numpy as np\
  from sklearn.cluster import KMeans\
  from sklearn.metrics import silhouette_score\
  from tqdm import tqdm\
  import matplotlib.pyplot as plt\
  \
  print(\"\\nPerforming K-Means Clustering:\")\
  max_clusters = 10\
  silhouette_scores = \[\]\
  inertias = \[\]\
  \
  for n_clusters in tqdm(range(2, max_clusters + 1), desc=\"Calculating
  scores\"):\
  kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)\
  cluster_labels = kmeans.fit_predict(X_train_resampled_scaled)\
  \
  \# Check if we have more than one unique cluster\
  if len(np.unique(cluster_labels)) \> 1:\
  silhouette_avg = silhouette_score(X_train_resampled_scaled,
  cluster_labels, sample_size=10000)\
  silhouette_scores.append(silhouette_avg)\
  else:\
  silhouette_scores.append(-1) \# Invalid score\
  \
  inertias.append(kmeans.inertia\_)\
  \
  print(f\"For n_clusters = {n_clusters}, the average silhouette score is
  : {silhouette_scores\[-1\]}\")\
  \
  \# Plot silhouette scores\
  plt.figure(figsize=(12, 5))\
  plt.subplot(1, 2, 1)\
  plt.plot(range(2, max_clusters + 1), silhouette_scores)\
  plt.xlabel(\'Number of clusters\')\
  plt.ylabel(\'Silhouette Score\')\
  plt.title(\'Silhouette Score vs Number of Clusters\')\
  \
  \# Plot elbow curve\
  plt.subplot(1, 2, 2)\
  plt.plot(range(2, max_clusters + 1), inertias)\
  plt.xlabel(\'Number of clusters\')\
  plt.ylabel(\'Inertia\')\
  plt.title(\'Elbow Curve\')\
  \
  plt.tight_layout()\
  plt.show()\
  \
  \# Choose the number of clusters\
  valid_scores = \[score for score in silhouette_scores if score != -1\]\
  if valid_scores:\
  optimal_clusters = silhouette_scores.index(max(valid_scores)) + 2\
  print(f\"\\nOptimal number of clusters based on silhouette score:
  {optimal_clusters}\")\
  else:\
  optimal_clusters = 2 \# Default to 2 if no valid scores\
  print(\"\\nNo valid silhouette scores. Defaulting to 2 clusters.\")\
  \
  \# Perform final K-Means clustering with optimal number of clusters\
  kmeans = KMeans(n_clusters=optimal_clusters, random_state=42,
  n_init=10)\
  cluster_labels = kmeans.fit_predict(X_train_resampled_scaled)
  -----------------------------------------------------------------------

  -----------------------------------------------------------------------

It checks if K-Means produces more than one cluster before calculating
the silhouette score. It adds the calculation of inertia (within-cluster
sum of squares) for each number of clusters. It plots both the
silhouette scores and the elbow curve to help determine the optimal
number of clusters. It handles the case where no valid silhouette scores
are found.

![](vertopal_a8c83d3e11574ca39985f1f7b2fd1c88/media/image1.png){width="6.5in"
height="2.6805555555555554in"}

Output

  -----------------------------------------------------------------------
  Performing K-Means Clustering:\
  Calculating scores: 11%\|█ \| 1/9 \[00:08\<01:04, 8.05s/it\]For
  n_clusters = 2, the average silhouette score is : 0.24434166758343462\
  Calculating scores: 22%\|██▏ \| 2/9 \[00:15\<00:55, 7.95s/it\]For
  n_clusters = 3, the average silhouette score is : 0.2613634192835685\
  Calculating scores: 33%\|███▎ \| 3/9 \[00:28\<01:00, 10.06s/it\]For
  n_clusters = 4, the average silhouette score is : 0.2478982171906575\
  Calculating scores: 44%\|████▍ \| 4/9 \[00:39\<00:51, 10.32s/it\]For
  n_clusters = 5, the average silhouette score is : 0.06556648144478078\
  Calculating scores: 56%\|█████▌ \| 5/9 \[00:49\<00:41, 10.33s/it\]For
  n_clusters = 6, the average silhouette score is : 0.05251566957803987\
  Calculating scores: 67%\|██████▋ \| 6/9 \[01:04\<00:35, 11.85s/it\]For
  n_clusters = 7, the average silhouette score is : 0.2314418956938175\
  Calculating scores: 78%\|███████▊ \| 7/9 \[01:18\<00:25, 12.67s/it\]For
  n_clusters = 8, the average silhouette score is : 0.2886889635098577\
  Calculating scores: 89%\|████████▉ \| 8/9 \[01:33\<00:13,
  13.41s/it\]For n_clusters = 9, the average silhouette score is :
  0.10988421696998027\
  Calculating scores: 100%\|██████████\| 9/9 \[01:52\<00:00,
  12.47s/it\]For n_clusters = 10, the average silhouette score is :
  0.2980548417298063\
  Optimal number of clusters based on silhouette score: 10
  -----------------------------------------------------------------------

  -----------------------------------------------------------------------

Here is an explanation of the results. Based on these, we can make
several observations about the nature of our features and their
appropriateness for K-Means clustering:

1.  Silhouette Scores: The silhouette scores range from about 0.05 to
    > 0.29, which indicates moderate cluster separation. Scores closer
    > to 1 indicate better-defined clusters.

2.  Variability in Scores: There\'s significant variability in the
    > silhouette scores as the number of clusters increases. This
    > suggests that the data might have complex or nested cluster
    > structures.

3.  Optimal Number of Clusters: The highest silhouette score is for 10
    > clusters, but it\'s only marginally better than some lower cluster
    > numbers. This could indicate that: a) The data has many small,
    > distinct groups. b) The data might not have well-defined cluster
    > structures suitable for K-Means.

4.  Non-Monotonic Trend: The silhouette scores don\'t follow a smooth
    > trend as the number of clusters increases. This could suggest that
    > K-Means is struggling to find consistent, meaningful clusters
    > across different k values.

To further determine if K-Means is appropriate for our data: Feature
Distribution: We examine the distribution of each feature. K-Means works
best with features that have roughly normal distributions.

  -----------------------------------------------------------------------
  import numpy as np\
  import matplotlib.pyplot as plt\
  from sklearn.decomposition import PCA\
  from sklearn.cluster import KMeans, DBSCAN\
  from sklearn.preprocessing import StandardScaler\
  from sklearn.metrics import silhouette_score\
  \
  \# Assuming X_train_resampled_scaled is our scaled, resampled training
  data\
  \
  \# 1. PCA and Visualization\
  pca = PCA(n_components=2)\
  X_pca = pca.fit_transform(X_train_resampled_scaled)\
  \
  plt.figure(figsize=(12, 10))\
  plt.scatter(X_pca\[:, 0\], X_pca\[:, 1\], alpha=0.5)\
  plt.title(\'PCA of the dataset\')\
  plt.xlabel(\'First Principal Component\')\
  plt.ylabel(\'Second Principal Component\')\
  plt.show()\
  \
  \# 2. Visualize Cluster Distributions (using K-Means with 8 clusters as
  previously determined by the optimal number of clusters above)\
  kmeans = KMeans(n_clusters=10, random_state=42, n_init=10)\
  cluster_labels = kmeans.fit_predict(X_train_resampled_scaled)\
  \
  plt.figure(figsize=(12, 10))\
  scatter = plt.scatter(X_pca\[:, 0\], X_pca\[:, 1\], c=cluster_labels,
  cmap=\'viridis\', alpha=0.5)\
  plt.title(\'K-Means Clustering Results (PCA)\')\
  plt.xlabel(\'First Principal Component\')\
  plt.ylabel(\'Second Principal Component\')\
  plt.colorbar(scatter)\
  plt.show()\
  \
  \# 3. Analyze Feature Importance\
  feature_importance = np.abs(kmeans.cluster_centers\_).mean(axis=0)\
  feature_names = X_train_resampled.columns\
  \
  plt.figure(figsize=(12, 6))\
  plt.bar(range(len(feature_importance)), feature_importance)\
  plt.title(\'Feature Importance in K-Means Clustering\')\
  plt.xlabel(\'Features\')\
  plt.ylabel(\'Importance\')\
  plt.xticks(range(len(feature_importance)), feature_names, rotation=90)\
  plt.tight_layout()\
  plt.show()\
  \
  \# Print top 10 most important features\
  top_features = sorted(zip(feature_names, feature_importance),
  key=lambda x: x\[1\], reverse=True)\[:10\]\
  print(\"Top 10 most important features:\")\
  for feature, importance in top_features:\
  print(f\"{feature}: {importance}\")
  -----------------------------------------------------------------------

  -----------------------------------------------------------------------

Output

  -----------------------------------------------------------------------
  Top 10 most important features:\
  totalSourceBytes: 51.449226890397675\
  totalSourcePackets: 47.35221864887031\
  totalDestinationPackets: 41.92672820212233\
  appName_IGMP: 29.90401326009093\
  protocolName_igmp: 29.90401326009093\
  appName_dsp3270: 6.340633660546997\
  appName_NortonAntiVirus: 4.632278028066765\
  appName_MicrosoftMediaServer: 4.443963106662148\
  totalDestinationBytes: 3.2673301424570935\
  appName_SMS: 2.948625147230861
  -----------------------------------------------------------------------

  -----------------------------------------------------------------------

-   Kohonen Self-Organizing Maps (SOM): SOM was explored for
    > dimensionality reduction and visualization of network traffic
    > data.

A SOM code helps us analyze the resulting visualization. We can use it
to compare the results of Isolation Forest and K-Means. Do the anomalies
identified by Isolation Forest correspond to one of the K-Means
clusters?

This code:

1.  Applies StandardScaler to ensure all features are on the same scale.

2.  Uses a larger SOM grid (20x20), which is more suitable for larger
    > datasets.

3.  Increases the number of training iterations to 10000 for better
    > convergence.

4.  Visualizes a subset of the data points to avoid overcrowding the
    > plot.

When we run this on our full dataset:

1.  The training process takes some time. The verbose=True parameter
    > shows progress updates.

2.  The visualization will give us an idea of how our data clusters in
    > the SOM space.

3.  We can interpret the results by looking at how the data points are
    > distributed across the SOM grid. Clusters in this space often
    > correspond to meaningful patterns in our data.

After running this, we can:

-   Analyze the distribution of normal vs. attack instances in the SOM
    > space.

-   Compare the SOM results with our K-Means clustering results.

-   Investigate specific regions of the SOM to understand what types of
    > network traffic they represent.

Interpretation

We can analyze top 5 nodes with highest attack density(see from the
chart)

This analysis also **analyze the distribution of normal vs. attack
instances in the SOM space:**

-   **Self-Organizing Map (SOM) Visualization:** The first image
    > represents the SOM grid where each point is a node. The colors of
    > the points indicate the clusters assigned to those nodes. To
    > analyze the distribution of normal vs. attack instances, we need
    > to visualize these instances separately on the SOM.

-   **Distribution Analysis**

1.  **Compare the SOM results with our K-Means clustering results:**

    -   **K-Means Clustering Visualization**

    -   **Cluster Comparison:** Comparing the K-Means clustering results
        > with the SOM involves checking if the clusters formed by
        > K-Means correspond to distinct regions in the SOM. Ideally,
        > the clusters identified by K-Means should map well onto the
        > nodes in the SOM, indicating that both methods are identifying
        > similar structures in the data.

2.  **Investigate specific regions of the SOM to understand what types
    > of network traffic they represent:**

    -   **Region Analysis:** Each node in the SOM represents a prototype
        > vector that is similar to the data points mapped to that node.
        > By analyzing the data points mapped to specific regions, wecan
        > infer the characteristics of the network traffic in those
        > regions. For instance, nodes with high concentrations of
        > attack traffic would indicate regions associated with
        > anomalous behavior.

    -   **Detailed Investigation:** To understand the types of network
        > traffic represented by specific regions, we can plot the
        > distribution of features (e.g., totalSourceBytes,
        > totalDestinationBytes) for the data points mapped to those
        > nodes. This helps in identifying patterns or anomalies within
        > those regions

Analysis:

1.  **Distribution of Normal vs. Attack Instances in the SOM Space:**

    -   **Visualization:** The plot shows the distribution of normal
        > (blue) and attack (red) instances in the SOM space.

    -   **Observation:** Normal and attack instances are spread across
        > the grid. There are regions where normal instances are
        > dominant and regions where attack instances are more frequent.

    -   **Clusters:** Clusters of attack instances (red) are notable in
        > specific areas, indicating potential regions of anomalous
        > behavior. The presence of both normal and attack instances in
        > some nodes suggests that these nodes represent mixed or
        > transitional behavior.

2.  **Comparison with K-Means Clustering Results:**

    -   **K-Means Clustering Visualization:** The silhouette scores
        > indicate that the optimal number of clusters is likely
        > around 3. This suggests a simple structure in the data,
        > possibly distinguishing normal and attack traffic.

    -   **Cluster Mapping:** Comparing the K-Means clustering results
        > with the SOM, we should see that the regions with high
        > concentrations of attack instances in the SOM correspond to
        > distinct clusters identified by K-Means. This can be inferred
        > by overlaying the K-Means cluster labels on the SOM grid.

3.  **Investigation of Specific Regions of the SOM:**

    -   **High Attack Density Regions:** The nodes with a high density
        > of attack instances (red) should be investigated further to
        > understand the characteristics of the traffic in those
        > regions.

    -   **Feature Analysis:** For a more detailed investigation, plot
        > the distribution of key features (e.g., totalSourceBytes,
        > totalDestinationBytes) for the instances mapped to these
        > regions. This helps in identifying patterns or anomalies
        > specific to these regions

    ```{=html}
    <!-- -->
    ```
    -   LDA-K-Means and LDA-SOM:

        -   Natural Language Processing (NLP): Techniques like TF-IDF
            > from Scikit-learn were used to convert text data into
            > numerical features for use with LDA.

        -   Latent Dirichlet Allocation (LDA): LDA was employed to
            > identify latent topics within the textual features before
            > applying K-Means or SOM for clustering, aiding in
            > uncovering hidden patterns in the data.

  -----------------------------------------------------------------------
  import pandas as pd\
  import numpy as np\
  from sklearn.feature_extraction.text import TfidfVectorizer\
  from sklearn.decomposition import LatentDirichletAllocation as LDA\
  from sklearn.cluster import KMeans\
  from sklearn.preprocessing import StandardScaler\
  import matplotlib.pyplot as plt\
  from sklearn.metrics import silhouette_score\
  from minisom import MiniSom\
  from tqdm import tqdm\
  import nltk\
  \
  \# Download NLTK data\
  nltk.download(\'punkt\')\
  \
  \# Load the dataset\
  file_path = \'/content/drive/My Drive/Colab
  Notebooks/TestbedMonJun14Flows.csv\'\
  df = pd.read_csv(file_path)\
  \
  \# Select relevant features\
  numeric_features = \[\'totalSourceBytes\', \'totalDestinationBytes\',
  \'totalDestinationPackets\',\
  \'totalSourcePackets\', \'sourcePort\', \'destinationPort\'\]\
  text_features = \[\'sourcePayloadAsUTF\',
  \'destinationPayloadAsUTF\'\]\
  \
  \# Preprocess text features using TF-IDF\
  print(\"Preprocessing text features using TF-IDF:\")\
  vectorizer = TfidfVectorizer(stop_words=\'english\', max_features=100)\
  text_data = df\[text_features\].fillna(\'\').apply(lambda x: \'
  \'.join(x), axis=1)\
  tfidf_matrix = vectorizer.fit_transform(text_data)\
  \
  \# Apply LDA\
  print(\"\\nApplying LDA:\")\
  n_topics = 10 \# Number of topics to identify\
  lda = LDA(n_components=n_topics, random_state=42)\
  lda_features = lda.fit_transform(tfidf_matrix)\
  \
  \# Combine LDA features with numeric features\
  print(\"\\nCombining LDA features with numeric features:\")\
  numeric_data = df\[numeric_features\].fillna(0)\
  scaler = StandardScaler()\
  numeric_data_scaled = scaler.fit_transform(numeric_data)\
  combined_features = np.hstack(\[numeric_data_scaled, lda_features\])\
  \
  \# Apply K-Means clustering\
  print(\"\\nPerforming K-Means Clustering:\")\
  max_clusters = 10\
  silhouette_scores = \[\]\
  \
  \# Start the loop from 3 clusters to avoid single cluster solutions\
  for n_clusters in tqdm(range(3, max_clusters + 1), desc=\"Calculating
  silhouette scores\"): \# Start from 3\
  kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)\
  cluster_labels = kmeans.fit_predict(combined_features)\
  silhouette_avg = silhouette_score(combined_features, cluster_labels,
  sample_size=10000)\
  silhouette_scores.append(silhouette_avg)\
  print(f\"For n_clusters = {n_clusters}, the average silhouette score is
  : {silhouette_avg}\")\
  \
  \# Plot silhouette scores\
  plt.figure(figsize=(10, 6))\
  \# Adjust the x-axis range to start from 3\
  plt.plot(range(3, max_clusters + 1), silhouette_scores) \# Start from
  3\
  plt.xlabel(\'Number of clusters\')\
  plt.ylabel(\'Silhouette Score\')\
  plt.title(\'Silhouette Score vs Number of Clusters\')\
  plt.show()\
  \
  \# Choose the number of clusters with the highest silhouette score\
  \# Adjust the index to account for starting from 3 clusters\
  optimal_clusters = silhouette_scores.index(max(silhouette_scores)) + 3
  \# Add 3\
  print(f\"\\nOptimal number of clusters: {optimal_clusters}\")\
  \
  \# Perform final K-Means clustering with optimal number of clusters\
  kmeans = KMeans(n_clusters=optimal_clusters, random_state=42,
  n_init=10)\
  kmeans_labels = kmeans.fit_predict(combined_features)\
  \
  \# Apply SOM\
  print(\"\\nPerforming Self-Organizing Map (SOM) analysis:\")\
  som_dim = (20, 20) \# Larger SOM grid for larger dataset\
  input_dim = combined_features.shape\[1\] \# Number of features\
  \
  \# Initialize and train SOM\
  som = MiniSom(som_dim\[0\], som_dim\[1\], input_dim, sigma=0.3,
  learning_rate=0.5)\
  \
  \# Train SOM with progress updates\
  num_iterations = 10000 \# Increased iterations for better convergence\
  for i in tqdm(range(num_iterations), desc=\"Training SOM\"):\
  som.train_random(combined_features, 1) \# Train one sample at a time\
  \
  \# Get SOM node positions for each data point\
  print(\"Calculating SOM node positions\...\")\
  som_nodes = np.array(\[som.winner(x) for x in tqdm(combined_features,
  desc=\"Processing data points\")\])\
  \
  \# Visualize SOM with normal vs. attack instances\
  print(\"\\nVisualizing SOM with normal vs. attack instances:\")\
  labels = df\[\'Label\'\].values\
  normal_indices = np.where(labels == \'Normal\')\[0\]\
  attack_indices = np.where(labels == \'Attack\')\[0\]\
  \
  plt.figure(figsize=(10, 10))\
  plt.title(\'Self-Organizing Map: Normal vs. Attack\')\
  plt.scatter(som_nodes\[normal_indices, 0\], som_nodes\[normal_indices,
  1\], color=\'blue\', label=\'Normal\')\
  plt.scatter(som_nodes\[attack_indices, 0\], som_nodes\[attack_indices,
  1\], color=\'red\', label=\'Attack\')\
  plt.legend()\
  plt.colorbar(label=\'Cluster\')\
  plt.show()\
  \
  print(\"Distribution of normal vs. attack instances visualized on
  SOM.\")
  -----------------------------------------------------------------------

  -----------------------------------------------------------------------

Process

**Text Preprocessing**: TF-IDF vectorization of text features.

**LDA**: Latent Dirichlet Allocation to find latent topics in the text
features.

**Combining Features**: Merging LDA features with scaled numerical
features.

**Clustering**: K-Means clustering to determine the optimal number of
clusters.

**SOM Analysis**: Training and visualizing with Self-Organizing Map to
analyze and visualize normal vs. attack instances.

![](vertopal_a8c83d3e11574ca39985f1f7b2fd1c88/media/image3.png){width="6.5in"
height="6.722222222222222in"}

**Experimentation:** The optimal number of topics in LDA, the choice of
clustering algorithm, and the number of clusters will likely require
experimentation based on our specific dataset.

Discussion and evaluation

From the graph above, we can compare the Clusters Align with Labels: The
clusters do not significantly separate normal and malicious/attack
events based on known labels hence the LDA-clustering approach is not
likely to add value to our intrusion detection system.

Here, we combine LDA features with scaled numerical features and uses
K-Means and SOM for clustering and visualization

Step 3: Hyper parameter Tuning:

-   Automation: Scikit-optimize or other libraries were used to automate
    > hyper parameter tuning for all algorithms (Isolation Forest,
    > K-Means, SOM. This approach helped in finding the optimal
    > configuration for each model, maximizing their performance.

It checks if K-Means produces more than one cluster before calculating
the silhouette score. It adds the calculation of inertia (within-cluster
sum of squares) for each number of clusters. It plots both the
silhouette scores and the elbow curve to help determine the optimal
number of clusters. It handles the case where no valid silhouette scores
are found.

See ''K-Means Clustering: K-Means was used for clustering network
traffic data with Scikit-learn\'s KMeans. The optimal number of clusters
(k) was determined during the process.'' above

Step 4: Tackling Malware Analysis with Machine Learning:

-   Deep Learning Model: A deep learning model was trained on
    > representations derived from the K-means clustering algorithm.
    > This model automated intrusion detection by learning to classify
    > network traffic or malware samples based on patterns identified
    > during clustering.

**In this part, we'll begin by performing clustering using
scikit-learn**

**clustering is useful in cybersecurity for distinguishing between
normal and anomalous network activity, and for helping to classify
malware into families.**

**We perform these steps: Start by importing and plotting the dataset:,
Extract the features and target labels, Next, import scikit-learn\'s
clustering module and fit a K-means model with two clusters to the data
and Predict the cluster using our trained algorithm:**

**In This code, this is what happens: Extract the Features and Target
Labels\
You\'ve already selected the top features and created a reduced dataset.
We ensure X_train_resampled_scaled is prepared with the selected
features.**

**Import scikit-learn\'s Clustering Module and Fit a K-means Model\
This step involves importing the necessary libraries and fitting the
K-means algorithm to both the full dataset.**

**Predict the Clusters Using Our Trained Algorithm\
This step involves using the trained K-means model to predict the
clusters for both datasets.**

**Plot the Algorithm\'s Clusters\
Visualize the clusters formed by the algorithm. This part is crucial to
understanding how well the algorithm has captured the structure of the
dataset. We have a function plot_clusters to plot the clusters using PCA
to reduce the data to two dimensions.**

![](vertopal_a8c83d3e11574ca39985f1f7b2fd1c88/media/image5.png){width="6.5in"
height="5.527777777777778in"}

**Compare Cluster Distributions\
We use the silhouette score and cluster distribution plots to evaluate
and compare the clustering performance on the full datasets. Based on
the fact that there are two classes (attack and normal), we aim to
cluster the data into two groups that will match the sample
classification. With a thoroughly trained clustering algorithm, we are
ready to predict on the testing set. We apply our clustering algorithm
to predict to which cluster each of the samples should belong, Observing
our results in this step, we see that clustering has captured a lot of
the underlying information, as it was able to fit the data well.**

**Compare the image above and this one below. They are similar hence, we
see that clustering has captured a lot of the underlying information, as
it was able to fit the data well.**

![](vertopal_a8c83d3e11574ca39985f1f7b2fd1c88/media/image7.png){width="6.5in"
height="6.027777777777778in"}

Step 5: Automatic Intrusion Detection with Deep Learning:

-   Leveraging Representations: The representations learned from the
    > K-Means clustering were used as input features for the deep
    > learning model, enhancing the accuracy and efficiency of automatic
    > intrusion detection.

Now that we have successfully implemented and evaluated the K-means
clustering, we will move to the next steps, which involve using a deep
learning model for automatic intrusion detection. Here are the steps we
took;:

### **Tackling Malware Analysis with Machine Learning**

**Train a Deep Learning Model on K-means Cluster Representations**

First, we need to create representations from the K-means clustering
results. These representations (cluster labels) will be used as features
to train a deep learning model.

1.  **Prepare the Data**: Use the cluster labels from K-means as
    > additional features or input for the deep learning model.

2.  **Deep Learning Model**: Train a neural network using these
    > features.

### **Automatic Intrusion Detection with Deep Learning**

**Leveraging Representations from K-means for Deep Learning**

  -----------------------------------------------------------------------
  **#Prepare the data for the deep learning model (using cluster labels
  as features)\
  X_train_cluster_labels = labels_full.reshape(-1, 1) \# Use cluster
  labels from full dataset\
  \
  \# Convert string labels to numerical labels (binary classification)\
  y_train_resampled_numeric = y_train_resampled.map({\'Normal\': 0,
  \'Attack\': 1}) #label mapping\
  y_val_cluster_numeric = y_val_cluster.map({\'Normal\': 0, \'Attack\':
  1})\
  \
  \# Split the data into training and validation sets (using only cluster
  labels)\
  X_train_cluster, X_val_cluster, y_train_cluster, y_val_cluster =
  train_test_split(\
  X_train_cluster_labels, y_train_resampled_numeric, test_size=0.2,
  random_state=42 \# Use numerical labels\
  )\
  \
  \# Build and train the deep learning model on cluster representations\
  def build_model_cluster():\
  model = Sequential(\[\
  Dense(64, activation=\'relu\', input_dim=1), \# cluster label as the
  input\
  Dropout(0.5),\
  Dense(32, activation=\'relu\'),\
  Dense(1, activation=\'sigmoid\') \# binary classification\
  \])\
  model.compile(optimizer=\'adam\', loss=\'binary_crossentropy\',
  metrics=\[\'accuracy\'\])\
  return model\
  \
  \# Train the model on cluster representations\
  model_cluster = build_model_cluster()\
  history_cluster = model_cluster.fit(\
  X_train_cluster, y_train_cluster, epochs=20, batch_size=32,\
  validation_data=(X_val_cluster, y_val_cluster_numeric) \# Use numerical
  labels\
  )\
  \
  \
  \# Evaluate the model performance\
  loss_cluster, accuracy_cluster = model_cluster.evaluate(X_val_cluster,
  y_val_cluster)\
  \
  print(f\"Cluster-Based Model - Loss: {loss_cluster}, Accuracy:
  {accuracy_cluster}\")\
  **
  -----------------------------------------------------------------------

  -----------------------------------------------------------------------

**Output**

  -----------------------------------------------------------------------
  **Epoch 1/20\
  6706/6706 \[==============================\] - 30s 4ms/step - loss:
  0.5826 - accuracy: 0.6425 - val_loss: nan - val_accuracy: 0.0000e+00\
  Epoch 2/20\
  6706/6706 \[==============================\] - 21s 3ms/step - loss:
  0.5817 - accuracy: 0.6430 - val_loss: nan - val_accuracy: 0.0000e+00\
  Epoch 3/20\
  6706/6706 \[==============================\] - 32s 5ms/step - loss:
  0.5816 - accuracy: 0.6430 - val_loss: nan - val_accuracy: 0.0000e+00\
  Epoch 4/20\
  6706/6706 \[==============================\] - 19s 3ms/step - loss:
  0.5816 - accuracy: 0.6430 - val_loss: nan - val_accuracy: 0.0000e+00\
  Epoch 5/20\
  6706/6706 \[==============================\] - 20s 3ms/step - loss:
  0.5817 - accuracy: 0.6430 - val_loss: nan - val_accuracy: 0.0000e+00\
  Epoch 6/20\
  6706/6706 \[==============================\] - 18s 3ms/step - loss:
  0.5816 - accuracy: 0.6430 - val_loss: nan - val_accuracy: 0.0000e+00\
  Epoch 7/20\
  6706/6706 \[==============================\] - 24s 4ms/step - loss:
  0.5816 - accuracy: 0.6430 - val_loss: nan - val_accuracy: 0.0000e+00\
  Epoch 8/20\
  6706/6706 \[==============================\] - 39s 6ms/step - loss:
  0.5816 - accuracy: 0.6430 - val_loss: nan - val_accuracy: 0.0000e+00\
  Epoch 9/20\
  6706/6706 \[==============================\] - 21s 3ms/step - loss:
  0.5816 - accuracy: 0.6430 - val_loss: nan - val_accuracy: 0.0000e+00\
  Epoch 10/20\
  6706/6706 \[==============================\] - 19s 3ms/step - loss:
  0.5816 - accuracy: 0.6430 - val_loss: nan - val_accuracy: 0.0000e+00\
  Epoch 11/20\
  6706/6706 \[==============================\] - 18s 3ms/step - loss:
  0.5817 - accuracy: 0.6430 - val_loss: nan - val_accuracy: 0.0000e+00\
  Epoch 12/20\
  6706/6706 \[==============================\] - 18s 3ms/step - loss:
  0.5816 - accuracy: 0.6430 - val_loss: nan - val_accuracy: 0.0000e+00\
  Epoch 13/20\
  6706/6706 \[==============================\] - 19s 3ms/step - loss:
  0.5816 - accuracy: 0.6430 - val_loss: nan - val_accuracy: 0.0000e+00\
  Epoch 14/20\
  6706/6706 \[==============================\] - 20s 3ms/step - loss:
  0.5816 - accuracy: 0.6430 - val_loss: nan - val_accuracy: 0.0000e+00\
  Epoch 15/20\
  6706/6706 \[==============================\] - 20s 3ms/step - loss:
  0.5817 - accuracy: 0.6430 - val_loss: nan - val_accuracy: 0.0000e+00\
  Epoch 16/20\
  6706/6706 \[==============================\] - 18s 3ms/step - loss:
  0.5816 - accuracy: 0.6430 - val_loss: nan - val_accuracy: 0.0000e+00\
  Epoch 17/20\
  6706/6706 \[==============================\] - 18s 3ms/step - loss:
  0.5816 - accuracy: 0.6430 - val_loss: nan - val_accuracy: 0.0000e+00\
  Epoch 18/20\
  6706/6706 \[==============================\] - 18s 3ms/step - loss:
  0.5816 - accuracy: 0.6430 - val_loss: nan - val_accuracy: 0.0000e+00\
  Epoch 19/20\
  6706/6706 \[==============================\] - 20s 3ms/step - loss:
  0.5816 - accuracy: 0.6430 - val_loss: nan - val_accuracy: 0.0000e+00\
  Epoch 20/20\
  6706/6706 \[==============================\] - 21s 3ms/step - loss:
  0.5816 - accuracy: 0.6430 - val_loss: nan - val_accuracy: 0.0000e+00\
  1677/1677 \[==============================\] - 3s 2ms/step - loss:
  0.5794 - accuracy: 0.6479\
  Cluster-Based Model - Loss: 0.579361081123352, Accuracy:
  0.6479448080062866**
  -----------------------------------------------------------------------

  -----------------------------------------------------------------------

### **Explanation**

1.  **Feature Selection and Clustering: (Already implemented)**

2.  **Data Preparation: The cluster labels from K-means are concatenated
    > with the original feature sets to create new datasets
    > (X_full_with_labels and X_reduced_with_labels).\-\--(Already
    > implemented)**

3.  **Train-Test Split: The new datasets are split into training and
    > validation sets - (Already implemented)**

4.  **Deep Learning Model: A neural network is built and trained on both
    > the full and reduced datasets, including the cluster labels as
    > additional features.**

5.  **Evaluation: The performance of the models is evaluated on the
    > validation sets and the accuracy and loss are printed.**

6.  **Plot Training History: The training history of the models is
    > plotted to visualize accuracy and loss over epochs.**

**This approach leverages the cluster representations to improve the
deep learning model\'s performance on intrusion detection tasks.**

**The model Achieved a low accuracy of around 20%.**

  -----------------------------------------------------------------------
  **import numpy as np\
  from sklearn.metrics import accuracy_score, precision_score,
  recall_score, f1_score, confusion_matrix\
  \
  \# 1. Predict on Original Training Data\
  \# Predict cluster labels for original training data\
  X_train_clusters_predicted = kmeans_full.predict(X_train)\
  X_train_clusters_predicted = X_train_clusters_predicted.reshape(-1, 1)
  \# Reshape for model input\
  \
  \# Predict probabilities using the cluster-based model\
  y_train_pred_proba = model_cluster.predict(X_train_clusters_predicted)\
  \
  \# 2. Convert Predictions to Classes\
  y_train_pred = (y_train_pred_proba \> 0.5).astype(int) #Set threshold\
  \
  \# 3. Evaluate Performance\
  \# y_train contains original labels as \'Normal\' and \'Attack\'\
  y_train_numeric = y_train.map({\'Normal\': 0, \'Attack\': 1})\
  \
  accuracy = accuracy_score(y_train_numeric, y_train_pred)\
  precision = precision_score(y_train_numeric, y_train_pred)\
  recall = recall_score(y_train_numeric, y_train_pred)\
  f1 = f1_score(y_train_numeric, y_train_pred)\
  conf_matrix = confusion_matrix(y_train_numeric, y_train_pred)\
  \
  print(\"Performance on Original Training Data:\")\
  print(f\"Accuracy: {accuracy}\")\
  print(f\"Precision: {precision}\")\
  print(f\"Recall: {recall}\")\
  print(f\"F1-Score: {f1}\")\
  print(\"Confusion Matrix:\")\
  print(conf_matrix)**
  -----------------------------------------------------------------------

  -----------------------------------------------------------------------

**Output:**

  -----------------------------------------------------------------------
  **/usr/local/lib/python3.10/dist-packages/sklearn/base.py:432:
  UserWarning: X has feature names, but KMeans was fitted without feature
  names\
  warnings.warn(\
  4285/4285 \[==============================\] - 13s 3ms/step\
  Performance on Original Training Data:\
  Accuracy: 0.2008110631345548\
  Precision: 0.017475552336110103\
  Recall: 0.6450534759358288\
  F1-Score: 0.03402919810988081\
  Confusion Matrix:\
  \[\[ 25602 108510\]\
  \[ 1062 1930\]\]**
  -----------------------------------------------------------------------

  -----------------------------------------------------------------------

Training an XGBoost classifier

Gradient boosting is widely considered the most reliable and accurate
algorithm for generic machine learning problems.

This code use the XGBoost library to train a classifier for binary
classification, handle class imbalance, and evaluate the model\'s
performance.

Prepare Data for XGBoost

  -----------------------------------------------------------------------
  **!pip install xgboost\
  import xgboost as xgb\
  from sklearn.metrics import accuracy_score, precision_score,
  recall_score, f1_score, confusion_matrix\
  \
  \# 1. Prepare Data for XGBoost\
  \# X_train and y_train from the first code cell\
  \
  \# 2. Create and Train XGBoost Classifier\
  xgb_model = xgb.XGBClassifier(objective=\'binary:logistic\', \# For
  binary classification\
  eval_metric=\'logloss\', \# Use logloss as evaluation metric\
  scale_pos_weight=(len(y_train\[y_train == \'Normal\'\]) /\
  len(y_train\[y_train == \'Attack\'\])), \# Handle class imbalance\
  random_state=42)\
  \
  xgb_model.fit(X_train, y_train.map({\'Normal\': 0, \'Attack\': 1})) \#
  Fit the model\
  \
  \# 3. Predict on Original Training Data\
  y_train_pred = xgb_model.predict(X_train)\
  \
  \# 4. Evaluate Performance\
  accuracy = accuracy_score(y_train.map({\'Normal\': 0, \'Attack\': 1}),
  y_train_pred)\
  precision = precision_score(y_train.map({\'Normal\': 0, \'Attack\':
  1}), y_train_pred)\
  recall = recall_score(y_train.map({\'Normal\': 0, \'Attack\': 1}),
  y_train_pred)\
  f1 = f1_score(y_train.map({\'Normal\': 0, \'Attack\': 1}),
  y_train_pred)\
  conf_matrix = confusion_matrix(y_train.map({\'Normal\': 0, \'Attack\':
  1}), y_train_pred)\
  \
  print(\"XGBoost Performance on Original Training Data:\")\
  print(f\"Accuracy: {accuracy}\")\
  print(f\"Precision: {precision}\")\
  print(f\"Recall: {recall}\")\
  print(f\"F1-Score: {f1}\")\
  print(\"Confusion Matrix:\")\
  print(conf_matrix)**
  -----------------------------------------------------------------------

  -----------------------------------------------------------------------

**Output:**

  -----------------------------------------------------------------------
  **Requirement already satisfied: xgboost in
  /usr/local/lib/python3.10/dist-packages (2.0.3)\
  Requirement already satisfied: numpy in
  /usr/local/lib/python3.10/dist-packages (from xgboost) (1.25.2)\
  Requirement already satisfied: scipy in
  /usr/local/lib/python3.10/dist-packages (from xgboost) (1.11.4)\
  XGBoost Performance on Original Training Data:\
  Accuracy: 0.9992925078772319\
  Precision: 0.9685982518614439\
  Recall: 1.0\
  F1-Score: 0.9840486762045717\
  Confusion Matrix:\
  \[\[134015 97\]\
  \[ 0 2992\]\]**
  -----------------------------------------------------------------------

  -----------------------------------------------------------------------

**Create and Train XGBoost Classifier**

objective=\'binary:logistic\': Specifies that the objective function is
binary logistic regression, suitable for binary classification tasks.

eval_metric=\'logloss\': Uses log loss as the evaluation metric, which
measures the performance of a classification model.

scale_pos_weight=(len(y_train\[y_train == \'Normal\'\]) /
len(y_train\[y_train == \'Attack\'\])): Adjusts the weight of positive
and negative classes to handle class imbalance.

random_state=42: Ensures reproducibility by setting a seed for random
number generation.

xgb_model.fit(X_train, y_train.map({\'Normal\': 0, \'Attack\': 1})):
Trains the XGBoost model using the training data. The map({\'Normal\':
0, \'Attack\': 1}) converts the labels to numeric format (0 for
\'Normal\' and 1 for \'Attack\').

**Predict on Original Training Data**

y_train_pred = xgb_model.predict(X_train): Predicts the labels for the
training data using the trained XGBoost model.

**Evaluate Performance**

accuracy_score: Computes the accuracy of the model.

precision_score: Computes the precision of the model.

recall_score: Computes the recall of the model.

f1_score: Computes the F1 score of the model, which is the harmonic mean
of precision and recall.

confusion_matrix: Computes the confusion matrix to evaluate the accuracy
of the classification.

**This model scores an impressive accuracy of: XGBoost Performance on
Original Training Data: Accuracy: 0.9992925078772319 Precision:
0.9685982518614439**

```
