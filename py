#import all libraries
import pandas as pd
import numpy as np
import seaborn as sns
import os
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances, silhouette_score
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs

#import original dataset and check values
def load_and_check_data(file_path):
    penguins = pd.read_csv("/Users/sim/dev/python/penguins.csv")
    penguins.info()
    penguins.value_counts().sum()
    penguins.reset_index(drop=True, inplace=True)
    penguins.describe().T
    return penguins 

def plot_gender(penguins):
    palette = sns.color_palette("pastel")
    sns.countplot(x="sex", data=penguins, palette=palette)
    plt.title("Count of Gender")
    plt.xlabel("Gender")
    plt.ylabel("Count")
    plt.show()


def plot_outliers(penguins):
    plt.figure(figsize=(9,7))
    penguins.boxplot()
    plt.title("Boxplot for Outliers")
    plt.xlabel("Variable Name")
    plt.show() #seem outliers for flipper length 

def clean_data(penguins, file_path):
    print(f"Missing values before cleaning:\n{penguins.isnull().sum()}")
    penguins.dropna(inplace=True, how="any")  # Dropping rows with missing values
    print(penguins.isnull().sum()) # Missing values in columns
    penguins = penguins[penguins["sex"].isin(["MALE", "FEMALE"])]  # Keep only male and female records
    # Outliers filtering: keep only flipper_length_mm between 0 and 4000
    outliers = penguins[(penguins["flipper_length_mm"] > 0) & (penguins["flipper_length_mm"] < 4000)]
    print(f"Outliers in flipper_length_mm (between 0 and 4000): \n{outliers}")
    #Filter penguins to keep only those with flipper_length_mm in the valid range
    penguins = penguins[(penguins["flipper_length_mm"] > 0) & (penguins["flipper_length_mm"] < 4000)]
    # Remove outlier with flipper_length_mm > 4000
    penguins.reset_index(drop=True, inplace=True)  # Resetting index after removal
    penguins.describe().T
    print(penguins["sex"].value_counts())
    return penguins

   
def plot_after_clean(penguins):
    plt.figure(figsize=(10, 5))
    palette = sns.color_palette("tab10")
    # Create the countplot
    ax = sns.countplot(x="sex", data=penguins, palette=palette)
    # Get the counts from the countplot (directly from seaborn's countplot result)
    counts = penguins['sex'].value_counts()
    # Loop through the bars and add the text labels manually
    for i, count in enumerate(counts):
        # Get the x-position for each bar, which is based on the index in `counts`
        x_pos = i
        ax.text(x_pos, count + 5,  # Position the text just above the bar
                str(count),  # Display the count as text
                ha='center', va='center',  # Align the text
                fontsize=12, color='black')  # Customize text appearance
    plt.xlabel("Gender")
    plt.ylabel("Count")
    plt.title("Count of Gender After Cleaning")
    plt.show()

def plot(penguins):
    sns.scatterplot(data=penguins,
                x="culmen_length_mm",
                y="culmen_depth_mm",
                hue="sex")
    plt.show()


def flipper_plot(penguins): 
    sns.scatterplot(data=penguins,
                x="flipper_length_mm",
                y="culmen_length_mm",
                hue="sex") #uzun yüzgeçlere sahip penguenlerin genellikle daha uzun culmenlere 
                           #sahip olması nedeniyle pozitif bir korelasyon belirgindir.
    plt.show()

def body_plot(penguins):
    sns.scatterplot(data=penguins,
                x="flipper_length_mm",
                y="body_mass_g",
                hue="sex")
    plt.title("Body Mass vs Flipper Length")
    plt.ylabel("Body Mass")
    plt.xlabel("Flipper Length")
    plt.show() #yüzgeç uzunluğu ile vücut kütlesi arasında pozitif bir korelasyon olduğunu ortaya koyar.

def histogram(penguins):
    sns.histplot(data=penguins, x="culmen_length_mm", bins=20, kde=True)
    plt.title("The histogram of culmen lengths")
    plt.xlabel("Culmen Length")
    plt.ylabel("Count")
    plt.show()

def histogram_flipper(penguins):
    sns.histplot(data=penguins, x="flipper_length_mm", bins=20, kde=True)
    plt.title("The histogram of Flipper lengths")
    plt.xlabel("Flipper Length")
    plt.ylabel("Count")
    plt.show()

def scatter_flipper(penguins):
    sns.scatterplot(data=penguins, x="flipper_length_mm", y="culmen_depth_mm")
    plt.title("Culmen length vs Flipper lengths")
    plt.xlabel("Flipper Length")
    plt.ylabel("Culmen Length")
    plt.show()

def body_hist(penguins):
    sns.histplot(data=penguins, x="body_mass_g", bins=20, hue="sex")
    plt.title("The histogram of Body Mass")
    plt.xlabel("Body Mass")
    plt.ylabel("Count") #Erkek penguenler dişilere kıyasla daha yüksek bir 
                        #vücut kütlesine sahip olma eğilimindedir.
    plt.show()

def corr(penguins):
    numeric_data = penguins.select_dtypes(include=["float64", "int64"])
    corr = numeric_data.corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".1g")
    plt.title("Correlation Matrix")
    plt.show()

# Initialize LabelEncoder categorical data because we have "Sex" column -> (categorical)
def preprocess_data(penguins):
    # Encode categorical data
    label_encoder = LabelEncoder()
    penguins['sex'] = label_encoder.fit_transform(penguins['sex'])

#Feature Scaling
def scale_data(penguins):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(penguins)
    return scaled_data

def euclidean_method(scaler): #Used euclidean distance for Kmeans
    distance_matrix = pairwise_distances(scaler, metric="euclidean")
    plt.figure(figsize=(12,8))
    plt.imshow(distance_matrix, cmap="viridis", aspect="auto")
    plt.title("Euclidean Matrix")
    plt.show()
    
#PCA prepares the data for more effective and interpretable clustering
# Modify the apply_pca function
def apply_pca(scaled_data, n_components=None):
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(scaled_data)
    explained_variance = pca.explained_variance_ratio_  # Calculate explained variance
    pca_penguins = pca_result[:, :2]  # Only take the first two components (2D for visualization)
    return pca_result,pca_penguins, explained_variance #56.8% and 28.2% capture the majority of the variance in the data, 
                      #thus we can choose n_component = 2

#K-Means 
#I will use 'wcss (within cluster sum of squared)' method why to decide cluster number 
#(To calculate WCSS, cluster numbers to change from 1 to 10)
def elbow_method(pca_penguins):
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init="k-means++", random_state=42)
        kmeans.fit(pca_penguins)
        wcss.append(kmeans.inertia_)
    
    plt.plot(range(1, 11), wcss)
    plt.title("The Elbow Method")
    plt.xlabel("Number of Clusters")
    plt.ylabel("WCSS")
    plt.show()
    return kmeans
 
#Training the K-Means model according to dataset
def train_kmeans(penguins ,pca_penguins, n_clusters=4):
    kmeans = KMeans(n_clusters=n_clusters, init="k-means++", random_state=42)
    y_pred = kmeans.fit_predict(pca_penguins)
    print(f'Inertia (Within-cluster sum of squares): {kmeans.inertia_}')
    return kmeans, y_pred

def create_cluster_label(penguins, kmeans):
    penguins["cluster"] = kmeans.labels_
    return penguins

def analyze_cluster(penguins):
    mean_cluster = penguins.groupby("cluster").mean()
    total_cluster = penguins["cluster"].value_counts()
    aggregate = penguins.groupby("cluster").agg({
        "culmen_length_mm":"sum",
        "culmen_depth_mm" : "sum",
        "flipper_length_mm" :"sum",
        "body_mass_g" : "sum"
    })

    print("Mean values for each cluster:")
    print(mean_cluster)
    print("\nCount of items in each cluster:")
    print(total_cluster)
    print("\nSum of specific columns for each cluster:")
    print(aggregate)

    return mean_cluster, total_cluster, aggregate

#Inertia measurement
#Calculate "Silhouette" score to evualte for clusters quality
def silhoutte_score(scaled_data, kmeans):
    score = silhouette_score(scaled_data, kmeans.labels_)
    print(f"Silhouette Score : {score}") #my silhouette score is 0.506. 
                                          #it should be -1 to +1. it is good but not perfect.
    return score, scale_data
 

#visualizing the all clusters
def visualize_clusters(pca_penguins, y_pred, kmeans):
    # Define colors for clusters
    colors = ["purple", "pink", "green", "blue"]
    
    # Loop through each cluster and plot the points
    unique_clusters = np.unique(y_pred)
    
    for cluster in unique_clusters:
        plt.scatter(
            pca_penguins[y_pred == cluster, 0], 
            pca_penguins[y_pred == cluster, 1], 
            label=f"Cluster {cluster + 1}",
            c=colors[cluster % len(colors)]  # Assign colors to clusters
        )
    
    # Plot the centroids
    plt.scatter(
        kmeans.cluster_centers_[:, 0], 
        kmeans.cluster_centers_[:, 1], 
        s=200, 
        c="red", 
        marker="X", 
        label="Centroids"
    )
    
    # Title and labels
    plt.title("Penguin Clusters")
    plt.xlabel("First Principal Component")
    plt.ylabel("Second Principal Component")
    plt.legend()
    plt.show()

#creating data' is your DataFrame with cluster labels
def cluster_summary(penguins, cluster_column='cluster'):
    cluster_summary_df = penguins.groupby('cluster').agg({
    'culmen_length_mm': ['mean', 'max', 'min'],
    'culmen_depth_mm': ['mean', 'max', 'min'],
    'flipper_length_mm': ['mean', 'max', 'min'],
    'body_mass_g': ['mean', 'max', 'min']}).reset_index()
    return cluster_summary_df


def generate(summary_df):
    report = []
    for idx, row in summary_df.iterrows():
        report.append(
            f"Cluster {row['cluster']}:\n"
            f"- Culmen Length: mean={row[('culmen_length_mm', 'mean')]:.2f}, "
            f"max={row[('culmen_length_mm', 'max')]:.2f}, min={row[('culmen_length_mm', 'min')]:.2f}\n"
            f"- Culmen Depth: mean={row[('culmen_depth_mm', 'mean')]:.2f}, "
            f"max={row[('culmen_depth_mm', 'max')]:.2f}, min={row[('culmen_depth_mm', 'min')]:.2f}\n"
            f"- Flipper Length: mean={row[('flipper_length_mm', 'mean')]:.2f}, "
            f"max={row[('flipper_length_mm', 'max')]:.2f}, min={row[('flipper_length_mm', 'min')]:.2f}\n"
            f"- Body Mass: mean={row[('body_mass_g', 'mean')]:.2f}, "
            f"max={row[('body_mass_g', 'max')]:.2f}, min={row[('body_mass_g', 'min')]:.2f}\n"
        )
        return "\n".join(report)


file_path = "/Users/sim/dev/python/penguins.csv"

# Step 1: Load and clean data
penguins = load_and_check_data(file_path)

# Clean the data
penguins = clean_data(penguins, file_path)
print(penguins.describe())

# Step 2: Plot gender distribution and outliers
plot_gender(penguins)
plot_outliers(penguins)
plot_after_clean(penguins)
flipper_plot(penguins)
body_plot(penguins)
histogram(penguins)
corr(penguins)
body_hist(penguins)
histogram_flipper(penguins)

# Step 3: Preprocess the data (encode 'sex')
preprocess_data(penguins)

# Step 4: Scale the data
scaled_data = scale_data(penguins)
euclidean = euclidean_method(scaled_data)

# Step 5: Apply PCA
pca_result, pca_penguins, explained_variance = apply_pca(scaled_data, n_components=2)

# Step 6: Determine the number of clusters
elbow_value = elbow_method(pca_penguins)
print(elbow_value)

# Step 7: Train K-Means
kmeans, y_pred = train_kmeans(penguins, pca_penguins, n_clusters=4)

# Step 8: Add cluster labels to the dataset
penguins = create_cluster_label(penguins, kmeans)

# Step 9: Analyze clusters
mean_cluster, total_cluster, aggregate = analyze_cluster(penguins)

# Calculate and print Silhouette score
score, scaled_data = silhoutte_score(scaled_data, kmeans)

# Step 11: Visualize clusters
visualize_clusters(pca_penguins, y_pred, kmeans)

# Example usage:
summary_df= cluster_summary(penguins)
print(type(summary_df)) 

cluster_report = generate(summary_df)
print(cluster_report)

