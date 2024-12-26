import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import skfuzzy as fuzz

# Loading the datasets
df = pd.read_csv("netflix_clustering/data/credits.csv")
df1 = pd.read_csv("netflix_clustering/data/titles.csv")

# Dropping duplicates
df = df.drop_duplicates()
df1 = df1.drop_duplicates()

# Merging datasets on 'id'
netflix = df.merge(df1, how="inner", on=["id"])
netflix = netflix.drop_duplicates(subset=['type', 'title', 'release_year'])

# Cleaning 'type' and 'genres' columns
netflix['type'] = netflix['type'].str.strip().str.lower()
netflix['genres'] = netflix["genres"].str.replace(r"[',\[\]]", '', regex=True) 
netflix['main_genres'] = netflix["genres"].str.split(',').str[0]
netflix.drop(columns=["genres", "id"], inplace=True)

# Handling missing values
netflix.main_genres = netflix.main_genres.replace('', np.nan)
netflix.drop(columns=['imdb_id'], inplace=True)

# Cleaning 'production_countries' and creating a 'country' column
netflix['production_countries'] = netflix['production_countries'].str.replace(r"[',\[\]]", '', regex=True)
netflix['country'] = netflix['production_countries'].str.split(',| ').str[0].str.strip()
netflix.drop(columns=["production_countries"], inplace=True)

# Dropping unnecessary columns
netflix.drop(columns=['person_id', 'character', 'role', 'name', 'description'], axis=1, inplace=True)

# Filling missing values with appropriate defaults
netflix.fillna({
    'imdb_votes': 0,
    'imdb_score': 0,
    'tmdb_popularity': 0,
    'tmdb_score': 0,
    'age_certification': 'Unknown',
    'title': 'Unknown',
    'main_genres': 'Unknown',
    'seasons': 0
}, inplace=True)

# Preprocessing the Netflix dataset
def preprocess_netflix_data(netflix):
    # Set title as the index
    title_column = netflix['title']
    netflix.set_index("title", inplace=True)
    
    # Perform One-Hot Encoding for categorical columns
    netflix_dum = pd.get_dummies(netflix[["type", "main_genres", "country"]], drop_first=True)
    
    # Combine the one-hot encoded columns with the rest of the dataset
    netflix_dum = pd.concat([netflix_dum, netflix], axis=1)
    
    # Drop the original categorical columns
    netflix_dum.drop(["type", "main_genres", "country"], axis=1, inplace=True)
    
    # Reassign the original title column as the index
    netflix_dum.index = title_column
    
    # Scaling the data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(netflix_dum.select_dtypes(include=['float64', 'int64']))
    scaled_df = pd.DataFrame(scaled_data, columns=netflix_dum.select_dtypes(include=['float64', 'int64']).columns, index=netflix_dum.index)
    
    # Combine the scaled columns with the original categorical columns
    final_df = pd.concat([scaled_df, netflix_dum.drop(columns=scaled_df.columns)], axis=1)
    
    return final_df

# Fuzzy C-Means Clustering and CVIs calculation
def fuzzy_cmeans_clustering(netflix_dum, min_clusters=2, max_clusters=10, m=2):
    partition_coefficients = []
    partition_entropies = []
    silhouette_scores = []
    calinski_harabasz_score_list = []
    davies_bouldin_score_list = []

    # Select only numeric columns for clustering
    netflix_dum_numeric = netflix_dum.select_dtypes(include=[np.number])

    # Iterate over the number of clusters (from min_clusters to max_clusters)
    for n_clusters in range(min_clusters, max_clusters + 1):
        # Perform Fuzzy C-Means Clustering
        cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(netflix_dum_numeric.T, c=n_clusters, m=m, maxiter=100, error=0.005, init=None)
        
        # Calculate the hard assignments (most likely cluster for each point)
        hard_assignment = np.argmax(u, axis=0)
        
        # Partition Coefficient (PC)
        pc = np.sum(u**2) / netflix_dum_numeric.shape[0]
        
        # Partition Entropy (PE)
        pe = -np.sum(u * np.log(u + 1e-10)) / netflix_dum_numeric.shape[0]  # Adding epsilon to avoid log(0)
        
        # Append the results
        partition_coefficients.append(pc)
        partition_entropies.append(pe)
        
        # Silhouette Score (only for hard assignments)
        sil_score = silhouette_score(netflix_dum_numeric, hard_assignment)
        silhouette_scores.append(sil_score)

        # Calculate the Calinski-Harabasz index using the hard assignments
        calinski_harabasz = calinski_harabasz_score(netflix_dum_numeric, hard_assignment)
        calinski_harabasz_score_list.append(calinski_harabasz)

        # Compute the Davies-Bouldin Index using the hard assignments
        db_index = davies_bouldin_score(netflix_dum_numeric, hard_assignment)
        davies_bouldin_score_list.append(db_index)
        
        # Print results for each cluster count
        print(f"Clusters: {n_clusters}, PC: {pc}, PE: {pe}, Silhouette Score: {sil_score},Calinski-Harabasz Score: {calinski_harabasz}, Davies Bouldin Score: {db_index}")
    
    # Store results in a DataFrame for easy viewing
    results_df = pd.DataFrame({
        'Clusters': range(min_clusters, max_clusters + 1),
        'PC': partition_coefficients,
        'PE': partition_entropies,
        'Silhouette Score': silhouette_scores,
        'Calinski Harabasz Score': calinski_harabasz_score_list,
        'Davies Bouldin Score' : davies_bouldin_score_list
    })
    
    return results_df

# Preprocess the Netflix dataset
netflix_dum = preprocess_netflix_data(netflix)

# Perform fuzzy clustering and calculate CVIs
results_df = fuzzy_cmeans_clustering(netflix_dum)

# Display the results
print(results_df)

# Plot the CVIs to visualize the best number of clusters
plt.figure(figsize=(10, 6))
plt.plot(results_df['Clusters'], results_df['PC'], label='Partition Coefficient (PC)', marker='o')
plt.plot(results_df['Clusters'], results_df['PE'], label='Partition Entropy (PE)', marker='o')
plt.plot(results_df['Clusters'], results_df['Silhouette Score'], label='Silhouette Score', marker='o')
plt.plot(results_df['Clusters'], results_df['Calinski Harabasz Score'], label='Calinski Harabasz Score', marker='o')
plt.plot(results_df['Clusters'], results_df['Davies Bouldin Score'], label='Davies Bouldin Score', marker='o')

plt.xlabel("Number of Clusters")
plt.ylabel("Score")
plt.title("Cluster Validation Indices (CVI) Over Different Numbers of Clusters")
plt.legend()
plt.show()