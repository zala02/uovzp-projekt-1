# CORE LIBRARIES

#import yaml
import json
import numpy as np
import os
#from tqdm import tqdm



# STORING AND LOADING DATA LIBRARIES

#from scipy.sparse import save_npz
from scipy.sparse import load_npz
#from joblib import dump
from joblib import load



# PREPROCESSING LIBRARIES

#import re 
#from nltk.tokenize import word_tokenize
#def load_stopwords(path):
#    with open(path, "r", encoding="utf-8") as file:
#        return set(line.strip() for line in file if line.strip())
#sl_stopwords = load_stopwords("../resources/stopwords_slovene.txt")
#import classla
#classla.download('sl') 
#nlp = classla.Pipeline('sl', processors='tokenize,pos,lemma')



# TF-IDF LIBRARIES
#from sklearn.feature_extraction.text import TfidfVectorizer



# COMPUTING DISTANCES LIBRARIES

from sklearn.metrics.pairwise import cosine_similarity
#from sklearn.metrics.pairwise import cosine_distances



# DIMENSION REDUCTION LIBRARIES
#from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
#from sklearn.decomposition import SparsePCA
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA



# CLUSTERING LIBRARIES
#from sklearn.cluster import DBSCAN
#from sklearn.preprocessing import StandardScaler
#from sklearn.neighbors import NearestNeighbors
#from sklearn.cluster import KMeans
#from sklearn.cluster import AgglomerativeClustering



# EVALUATING LIBRARIES 
#from sklearn.metrics import silhouette_score



# VISUALIZATION LIBRARIES
import matplotlib.pyplot as plt
#from collections import defaultdict
#import seaborn as sns
#import random
from adjustText import adjust_text
from color_palette import my_colors
#from scipy.cluster.hierarchy import dendrogram, linkage


def load_articles():
    """Function that loads the articles. Change the path if its not in resources directory."""
    return yaml.load(open("../resources/articles.yaml", "r", encoding="utf-8"), Loader=yaml.CFullLoader)




def lemmatize_sl(text, nlp_pipeline):
    """Function that returns lemmatized slovene text."""
    doc = nlp_pipeline(text)
    lemmas = [word.lemma for sentence in doc.sentences for word in sentence.words]
    return lemmas


def preprocess(articles):
    """
    Function that opens the articles and saves their paragraphs.
    Then preprocess them (lowercase, cleaning, stemming, stop words)
    It creates json file ready for tfidf
    """

    # used similar process as:
    # https://medium.com/@mifthulyn07/comparing-text-documents-using-tf-idf-and-cosine-similarity-in-python-311863c74b2c


    texts = []

    # extracting text:
    for article in tqdm(articles, desc="Extracting articles"):
        # Apparently there are articles without paragraphs.  For those we take "lead". If they don't have that we just skip them.
        if article["paragraphs"] != []:
            odstavki = ""
            for paragraph in article["paragraphs"]:
                odstavki += " " + paragraph

            texts.append(odstavki)
        elif article["lead"] != []:
            texts.append(article["lead"])
        else:
            continue

    # actual preprocessing:
    for idx, text in enumerate(tqdm(texts, desc="Preprocessing texts")):
        # 1. lowercase
        lowercased_text = text.lower()

        # 2. cleaning - remove some weird symbols
        text_with_spaces = lowercased_text.replace("-", " ")            # replace - with space 
        remove_punctuation = re.sub(r'[^\w\s]', '', text_with_spaces)
        remove_white_space = remove_punctuation.strip()

        # 3. tokenize -> no need, lemmatize_sl does that
        # tokenized_text = word_tokenize(remove_white_space)

        # 4. stemming using classla (stanza)
        stemmed_text = lemmatize_sl(remove_white_space, nlp)

        # 5. remove stop words
        stopwords_removed = [word for word in stemmed_text if word not in sl_stopwords]

        texts[idx] = stopwords_removed

    # apparently tfidf expects strings not tokens anyway so 
    cleaned_texts = [" ".join(words) for words in texts]

    # Save to file
    save_path = os.path.join("resources", "cleaned_text.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(cleaned_texts, f, ensure_ascii=False, indent=2)

    return

def load_data(path):
    """Function for opening json data."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def create_tfidf(path, matrix_save_path, feature_names_save_path):
    """
    Takes path as argument (path where data used for tfidf is)
    And another path where to store tfidf
    Creates tfidf and saves it to .npz file
    """

    # load data
    clean_text = load_data(path)

    # create tfidf
    # possible useful parameters
    # max_features - ohrani prvih 10.000 besed (skupej mamo 308.998)
    # min_df - beseda se more pojavt vsaj v 3 dokumentih (preciscen noise)
    # max_df - beseda se lahko pojavi v najvec 80% dokumentih (da se znebimo najbl pogostih besed)
    # samo ker sem ze nrdila preprocess i guess im good to go
    # max_df=0.5, min_df=5 -> iz scikit-learn primer
    vectorizer = TfidfVectorizer(decode_error='ignore')
    tfidf_matrix = vectorizer.fit_transform(clean_text)
    tfidf_feauter_names = vectorizer.get_feature_names_out()

    # save matrix to npz and feature names to json
    save_npz(matrix_save_path, tfidf_matrix)
    with open(feature_names_save_path, "w", encoding="utf-8") as f:
        json.dump(tfidf_feauter_names.tolist(), f, ensure_ascii=False, indent=2)

    return


def print_top_tfidf_words(tfidf_matrix, feature_names, top_n=5):
    """Function that prints top_n words in each article that are most influential for that article."""

    for doc_idx in range(tfidf_matrix.shape[0]):
        # Get the row (article) as a sparse vector
        row = tfidf_matrix.getrow(doc_idx)
        # Convert to (word_idx, tfidf_value) tuples
        tuples = list(zip(row.indices, row.data))
        # Sort by tfidf value descending
        sorted_items = sorted(tuples, key=lambda x: x[1], reverse=True)
        top_items = sorted_items[:top_n]
        top_words = [feature_names[i] for i, score in top_items]
        top_scores = [score for i, score in top_items]

        print(f"Article {doc_idx+1} top {top_n} words:")
        for word, score in zip(top_words, top_scores):
            print(f"  {word}: {score:.4f}")
        print()

def create_cosine_sim_matrix(matrix1, matrix2, save_path, sparse=True):
    """Function that creates cosine similarity matrix between two matrices."""

    cosine_similarity_matrix = cosine_similarity(matrix1, matrix2, dense_output=True)
    # fyi, using linear_kernel(matrix1, matrix2) should be faster for larger data but ok

    np.save(save_path, cosine_similarity_matrix)

    return

def create_cosine_dist_matrix(matrix1, matrix2, save_path, sparse=True):
    """Function that creates cosine distance matrix between two matrices."""

    cosine_distances_matrix = cosine_distances(matrix1, matrix2)

    np.save(save_path, cosine_distances_matrix)

    return


def analyse_truncatedSVD(matrix):
    """Function that analyses how many dimensions is the best to reduce it to."""

    max_components=500
    #svd = TruncatedSVD(n_components=max_components, n_iter=10, random_state=13)
    svd = TruncatedSVD(n_components=max_components, algorithm='arpack', random_state=13)
    # taking .fit not .fit_transform because we only need explained variance ration not the actual matrix
    svd.fit(matrix)       
    
    explained_variance = svd.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_components+1), cumulative_variance, marker='o')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('SVD Explained Variance - Choose Dimensionality')
    plt.grid(True)
    plt.axhline(y=0.9, color='r', linestyle='--', label='90% threshold')
    plt.legend()
    plt.tight_layout()
    plt.show()


def create_truncatedSVD(matrix, save_path):
    """Function that creates dense matrix that is reduced to the analysed components."""

    max_components = 2000
    #truncated_svd = TruncatedSVD(n_components=max_components, n_iter=10, random_state=13)
    truncated_svd = TruncatedSVD(n_components=max_components, algorithm='arpack', random_state=13)
    truncated_svd_matrix = truncated_svd.fit_transform(matrix)

    explained_variance = truncated_svd.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_components+1), cumulative_variance, marker='o')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('SVD Explained Variance - Choose Dimensionality')
    plt.grid(True)
    plt.axhline(y=0.9, color='r', linestyle='--', label='90% threshold')
    plt.legend()
    plt.tight_layout()
    plt.show()

    #np.save(save_path, truncated_svd_matrix)
    np.savez_compressed(
        save_path,
        svd_matrix=truncated_svd_matrix,
        explained_variance=explained_variance,
        cumulative_variance=cumulative_variance,
        components=truncated_svd.components_,
    )


def create_kmeans(matrix, original_features, projected_features):
    """
    Function that finds the best number of clusters based on silhouette.
    We use number of clusters in range between 2 and 50
    seeds 0, 13, 42
    """

    """
    Conclusion:
        - seed doesn't make much difference
        - n_init gives usually the same result, in some cases better
        - clusters between above 17 and below 35 give the best results
        - see more in rezultati\silhuete_kmeans.pgn
    """

    """
    Dve moznosti za tfidf:
    . sparse matrix, to je og. tfidf: 
        - ne mors prikazat, edin mejbi s kakim umam al kaj ze
        - n_init das na npr. 5
    . dense matrix
        - narejen verjetno z TruncatedSVD + normalization (=LSA)
        - pridobis cas, k means bolj stabilni
        - n_init das lahko na 1
    """

    # trying the following parameters:
    seeds = [0, 13, 42]
    min_clusters = 2
    max_clusters = 50
    n_inits = [1, 5, 10, 25, 50]

    # saving the best one:
    best_silhouette = -1
    best_kmeans = None 
    best_k = None 
    best_seed = None 
    best_n_init = None

    k=34
    #for k in range(min_clusters, max_clusters+1):
    for seed in seeds:
        for n in n_inits:

            kmeans = KMeans(
                n_clusters = k,
                n_init = n,
                random_state=seed,
            ).fit(matrix)
            cluster_ids, cluster_sizes = np.unique(kmeans.labels_, return_counts=True)

            # evaluate seed and number of clusters
            score = silhouette_score(matrix, kmeans.labels_)
            print(f"Silhouette score for k={k}, seed={seed}, n_init={n}: {score:.4f}")
            #print(f"Number of elements assigned to each cluster: {cluster_sizes}")

            if score > best_silhouette:
                best_silhouette = score
                best_kmeans = kmeans
                best_k = k
                best_seed = seed
                best_n_init = n
            
            
            # print most influential words
            #order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
            #order_centroids = cluster_centers_original.argsort()[:, ::-1]
            #for i in range(kmeans.n_clusters):
            #    top_terms = [original_features[ind] for ind in order_centroids[i, :10]]
            #    print(f"Cluster {i}: {' '.join(top_terms)}")

    print(f"\nBest model: k={best_k}, seed={best_seed}, n_init={best_n_init}, silhouette={best_silhouette:.4f}")
    return best_kmeans, best_k, best_seed, best_n_init, best_silhouette



  
def plot_avg_silhouette(file_path):
    """
    Function that takes one specific txt file as input (the result of create_kmeans() function),
    calculates average silhouette for k clusters (taking different seeds and n_inits),
    and plots that. See more in rezultati\silhuete_kmeans.pgn
    """
    silhouette_per_k = defaultdict(list)

    with open(file_path, 'r') as f:
        for line in f:
            match = re.search(r'k=(\d+),.*: ([\d.]+)', line)
            if match:
                k = int(match.group(1))
                score = float(match.group(2))
                silhouette_per_k[k].append(score)

    # Compute average silhouette for each k
    avg_silhouette = {k: sum(scores) / len(scores) for k, scores in silhouette_per_k.items()}

    # Sort by number of clusters (k)
    sorted_k = sorted(avg_silhouette.keys())
    sorted_scores = [avg_silhouette[k] for k in sorted_k]

    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(sorted_k, sorted_scores, marker='o')
    plt.xticks(sorted_k)  
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Average Silhouette Score")
    plt.title("Average Silhouette Score by Number of Clusters")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_seed_vs_n_init(file_path, target_k=34):
    """
    Function for finding the best parameters for kmeans once we have the k (number of clusters).
    See more in rezultati\silhuete_kmeans_k=34.pgn
    """

    # Dict structure: {seed: {n_init: score}}
    silhouette_data = defaultdict(dict)

    with open(file_path, 'r') as f:
        for line in f:
            match = re.search(rf'k={target_k}, seed=(\d+), n_init=(\d+): ([\d.]+)', line)
            if match:
                seed = int(match.group(1))
                n_init = int(match.group(2))
                score = float(match.group(3))
                silhouette_data[seed][n_init] = score

    plt.figure(figsize=(10, 6))
    for seed in sorted(silhouette_data):
        n_inits = sorted(silhouette_data[seed])
        scores = [silhouette_data[seed][n] for n in n_inits]
        plt.plot(n_inits, scores, marker='o', label=f"Seed = {seed}")

    plt.xlabel("n_init")
    plt.ylabel("Silhouette Score")
    plt.title(f"Silhouette Scores for k={target_k} over different n_init values")
    plt.grid(True)
    plt.legend()
    plt.xticks(sorted(set(n for s in silhouette_data.values() for n in s)))
    plt.tight_layout()
    plt.show()


def plot_final_clusters_tSNE(matrix, kmeans, projected_features, original_features):
    """Function for projecting data to 2D dimensions with t-SNE and plot the clusters."""

    # Reduce to 2D for visualization
    tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42)
    reduced = tsne.fit_transform(matrix)

    labels = kmeans.labels_
    unique_labels = np.unique(labels)
    cluster_centers_original = kmeans.cluster_centers_ @ projected_features
    order_centroids = cluster_centers_original.argsort()[:, ::-1]

    plt.figure(figsize=(14, 10))
    palette = my_colors[:kmeans.n_clusters]
    plt.scatter(reduced[:, 0], reduced[:, 1], c=[palette[i] for i in labels], s=10, alpha=0.6)

    # find the top 5 words - using 5 most significant articles for a cluster and using their top significant word
    top_words = []
    for label in unique_labels:
        cluster_indices = np.where(labels == label)[0]
        cluster_vectors = matrix[cluster_indices]

        if len(cluster_vectors) == 0:
            top_words.append(f"{label}: no data")
            continue

        # Use the first article as the central one
        central_vector = cluster_vectors[0]
        similarities = cosine_similarity([central_vector], cluster_vectors)[0]
        top_k_indices = np.argsort(similarities)[::-1][:5]
        top_k_vectors = cluster_vectors[top_k_indices]
        avg_vector = top_k_vectors.mean(axis=0)

        centroid_original = avg_vector @ projected_features
        top_indices = np.argsort(centroid_original)[::-1]

        seen = set()
        words = []
        for idx in top_indices:
            word = original_features[idx]
            if word not in seen:
                words.append(word)
                seen.add(word)
            if len(words) == 5:
                break

        top_words.append(f"{label}: " + ", ".join(words))


    # Plot top words near cluster centers (approximate via mean of reduced coords per cluster)
    texts = []
    for i in range(kmeans.n_clusters):
        points = reduced[labels == i]
        if len(points) > 0:
            x_mean, y_mean = points.mean(axis=0)
            text = plt.text(
                x_mean, y_mean, top_words[i], 
                fontsize=9, weight='bold', ha='center', 
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=palette[i], lw=2.5)
            )
            texts.append(text)

    # Adjust labels to avoid overlap
    adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))

    plt.title("t-SNE Visualization of Clusters with Top Words (k=28)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_final_clusters_PCA(matrix, kmeans, projected_features, original_features):
    """Function for projecting data to 2D dimensions with PCA and plot the clusters."""

    # Reduce to 2D using PCA
    pca = PCA(n_components=2, random_state=42)
    reduced = pca.fit_transform(matrix)

    labels = kmeans.labels_
    unique_labels = np.unique(labels)
    cluster_centers_original = kmeans.cluster_centers_ @ projected_features
    order_centroids = cluster_centers_original.argsort()[:, ::-1]
    
    # find the top 5 words - using 5 most significant articles for a cluster and using their top significant word
    top_words = []
    for label in unique_labels:
        cluster_indices = np.where(labels == label)[0]
        cluster_vectors = matrix[cluster_indices]

        if len(cluster_vectors) == 0:
            top_words.append(f"{label}: no data")
            continue

        # Use the first article as the central one
        central_vector = cluster_vectors[0]
        similarities = cosine_similarity([central_vector], cluster_vectors)[0]
        top_k_indices = np.argsort(similarities)[::-1][:5]
        top_k_vectors = cluster_vectors[top_k_indices]
        avg_vector = top_k_vectors.mean(axis=0)

        centroid_original = avg_vector @ projected_features
        top_indices = np.argsort(centroid_original)[::-1]

        seen = set()
        words = []
        for idx in top_indices:
            word = original_features[idx]
            if word not in seen:
                words.append(word)
                seen.add(word)
            if len(words) == 5:
                break

        top_words.append(f"{label}: " + ", ".join(words))

    plt.figure(figsize=(14, 10))
    palette = my_colors[:kmeans.n_clusters]
    plt.scatter(reduced[:, 0], reduced[:, 1], c=[palette[i] for i in labels], s=10, alpha=0.6)

    # Plot top words near cluster centers (approximate via mean of reduced coords per cluster)
    texts = []
    for i in range(kmeans.n_clusters):
        points = reduced[labels == i]
        if len(points) > 0:
            x_mean, y_mean = points.mean(axis=0)
            text = plt.text(
                x_mean, y_mean, top_words[i], 
                fontsize=9, weight='bold', ha='center', 
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=palette[i], lw=2.5)
            )
            texts.append(text)

    # Adjust labels to avoid overlap
    adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))

    plt.title("PCA Visualization of Clusters with Top Words (k=28)")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def visualize_best_kmeans(matrix, original_features, projected_features, model):
    """Function that visualizes kmeans"""

    plot_final_clusters_tSNE(
        matrix=matrix,
        kmeans=model,
        projected_features=projected_features,
        original_features=original_features
    )

    plot_final_clusters_PCA(
        matrix=matrix,
        kmeans=model,
        projected_features=projected_features,
        original_features=original_features
    )



    return

def create_best_kmeans(matrix, save_path):
    """Function that creates kmeans with best parameters and saves it."""

    k = 28
    n = 5
    seed = 0

    kmeans = KMeans(
        n_clusters = k,
        n_init = n,
        random_state=seed,
    ).fit(matrix)
    cluster_ids, cluster_sizes = np.unique(kmeans.labels_, return_counts=True)

    # evaluate seed and number of clusters
    score = silhouette_score(matrix, kmeans.labels_)
    print(f"Silhouette score for k={k}, seed={seed}, n_init={n}: {score:.4f}")

    # Save the trained KMeans model
    dump(kmeans, save_path)


    
def plot_dendrogram(model, **kwargs):
    """Function for plotting dendrogram based on the model."""

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)

def analyse_hca_ward(matrix, original_features, projected_features):
    """Function that creates a dendrogram for HCA ward."""

    # we know its gonna be below 50 clusters so ill do the tree of 50 clusters
    hca_ward = AgglomerativeClustering(n_clusters=50, linkage='ward', compute_full_tree=False, compute_distances=True)
    hca_ward_labels = hca_ward.fit_predict(matrix)

    # evaluate seed and number of clusters
    score = silhouette_score(matrix, hca_ward_labels)
    print(f"Silhouette score for k={50}: {score:.4f}")
    #Silhouette score for k=50: 0.0278

    # Plot dendrogram
    plt.figure(figsize=(10, 6))
    plt.title("Hierarchical Clustering Dendrogram (truncated)")
    plot_dendrogram(hca_ward, truncate_mode="level", p=50)
    plt.xlabel("Sample index or cluster size")
    plt.ylabel("Distance")
    plt.tight_layout()
    plt.show()

    return

def create_hca_ward(matrix, original_features, projected_features, save_path):
    """Function that creates HCA ward with specified clusters and saves the labels."""

    # based on hac_ward_dendrogram_cut we got 26 clusters looking as the best option
    hca_ward = AgglomerativeClustering(n_clusters=26, linkage='ward', compute_full_tree=False)
    hca_ward_labels = hca_ward.fit_predict(matrix)
  
    # evaluate seed and number of clusters
    score = silhouette_score(matrix, hca_ward_labels)
    print(f"Silhouette score for k=26: {score:.4f}")

    # save the labels 
    np.save(save_path, hca_ward_labels)

    return


def plot_final_clusters_PCA_hca(matrix, labels, projected_features, original_features):
    """Function for projecting data to 2D dimensions with PCA and plot the clusters."""

    # Reduce to 2D with PCA (optional if you already have projected_features in 2D)
    pca = PCA(n_components=2, random_state=42)
    reduced = pca.fit_transform(matrix)

    unique_labels = np.unique(labels)
    palette = my_colors[:len(unique_labels)]
    
    top_words = []
    for label in unique_labels:
        cluster_indices = np.where(labels == label)[0]
        cluster_vectors = matrix[cluster_indices]

        if len(cluster_vectors) == 0:
            top_words.append(f"{label}: no data")
            continue

        central_vector = cluster_vectors[0]
        similarities = cosine_similarity([central_vector], cluster_vectors)[0]
        top_k_indices = np.argsort(similarities)[::-1][:5]
        top_k_vectors = cluster_vectors[top_k_indices]
        avg_vector = top_k_vectors.mean(axis=0)

        centroid_original = avg_vector @ projected_features
        top_indices = np.argsort(centroid_original)[::-1]

        seen = set()
        words = []
        for idx in top_indices:
            word = original_features[idx]
            if word not in seen:
                words.append(word)
                seen.add(word)
            if len(words) == 5:
                break

        top_words.append(f"{label}: " + ", ".join(words))

    plt.figure(figsize=(14, 10))
    plt.scatter(reduced[:, 0], reduced[:, 1], c=[palette[i] for i in labels], s=10, alpha=0.6)

    # Plot top words near cluster centers
    texts = []
    for i, cluster_label in enumerate(unique_labels):
        points = reduced[labels == cluster_label]
        if len(points) > 0:
            x_mean, y_mean = points.mean(axis=0)
            text = plt.text(
                x_mean, y_mean, top_words[i], 
                fontsize=9, weight='bold', ha='center', 
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=palette[i], lw=2.5)
            )
            texts.append(text)

    adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))

    plt.title(f"PCA Visualization of HCA Clusters with Top Words (k={len(unique_labels)})")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_final_clusters_tSNE_hca(matrix, labels, projected_features, original_features):

    # Reduce to 2D for visualization
    tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42)
    reduced = tsne.fit_transform(matrix)

    unique_labels = np.unique(labels)
    palette = my_colors[:len(unique_labels)]

    # Calculate cluster centroids in the original TF-IDF space
    top_words = []
    for i in unique_labels:
        cluster_points = matrix[labels == i]
        centroid_reduced = cluster_points.mean(axis=0)  # average in reduced space

        # Project centroid back to original TF-IDF space (approximate)
        centroid_original = centroid_reduced @ projected_features  # shape: (n_features,)
        top_indices = np.argsort(centroid_original)[::-1][:5]
        words = [original_features[idx] for idx in top_indices]
        top_words.append(f"{i}: " + ", ".join(words))

    plt.figure(figsize=(14, 10))
    plt.scatter(reduced[:, 0], reduced[:, 1], c=[palette[i] for i in labels], s=10, alpha=0.6)

    # Plot top words near cluster centers
    texts = []
    for i, cluster_label in enumerate(unique_labels):
        points = reduced[labels == cluster_label]
        if len(points) > 0:
            x_mean, y_mean = points.mean(axis=0)
            text = plt.text(
                x_mean, y_mean, top_words[i], 
                fontsize=9, weight='bold', ha='center', 
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=palette[i], lw=2.5)
            )
            texts.append(text)

    adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))


    plt.title(f"t-SNE Visualization of HCA Clusters with Top Words (k={len(unique_labels)})")
    plt.grid(True)
    plt.tight_layout()
    plt.show()



def visualize_best_hca_ward(labels, matrix, projected_features, original_features):
    """Function that plots labels created by hierarchical clustering algorithm in 2D."""

    plot_final_clusters_PCA_hca(
        matrix=matrix,
        labels=labels,
        projected_features=projected_features,
        original_features=original_features
    )


    plot_final_clusters_tSNE_hca(
        matrix=matrix,
        labels=labels,
        projected_features=projected_features,
        original_features=original_features
    )

    return


def analyse_hca_notward(distance_matrix, _linkage, tfidf_sparse): 
    """Function that analyses parameters of hac complete, single, and average linkage."""

    
    hca_linkage = AgglomerativeClustering(
        n_clusters=50, 
        linkage=_linkage, 
        compute_full_tree=False, 
        compute_distances=True, 
        metric='precomputed'
    )
    hca_linkage_labels = hca_linkage.fit_predict(distance_matrix)

    # evaluate seed and number of clusters
    score = silhouette_score(tfidf_sparse, hca_linkage_labels, metric="cosine")
    print(f"Silhouette score for k={50}: {score:.4f}")
    #Silhouette score for k=50: -0.0021

    # Plot dendrogram
    plt.figure(figsize=(10, 6))
    plt.title(f"Hierarchical Clustering Dendrogram - {_linkage} linkage")
    plot_dendrogram(hca_linkage, truncate_mode="level", p=50)
    plt.xlabel("Sample index or cluster size")
    plt.ylabel("Distance")
    plt.tight_layout()
    plt.show()
    
    return


def create_hca_notward(distance_matrix, _linkage, tfidf_sparse, n):
    """Function that creates HCA linkage with specified clusters and saves the labels."""

    hca_linkage = AgglomerativeClustering(
        n_clusters=n, 
        linkage=_linkage, 
        compute_full_tree=False, 
        compute_distances=True, 
        metric='precomputed'
    )
    hca_linkage_labels = hca_linkage.fit_predict(distance_matrix)

    # evaluate seed and number of clusters
    score = silhouette_score(tfidf_sparse, hca_linkage_labels, metric="cosine")
    print(f"Silhouette score for k={n}: {score:.4f}")

    # save the labels 
    #np.save(save_path, hca_ward_labels)

    return


def visualize_best_hca_notward(distance_matrix, _linkage, tfidf_sparse, matrix, projected_features, original_features, n):
    """Function that creates hac _linkage with n clusters  and plots it in 2D."""

    hca_linkage = AgglomerativeClustering(
        n_clusters=n, 
        linkage=_linkage, 
        compute_full_tree=False, 
        compute_distances=True, 
        metric='precomputed'
    )

    hca_linkage_labels = hca_linkage.fit_predict(distance_matrix)

    # evaluate seed and number of clusters
    score = silhouette_score(tfidf_sparse, hca_linkage_labels, metric="cosine")
    print(f"Silhouette score for k=42: {score:.4f}")

    plot_final_clusters_PCA_hca(
        matrix=matrix,
        labels=hca_linkage_labels,
        projected_features=projected_features,
        original_features=original_features
    )

    plot_final_clusters_tSNE_hca(
        matrix=matrix,
        labels=hca_linkage_labels,
        projected_features=projected_features,
        original_features=original_features
    )

    """
    Some of the results:
    # n = 30
    visualize_best_hca_notward(cosine_dist_matrix, "complete", tfidf_sparse, tfidf_truncatedSVD_dense, tfidf_truncatedSVD_components, tfidf_feauter_names, 30)
    # Silhouette score for k=42: -0.0276
    visualize_best_hca_notward(cosine_dist_matrix, "single", tfidf_sparse, tfidf_truncatedSVD_dense, tfidf_truncatedSVD_components, tfidf_feauter_names, 30)
    # Silhouette score for k=42: -0.0088
    visualize_best_hca_notward(cosine_dist_matrix, "average", tfidf_sparse, tfidf_truncatedSVD_dense, tfidf_truncatedSVD_components, tfidf_feauter_names, 30)
    # Silhouette score for k=42: 0.0004

    # n = 50
    visualize_best_hca_notward(cosine_dist_matrix, "complete", tfidf_sparse, tfidf_truncatedSVD_dense, tfidf_truncatedSVD_components, tfidf_feauter_names, 50)
    # Silhouette score for k=42: -0.0234
    visualize_best_hca_notward(cosine_dist_matrix, "single", tfidf_sparse, tfidf_truncatedSVD_dense, tfidf_truncatedSVD_components, tfidf_feauter_names, 50)
    # Silhouette score for k=42: -0.0159
    visualize_best_hca_notward(cosine_dist_matrix, "average", tfidf_sparse, tfidf_truncatedSVD_dense, tfidf_truncatedSVD_components, tfidf_feauter_names, 50)
    # Silhouette score for k=42: -0.0053
    

    # n = 70
    visualize_best_hca_notward(cosine_dist_matrix, "complete", tfidf_sparse, tfidf_truncatedSVD_dense, tfidf_truncatedSVD_components, tfidf_feauter_names, 70)
    # Silhouette score for k=42: -0.0195
    visualize_best_hca_notward(cosine_dist_matrix, "single", tfidf_sparse, tfidf_truncatedSVD_dense, tfidf_truncatedSVD_components, tfidf_feauter_names, 70)
    # Silhouette score for k=42: -0.0229
    visualize_best_hca_notward(cosine_dist_matrix, "average", tfidf_sparse, tfidf_truncatedSVD_dense, tfidf_truncatedSVD_components, tfidf_feauter_names, 70)
    # Silhouette score for k=42: 0.0007
    

    # n = 100
    visualize_best_hca_notward(cosine_dist_matrix, "complete", tfidf_sparse, tfidf_truncatedSVD_dense, tfidf_truncatedSVD_components, tfidf_feauter_names, 100)
    # Silhouette score for k=42: -0.0061
    visualize_best_hca_notward(cosine_dist_matrix, "single", tfidf_sparse, tfidf_truncatedSVD_dense, tfidf_truncatedSVD_components, tfidf_feauter_names, 100)
    # Silhouette score for k=42: -0.0287
    visualize_best_hca_notward(cosine_dist_matrix, "average", tfidf_sparse, tfidf_truncatedSVD_dense, tfidf_truncatedSVD_components, tfidf_feauter_names, 100)
    # Silhouette score for k=42: 0.0029
    
    """

def plot_k_distance_graph(distance_matrix, k):
    """Function to plot k-distance graph, used for DBSCAN analysis."""

    neigh = NearestNeighbors(n_neighbors=k, metric="precomputed")
    neigh.fit(distance_matrix)
    distances, _ = neigh.kneighbors(distance_matrix)
    distances = np.sort(distances[:, k-1])

    plt.figure(figsize=(10, 6))
    plt.plot(distances)
    plt.xlabel('Points')
    plt.ylabel(f'{k}-th nearest neighbor distance')
    plt.title('K-distance Graph')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Set y-ticks every 0.05
    y_min, y_max = distances.min(), distances.max()
    y_ticks = np.arange(np.floor(y_min * 20) / 20, np.ceil(y_max * 20) / 20 + 0.05, 0.05)
    plt.yticks(y_ticks)

    plt.tight_layout()
    plt.show()


def print_dbscan_cluster_top_words(matrix, labels, projected_features, original_features, top_n=5):
    """Function for printing the top 5 words made by dbscan."""

    unique_labels = [label for label in np.unique(labels) if label != -1]  # Exclude noise
    for i in unique_labels:
        cluster_points = matrix[labels == i]
        centroid_reduced = cluster_points.mean(axis=0)
        centroid_original = centroid_reduced @ projected_features

        top_indices = np.argsort(centroid_original)[::-1][:top_n]
        top_words = [original_features[idx] for idx in top_indices]
        print(f"Cluster {i}: {', '.join(top_words)}")


def analyse_dbscan(distance_matrix, tfidf_sparse, matrix, projected_features, original_features):
    """Function for analysing DBSCAN's parameters epsilon and minimum samples."""

    """
    For choosing eps:
        - domain knowledge
            . what distance is meaningful for this problem
            . I'd say 0.072 -> average 1000th similarity 
        - K-distance graph
            . Calculate the distance to the k-th nearest neighbor for each point (where k = MinPts).
            . Plot these k-distances in ascending order.
            . Look for an "elbow" in the graph - a point where the curve starts to level off.
            . The ε value at this elbow is often a good choice.
    """

    """
    For choosing min_samples:
        1. General rule: 
        A good starting point is to set MinPts = 2 * num_features, 
        where num_features is the number of dimensions in your dataset.
        But I guess that's for 2D data.

        2. Noise consideration: If your data has noise or you want to detect 
        smaller clusters, you might want to decrease MinPts.

        3. Dataset size: For larger datasets, you might need to increase MinPts 
        to avoid creating too many small clusters.
    """
    """
    cosine_similarity_matrix = np.clip(distance_matrix, 0, 1)
    cosine_distance_matrix = 1 - cosine_similarity_matrix
    #print("Any negatives?", (cosine_distance_matrix < 0).any())  # Should be False


    k_nearest_neighbours = [5, 10, 20, 50, 100, 500, 1000]
    #k_nearest_neighbours = [100, 500, 1000]
    for k in k_nearest_neighbours:
        plot_k_distance_graph(cosine_distance_matrix, k)
    """

    """
               eps
    k=5     -> 0.88
    k=10    -> 0.88
    k=20    -> 0.93
    k=50    -> 0.93
    k=100   -> 0.95
    k=500   -> 0.98
    k=1000  -> 0.98
    """

    cosine_similarity_matrix = np.clip(distance_matrix, 0, 1)
    cosine_distance_matrix = 1 - cosine_similarity_matrix


    # based on the graph up there:
    #_eps = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    #_min_samples = [5, 10, 50, 100, 500, 1000]
    #_eps = [0.93]
    #_min_samples = [20]
    #_eps = [0.70, 0.71, 0.72, 0.73, 0.74, 0.75, 0.76, 0.77, 0.78, 0.79, 0.80]
    #_min_samples = [5, 10, 15, 20, 25, 30, 35, 40]

    _eps = [0.5, 0.53, 0.55, 0.57, 0.6, 0.63, 0.65, 0.67, 0.7, 0.73, 0.75, 0.77, 0.8]
    _min_samples = [50]

    for e in _eps:
        for sample in _min_samples:
            print("-----------------------------------------------------------------------------------------------")
            print(f"EPSILON: {e}, MIN SAMPLES: {sample}")

            dbscan = DBSCAN(eps=e, min_samples=sample, metric="precomputed")
            clusters = dbscan.fit_predict(cosine_distance_matrix)
            centers = dbscan.components_

            # analyse what we got
            n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
            n_noise = list(clusters).count(-1)
            print(f'Number of clusters: {n_clusters}')
            print(f'Number of noise points: {n_noise}')

            """
            # print top 5 words from centers
            #print(centers)
            print_dbscan_cluster_top_words(
                matrix=tfidf_truncatedSVD_dense_normalized,
                labels=clusters,
                projected_features=projected_features,
                original_features=original_features
            )
            """

    """
    RESULT
    in a form: number of clusters / noise (which you need to multiply by 1000)

    e \ s       5       10      15      20      25      30      35      40
    0.70        131/8   45/10   40/12   22/13   35/14   28/15   24/15   18/16
    0.71        112/7   47/9    35/11   16/12   20/13   26/13   23/14   18/15
    0.72        92/6    36/8    26/10   14/11   12/12   12/12   18/13   16/14
    0.73        76/6    28/7    21/9    21/10   7/11    8/11    9/12    13/13
    0.74        63/5    18/7    17/8    14/9    6/10    4/10    6/11    9/11
    0.75        49/4
    0.76        28/4

    so based on previous knowledge the groups from 15 to 35 clusters are the most suitable.:
    
    Let's try them out (maybe first the ones with less noise)
    """

def advanced_dbscan_cluster_top_words(matrix, labels, projected_features, original_features, top_n=5):
    """A better function for printing top 5 words made by dbscan."""

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    
    for i in range(n_clusters):
        cluster_indices = np.where(labels == i)[0]
        
        if len(cluster_indices) == 0:
            continue

        cluster_vectors = matrix[cluster_indices]

        # Take a core point (e.g. the first one)
        central_vector = cluster_vectors[0]

        # Calculate similarity to all others in the cluster
        similarities = cosine_similarity([central_vector], cluster_vectors)[0]
        top_k = np.argsort(similarities)[::-1][:5]  # 5 most similar articles in cluster
        top_vectors = cluster_vectors[top_k]

        # Average their vectors
        avg_vector = top_vectors.mean(axis=0)

        # Project to original TF-IDF space
        centroid_original = avg_vector @ projected_features  # shape: (n_original_features,)

        # Find top words
        top_indices = np.argsort(centroid_original)[::-1]
        seen = set()
        top_words = []
        for idx in top_indices:
            word = original_features[idx]
            if word not in seen:
                top_words.append(word)
                seen.add(word)
            if len(top_words) == top_n:
                break

        print(f"Cluster {i}: {', '.join(top_words)}")


def plot_final_clusters_PCA_dbscan(matrix, labels, projected_features, original_features):
    """Function for projecting data to 2D dimensions with PCA and plot the clusters."""

    # Reduce to 2D with PCA
    pca = PCA(n_components=2, random_state=42)
    reduced = pca.fit_transform(matrix)

    unique_labels = np.unique(labels)
    has_noise = -1 in unique_labels
    n_clusters = len(unique_labels) - (1 if has_noise else 0)

    palette = my_colors[:n_clusters]
    non_noise_labels = [label for label in unique_labels if label != -1]
    label_to_color = {label: palette[i] for i, label in enumerate(non_noise_labels)}

    top_words = []
    for label in unique_labels:
        if label == -1:
            top_words.append("Noise")
            continue

        cluster_indices = np.where(labels == label)[0]
        cluster_vectors = matrix[cluster_indices]

        if len(cluster_vectors) == 0:
            top_words.append(f"{label}: no data")
            continue

        central_vector = cluster_vectors[0]
        similarities = cosine_similarity([central_vector], cluster_vectors)[0]
        top_k_indices = np.argsort(similarities)[::-1][:5]
        top_k_vectors = cluster_vectors[top_k_indices]
        avg_vector = top_k_vectors.mean(axis=0)

        centroid_original = avg_vector @ projected_features
        top_indices = np.argsort(centroid_original)[::-1]

        seen = set()
        words = []
        top_n = 5
        for idx in top_indices:
            word = original_features[idx]
            if word not in seen:
                words.append(word)
                seen.add(word)
            if len(words) == top_n:
                break

        top_words.append(f"{label}: " + ", ".join(words))

    # Plotting
    plt.figure(figsize=(14, 10))
    for label in unique_labels:
        color = 'black' if label == -1 else label_to_color[label]
        size = 3 if label == -1 else 10
        alpha = 0.3 if label == -1 else 0.6

        cluster_points = reduced[labels == label]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], c=color, s=size, alpha=alpha, label=f'Cluster {label}' if label != -1 else "Noise")

    # Plot top words as annotations
    texts = []
    for i, label in enumerate(unique_labels):
        if label == -1:
            continue
        points = reduced[labels == label]
        if len(points) > 0:
            x_mean, y_mean = points.mean(axis=0)
            text = plt.text(
                x_mean, y_mean, top_words[i],
                fontsize=9, weight='bold', ha='center',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=label_to_color[label], lw=2.5)
            )
            texts.append(text)

    adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))

    plt.title(f"PCA Visualization of DBSCAN Clusters with Top Words (k={n_clusters})")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_final_clusters_tSNE_dbscan(matrix, labels, projected_features, original_features):
    """Function for projecting data to 2D dimensions with t-SNE and plot the clusters."""

    # Reduce to 2D with PCA
    tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42)
    reduced = tsne.fit_transform(matrix)

    unique_labels = np.unique(labels)
    has_noise = -1 in unique_labels
    n_clusters = len(unique_labels) - (1 if has_noise else 0)
    #print(f"number of clusters wihout noise: {n_clusters}")

    palette = my_colors[:n_clusters]
    #print(f"palette size: {len(palette)}")
    non_noise_labels = [label for label in unique_labels if label != -1]
    label_to_color = {label: palette[i] for i, label in enumerate(non_noise_labels)}

    top_words = []
    for label in unique_labels:
        if label == -1:
            top_words.append("Noise")
            continue

        cluster_indices = np.where(labels == label)[0]
        cluster_vectors = matrix[cluster_indices]

        if len(cluster_vectors) == 0:
            top_words.append(f"{label}: no data")
            continue

        central_vector = cluster_vectors[0]
        similarities = cosine_similarity([central_vector], cluster_vectors)[0]
        top_k_indices = np.argsort(similarities)[::-1][:5]
        top_k_vectors = cluster_vectors[top_k_indices]
        avg_vector = top_k_vectors.mean(axis=0)

        centroid_original = avg_vector @ projected_features
        top_indices = np.argsort(centroid_original)[::-1]

        seen = set()
        words = []
        top_n = 5
        for idx in top_indices:
            word = original_features[idx]
            if word not in seen:
                words.append(word)
                seen.add(word)
            if len(words) == top_n:
                break

        top_words.append(f"{label}: " + ", ".join(words))

    # Plotting
    plt.figure(figsize=(14, 10))
    for label in unique_labels:
        color = 'black' if label == -1 else label_to_color[label]
        size = 3 if label == -1 else 10
        alpha = 0.3 if label == -1 else 0.6

        cluster_points = reduced[labels == label]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], c=color, s=size, alpha=alpha, label=f'Cluster {label}' if label != -1 else "Noise")

    # Plot top words as annotations
    texts = []
    for i, label in enumerate(unique_labels):
        if label == -1:
            continue
        points = reduced[labels == label]
        if len(points) > 0:
            x_mean, y_mean = points.mean(axis=0)
            text = plt.text(
                x_mean, y_mean, top_words[i],
                fontsize=9, weight='bold', ha='center',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=label_to_color[label], lw=2.5)
            )
            texts.append(text)

    adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))

    plt.title(f"PCA Visualization of DBSCAN Clusters with Top Words (k={n_clusters})")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.grid(True)
    plt.tight_layout()
    plt.show()




def create_dbscan(distance_matrix, tfidf_sparse, matrix, projected_features, original_features):
    """Creates the best dbscan and plots it."""

    e = 0.71
    sample = 30

    dbscan = DBSCAN(eps=e, min_samples=sample, metric="precomputed")
    clusters = dbscan.fit_predict(distance_matrix)
    centers = dbscan.components_

    # print top 5 words from centers
    """
    advanced_dbscan_cluster_top_words(
        matrix=matrix,
        labels=clusters,
        projected_features=projected_features,
        original_features=original_features
    )
    """
    plot_final_clusters_PCA_dbscan(
        matrix=matrix,
        labels=clusters,
        projected_features=projected_features,
        original_features=original_features
    )

    
    plot_final_clusters_tSNE_dbscan(
        matrix=matrix,
        labels=clusters,
        projected_features=projected_features,
        original_features=original_features
    )



if __name__ == "__main__":

    # UNCOMMENT THE FOLLOWING PART FOR WHOLE PROCESS
    
    """
    # 1. PREPROCESS

    # load the articles
    articles = load_articles()

    # preprocess the articles
    preprocess(articles)
    

    
    # 2. TFIDF

    # save all the paths (we'll need them later)
    clean_text_path = os.path.join("resources", "cleaned_text.json")
    tfidf_sparse_path = os.path.join("resources", "tfidf_sparse.npz")
    tfidf_feauter_names_path = os.path.join("resources", "tfidf_feauter_names.json")

    # create the actual matrix
    create_tfidf(clean_text_path, tfidf_sparse_path, tfidf_feauter_names_path)

    # once we have the matrix we can simply load it by:
    tfidf_sparse = load_npz(tfidf_sparse_path)
    tfidf_feauter_names = load_data(tfidf_feauter_names_path)

    # to print top words of each article use the following function:
    #print_top_tfidf_words(tfidf_sparse, tfidf_feauter_names)




    # 3. KOSINUSNA RAZDALJA MED TFIDF 

    # 3.1. cosine similarity matrix 
    cosine_sim_matrix_path = os.path.join("resources", "cosine_sim_matrix.npy")
    create_cosine_sim_matrix(tfidf_sparse, tfidf_sparse, cosine_sim_matrix_path)

    cosine_similarity_matrix = np.load(cosine_sim_matrix_path)

    # 3.2. cosine distance matrix

    cosine_dist_matrix_path = os.path.join("resources", "cosine_dist_matrix.npy")
    create_cosine_dist_matrix(tfidf_sparse, tfidf_sparse, cosine_dist_matrix_path)

    cosine_dist_matrix = np.load(cosine_dist_matrix_path)




    # 4. DIMENSION REDUCTION TO 2000 FEATURES

    # TruncatedSVD 
    # using truncated svd not svd since were working on sparse matrix 
    # also using arpack since iguess its more accurate 
    # got 0.55 of covered variance for 2000 components, waited around 20min
    tfidf_truncatedSVD_2000_path = os.path.join("resources", "tfidf_truncatedSVD_2000.npz")
    tfidf_truncatedSVD = create_truncatedSVD(tfidf_sparse, tfidf_truncatedSVD_2000_path)

    with np.load(tfidf_truncatedSVD_2000_path) as data:
        tfidf_truncatedSVD_dense = data['svd_matrix']
        tfidf_truncatedSVD_variance = data['explained_variance']
        tfidf_truncatedSVD_cumulative = data['cumulative_variance']
        tfidf_truncatedSVD_components = data['components']

    # to analyse how many features is the most optimal we can use the following function:
    #analyse_truncatedSVD(tfidf_sparse)

    # svd is not normalised by itself so we need to do the following:
    normalizer = Normalizer(copy=False)
    tfidf_truncatedSVD_dense_normalized = normalizer.transform(tfidf_truncatedSVD_dense)


    
    # 5. CLUSTERING


    # 5.1. K - means

    # for kmeans analysis check and use the following functions:
    # create_kmeans(), plot_avg_silhouette(), plot_seed_vs_n_init()

    # to save the best model:
    kmeans_model_path = os.path.join("resources", "best_kmeans_model.joblib")
    create_best_kmeans(tfidf_truncatedSVD_dense_normalized, kmeans_model_path)

    # to visualize the best model:
    kmeans_model = load(kmeans_model_path)
    visualize_best_kmeans(tfidf_truncatedSVD_dense_normalized, tfidf_feauter_names, tfidf_truncatedSVD_components, kmeans_model)




    # 5.2. Hiearchical clustering - ward

    # to analyse hac - ward check and use the following function:
    #analyse_hca_ward(tfidf_truncatedSVD_dense_normalized, tfidf_feauter_names, tfidf_truncatedSVD_components)

    hca_ward_path = os.path.join("resources", "hca_ward_labels.npy")
    create_hca_ward(tfidf_truncatedSVD_dense_normalized, tfidf_feauter_names, tfidf_truncatedSVD_components, hca_ward_path)

    # load hca ward labels
    hca_ward_path = os.path.join("resources", "hca_ward_labels.npy")
    hca_ward_labels = np.load(hca_ward_path)

    visualize_best_hca_ward(hca_ward_labels, tfidf_truncatedSVD_dense_normalized, tfidf_truncatedSVD_components, tfidf_feauter_names)



    # 5.3. Hiearchical clustering - complete, average, single - cosinus matrix 

    # to analyse hac - complete, average and single linkages check and use the following function:
    #analyse_hca_notward()

    # for example:
    analyse_hca_notward(cosine_dist_matrix, "complete", tfidf_sparse)
    analyse_hca_notward(cosine_dist_matrix, "single", tfidf_sparse)
    analyse_hca_notward(cosine_dist_matrix, "average", tfidf_sparse)

    # since all the silhouette scores for hac linkages are horrible, I highly recommend visualising directly
    # for example:
    n = 100
    visualize_best_hca_notward(cosine_dist_matrix, "complete", tfidf_sparse, tfidf_truncatedSVD_dense, tfidf_truncatedSVD_components, tfidf_feauter_names, n)
    visualize_best_hca_notward(cosine_dist_matrix, "single", tfidf_sparse, tfidf_truncatedSVD_dense, tfidf_truncatedSVD_components, tfidf_feauter_names, n)
    visualize_best_hca_notward(cosine_dist_matrix, "average", tfidf_sparse, tfidf_truncatedSVD_dense, tfidf_truncatedSVD_components, tfidf_feauter_names, n)

    # However, even tho we increased number of clusters to 100, results are still horrible



    # 5.4 DBSCAN
    
    # use the following function to analyse the parameters:
    #analyse_dbscan(cosine_similarity_matrix, tfidf_sparse, tfidf_truncatedSVD_dense, tfidf_truncatedSVD_components, tfidf_feauter_names)
    # Higly recommending s as high as possible or otherwise we get one big cluster and lots of small ones

    # use the following function to visualize DBSCAN:
    create_dbscan(cosine_dist_matrix, tfidf_sparse, tfidf_truncatedSVD_dense, tfidf_truncatedSVD_components, tfidf_feauter_names)

    """

    # import tfidf
    tfidf_sparse_path = os.path.join("resources", "tfidf_sparse.npz")
    tfidf_feauter_names_path = os.path.join("resources", "tfidf_feauter_names.json")

    tfidf_sparse = load_npz(tfidf_sparse_path)
    tfidf_feauter_names = load_data(tfidf_feauter_names_path)


    # import reduced tfidf (by truncated svd)
    tfidf_truncatedSVD_2000_path = os.path.join("resources", "tfidf_truncatedSVD_2000.npz")
    with np.load(tfidf_truncatedSVD_2000_path) as data:
        tfidf_truncatedSVD_dense = data['svd_matrix']
        tfidf_truncatedSVD_variance = data['explained_variance']
        tfidf_truncatedSVD_cumulative = data['cumulative_variance']
        tfidf_truncatedSVD_components = data['components']

    # normalize reduced tfidf:
    normalizer = Normalizer(copy=False)
    tfidf_truncatedSVD_dense_normalized = normalizer.transform(tfidf_truncatedSVD_dense)
    

    # load the best kmeans model and visualize the results:
    kmeans_model_path = os.path.join("resources", "best_kmeans_model.joblib")
    kmeans_model = load(kmeans_model_path)
    visualize_best_kmeans(tfidf_truncatedSVD_dense_normalized, tfidf_feauter_names, tfidf_truncatedSVD_components, kmeans_model)
