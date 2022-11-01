import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import FastICA
from sklearn.model_selection import KFold, learning_curve, GridSearchCV
from sklearn.metrics import silhouette_score, adjusted_mutual_info_score, accuracy_score
import matplotlib.pyplot as plt
import scipy
import time

from mpl_toolkits.mplot3d import Axes3D
import warnings

warnings.filterwarnings('ignore')
pd.set_option('max_columns', None)
kf = KFold(5)

def plot_inertia(X, data_set, target_col, n_clusters=25, split=False):
    n_clusters = np.arange(2,n_clusters,1)
    inertia = []
    s_scores = []
    for n in n_clusters:
        inertia_val = None
        if split == True:
            for train_index, test_index in kf.split(X):
                X_train, X_test = X[train_index], X[test_index]
                k_means_clustering = KMeans(n_clusters=n, random_state=0)
                k_means_clustering.fit(X_train)
                s_score = silhouette_score(X_train, k_means_clustering.predict(X_train), metric='euclidean')
                print(f'{n} Silhouette Score: {s_score}', end='\r')
                s_scores.append({
                    'cluster':n, 
                    'silhouette score':s_score
                })
                inertia_val = k_means_clustering.inertia_
        else:
            k_means_clustering = KMeans(n_clusters=n, random_state=0)
            k_means_clustering.fit(X)
            inertia_val = k_means_clustering.inertia_
        inertia.append(inertia_val)
    inertia = np.array(inertia)
    plt.plot(n_clusters,inertia)
    plt.xlabel('Clusters')
    plt.ylabel('Inertia')
    plt.title(f'Clusters vs. Inertia for k-Means using {data_set} for {target_col}')
    plt.grid()
    if split:
        return s_scores

def plot_kmeans_cluster_dist(k, X, y, data_set, target_col):
    k_means_clustering = KMeans(n_clusters=k, random_state=0)
    k_means_clustering.fit(X)
    silhouette_score_value = silhouette_score(X, k_means_clustering.labels_)
    adjusted_mutual_info_score_value = adjusted_mutual_info_score(y, k_means_clustering.labels_)
    print('Inertia          : ', k_means_clustering.inertia_)
    print('Silhouette score : ', silhouette_score_value)
    print('AMI score        : ', adjusted_mutual_info_score_value)
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12,5), sharey=True, sharex=True)
    ax1.hist(k_means_clustering.labels_, bins=np.arange(0, k + 1) - 0.5, rwidth=0.5, zorder=2)
    ax1.set_xticks(np.arange(0, k))
    ax1.set_xlabel('Cluster')
    ax1.set_ylabel('No. Samples')
    title = f'Distribution of data per cluster \nfor K-Means using {data_set} for {target_col}'
#     plt.suptitle(
#         title,
#         fontsize=14,
#         fontweight="bold",
#     )
    t = pd.DataFrame(zip(k_means_clustering.labels_.reshape(-1), y.values.reshape(-1)),columns=['Predicted', 'Actual'])
    t['ones'] = 1
    t.groupby(['Predicted', 'Actual']).count()['ones'].unstack().plot.bar(
        ax=ax2, 
        xlabel='Cluster', 
        ylabel='Samples'
    )
    ax1.grid()
    ax2.grid()
    ax2.set_title(title)
    return k_means_clustering

def run_gmm(X, covs=['spherical', 'tied', 'diag', 'full'], n_comps=range(1, 20), verbose=False):
    best_score = -np.infty
    s_scores = np.zeros((len(covs),len(n_comps)))
    for i, cov in enumerate(covs):
        for j, n in enumerate(n_comps):
            gmm = GaussianMixture(n_components=n, covariance_type=cov)
            gmm.fit(X)
            s_scores[i][j] = silhouette_score(X, gmm.predict(X))
            if s_scores[i][j] > best_score:
                best_score = s_scores[i][j]
                if verbose:
                    print("Best Silhouette Score currently for", cov, n)
                best_gmm = gmm
    return best_gmm, s_scores

def plot_bic(bic, model_name, data_set, target_col, n_comps=range(1, 20), covs=['spherical', 'tied', 'diag', 'full']):
    plt.figure()
    for i, b in enumerate(bic):
        plt.plot(n_comps, b, label=covs[i])
    plt.legend()
    plt.xticks(n_comps)
    plt.title(f"Number of Components vs. Silhouette Score\nfor {model_name} using {data_set} for {target_col}")
    plt.xlabel("Number of Components")
    plt.ylabel("Silhouette Scores")
    plt.show()

def plot_cluster_dist(best_model, X, y, model_name, data_set, target_col):
    labels = best_model.predict(X)
    silhouette_score_value = silhouette_score(X, labels)
    adjusted_mutual_info_score_value = adjusted_mutual_info_score(y, labels)
    print('Silhouette score : ', silhouette_score_value)
    print('AMI score        : ', adjusted_mutual_info_score_value)
    title = f'Distribution of data per cluster \nfor {target_col} for {model_name} using {data_set}'
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,5), sharey=True)
#     plt.suptitle(title)
    ax1.hist(labels, bins=np.arange(0, best_model.n_components+1) - 0.5, rwidth=0.5, zorder=2)
    ax1.set_xlabel('Cluster')
    ax1.set_ylabel('No. Samples')
    t = pd.DataFrame(zip(labels.reshape(-1), y.values.reshape(-1)),columns=['Predicted', 'Actual'])
    t['ones'] = 1
    t.groupby(['Predicted', 'Actual']).count()['ones'].unstack().plot.bar(xlabel='Cluster', ylabel='Samples', ax=ax2)
    ax1.grid()
    ax2.grid()
    ax2.set_title(title)

def get_cluster_breakdown(labels, y):
    labels = pd.DataFrame([y, labels]).T
    labels.columns=['y', 'label']
    for cluster in set(labels.label):
        print(f'Cluster {cluster}:')
        cur = labels[(labels['label'] == cluster)]
        trues = cur[cur['y'] == True].shape[0]
        falses = cur[cur['y'] == False].shape[0]
        print(f'\tTrue  : {np.round(trues/cur.shape[0] * 100)}%')
        print(f'\tFalse : {np.round(falses/cur.shape[0] * 100)}%')


def plot_pca_variance(pca, target_col='Diabetes'):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,5))
    ax1.plot(np.arange(1, pca.explained_variance_ratio_.size + 1), pca.explained_variance_ratio_)
    ax1.set_xticks(np.arange(1, pca.explained_variance_ratio_.size + 1, 2))
    ax1.set_xlabel('Component')
    ax1.set_ylabel('Variance')
    ax1.set_title(f'Variance vs.\nPCA Component for {target_col}')
    ax1.grid()

    ax2.plot(np.arange(1, pca.explained_variance_ratio_.size + 1), pca.explained_variance_ratio_, label='Variance')
    ax2.plot(np.arange(1, pca.explained_variance_ratio_.size + 1), np.cumsum(pca.explained_variance_ratio_), label='Cumulative Variance')
    ax2.axhline(0.95, color='red')
    ax2.set_xticks(np.arange(1, pca.explained_variance_ratio_.size + 1, 2))
    ax2.set_xlabel('Component')
    ax2.set_ylabel('Variance')
    ax2.set_title(f'Cumulative Variance vs.\nPCA Component for {target_col}')
    ax2.legend()
    ax2.grid()
    plt.tight_layout()

def plot_3d(X, y, target_col='Diabetes', dataset='PCA'):
    fig = plt.figure()
    ax = Axes3D(fig)
    for i in range(len(y)):
        if y[i] == 0:
            ax.scatter(X[i, :][0], X[i, :][1], X[i, :][2], c = 'g', marker='o', label='0')
        elif y[i] == 1:
            ax.scatter(X[i, :][0], X[i, :][1], X[i, :][2], c = 'r', marker='o', label='1')
    ax.set_xlabel(f'{dataset} 1st Component')
    ax.set_ylabel(f'{dataset} 2nd Component')
    ax.set_zlabel(f'{dataset} 3rd Component')
    plt.title(f'{target_col} Dataset Reduced to 3D ({dataset})')
    plt.show()

def plot_kurtosis_values(X, n_comps=range(1, 30), target_col='Diabetes'):
    kurtosis_values = []
    for n in n_comps:
        X_ICA = FastICA(n_components = n).fit_transform(X)
        kur = scipy.stats.kurtosis(X_ICA)
        kurtosis_values.append(np.mean(kur)/n)
    kurtosis_values = np.array(kurtosis_values)
    plt.plot(n_comps, kurtosis_values)
    plt.xlabel('Components')
    plt.ylabel('Normalized Mean Kurtosis Value')
    plt.grid()
    plt.title(f'Normalized Mean Kurtosis Value vs. Components ({target_col})')
    plt.show()
    return kurtosis_values

def plot_learning_loss_curves(ls, nn, X_train, y_train, dataset, target_col):
    _, train_scores, test_scores = learning_curve(
        nn, 
        X_train, 
        y_train, 
        train_sizes=ls, 
        cv=4
    )
    
    nn.fit(X_train, y_train)
    nn.loss_curve_
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,5))
    ax1.plot(ls*100, np.mean(train_scores, axis=1), label='Train Score')
    ax1.plot(ls*100, np.mean(test_scores, axis=1), label='CV Score')
    ax1.legend()
    ax1.set_title(f"Learning Curve (Neural Network)\nUsing {dataset} for {target_col}")
    ax1.set_xlabel("Percentage of Training Examples")  
    ax1.set_ylabel("Score")
    ax1.set_xticks(ls*100)
    ax1.grid()
    
    ax2.plot(nn.loss_curve_)
    ax2.set_title(f'Loss Curve (Neural Network)\nUsing {dataset} for {target_col}')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')
    ax2.grid()


def nn_gs(param_grid, X_train, y_train, X_test, y_test, nn):
    gs_nn = GridSearchCV(nn, param_grid=param_grid, cv=4)

    start_time = time.time()
    gs_nn.fit(X_train, y_train)
    end_time = time.time()
    time_train = end_time-start_time
    print("Best params for neural network:",gs_nn.best_params_)
    print("Time to train:",time_train)

    start_time = time.time()
    classifier_accuracy = accuracy_score(y_test, gs_nn.predict(X_test))
    end_time = time.time()
    pred_time = end_time-start_time
    print("Accuracy for best neural network:", classifier_accuracy)
    print("Time to infer:",pred_time)
    return gs_nn, time_train, pred_time, classifier_accuracy

def gmm_js(gmm_p, gmm_q, n_samples=10**5):
    X = gmm_p.sample(n_samples)[0]
    log_p_X = gmm_p.score_samples(X)
    log_q_X = gmm_q.score_samples(X)
    log_mix_X = np.logaddexp(log_p_X, log_q_X)

    Y = gmm_q.sample(n_samples)[0]
    log_p_Y = gmm_p.score_samples(Y)
    log_q_Y = gmm_q.score_samples(Y)
    log_mix_Y = np.logaddexp(log_p_Y, log_q_Y)

    return np.sqrt((log_p_X.mean() - (log_mix_X.mean() - np.log(2))
            + log_q_Y.mean() - (log_mix_Y.mean() - np.log(2))) / 2)