import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.preprocessing import scale
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN

iris = load_iris()
irisDF = pd.DataFrame(iris.data, columns = iris.feature_names)
irisDF.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
# 일부만 열 이름 변경 하고싶을 때는 이게 편함irisDF.rename(columns = {'sepan length (cm)': 'sepal length'})
irisDF.head()

kmeans = KMeans(n_clusters = 3, init = 'k-means++', max_iter = 300, random_state = 0)
kmeans.fit(irisDF)
print(kmeans.labels_)

irisDF['target'] = iris.target
irisDF['cluster'] = kmeans.labels_
iris_result = irisDF.groupby(['target', 'cluster'])['sepal_length'].count()
iris_result

pca = PCA(n_components = 2)
pca_transformed = pca.fit_transform(iris.data)

irisDF['pca1'] = pca_transformed[:, 0]
irisDF['pca2'] = pca_transformed[:, 1]
irisDF.head()

marker0_ind = irisDF[irisDF['cluster'] == 0].index
marker1_ind = irisDF[irisDF['cluster'] == 1].index
marker2_ind = irisDF[irisDF['cluster'] == 2].index

plt.scatter(irisDF.loc[marker0_ind, 'pca1'], irisDF.loc[marker0_ind, 'pca2'], marker = 'o')
plt.scatter(irisDF.loc[marker1_ind, 'pca1'], irisDF.loc[marker1_ind, 'pca2'], marker = 's')
plt.scatter(irisDF.loc[marker2_ind, 'pca1'], irisDF.loc[marker2_ind, 'pca2'], marker = '^')

plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.title('3 Clusters Visualization by 2 PCA Components')
plt.show()

score_samples = silhouette_samples(iris.data, irisDF['cluster'])
print('silhouette_samples() return 값의 shape: ', score_samples.shape)

irisDF['silhouette_coeff'] = score_samples

avg_score = silhouette_score(iris.data, irisDF['cluster'])
print('붓꽃 데이터세트 Silhouette Analysis Score: {0:.3f}'.format(avg_score))
irisDF.head()

irisDF.groupby(['cluster'])['silhouette_coeff'].mean()

##### GMM #####
irisDF2 = pd.DataFrame(data = iris.data, columns = ['sepan_length', 'sepal_width', 'petal_length', 'petal_width'])
irisDF2['target'] = iris.target

gmm = GaussianMixture(n_components = 3, random_state = 0).fit(iris.data)
gmm_cluster_labels = gmm.predict(iris.data)

irisDF2['gmm_cluster'] = gmm_cluster_labels

iris_result1 = irisDF2.groupby('target')['gmm_cluster'].value_counts()
iris_result1

##### KMeans #####
kmeans = KMeans(n_clusters = 3, init = 'k-means++', max_iter = 300, random_state = 0)
kmeans.fit(iris.data) # irisDF2에 target도 들어가 있으므로 iris.data만 함
kmeans_cluster_labels = kmeans.predict(iris.data)
irisDF2['kmeans_cluster'] = kmeans_cluster_labels
iris_result2 = irisDF.groupby('target')['kmeans_cluster'].value_counts()
iris_result2

##### DBSCAN #####
dbscan = DBSCAN(eps = 0.6, min_samples = 8, metric = 'euclidean')
dbscan_labels = dbscan.fit_predict(iris.data)
irisDF2['dbscan_cluster'] = dbscan_labels
iris_result3 = irisDF.groupby('target')['dbscan_cluster'].value_counts()
iris_result3

# 클러스터 결과를 담은 dataframe과 사이킷런의 cluster 객체 등을 인자로 받아 클러스터링 결과를 시각화한 함수
def visualize_cluster_plot(clusterobj, dataframe, label_name, iscenter = True):
    if iscenter:
        centers = clusterobj.cluster_centers_
    
    unique_labels = np.unique(dataframe[label_name].values)
    markers = ['o', 's', '^', 'x', '*']
    isNoise=False

    for label in unique_labels:
        label_cluster = dataframe[dataframe[label_name] == label]
        if label == -1:
            cluster_legend = 'Noise'
            isNoise = True
        else:
            cluster_legend = 'Cluster' + str(label)

        plt.scatter(label_cluster['ftr1'], label_cluster['ftr2'], s = 70, \
                    edgecolor = 'k', marker = markers[label], label = cluster_legend)

        if iscenter:
            center_x_y = centers[label]
            plt.scatter(center_x_y[0], center_x_y[1], s = 250, color = 'white', alpha = 0.9, \
                        edgecolor = 'k', marker = markers[label])
            plt.scatter(center_x_y[0], center_x_y[1], s = 70, color = 'k', \
                        edgecolor = 'k', marker = '$%d%' %label)
            
    if isNoise:
        legend_loc = 'upper center'
    else:
        legend_loc = 'upper right'

    plt.legend(loc = legend_loc)
    plt.show()

pca = PCA(n_components = 2)
pca_transformed = pca.fit_transform(iris.data)

irisDF2['ftr1'] = pca_transformed[:, 0]
irisDF2['ftr2'] = pca_transformed[:, 1]

visualize_cluster_plot(dbscan, irisDF2, 'dbscan_cluster', iscenter = False)

dbscan2 = DBSCAN(eps = 0.6, min_samples = 16, metric = 'euclidean')
visualize_cluster_plot(dbscan2, irisDF2, 'dbscan_cluster', iscenter = False)
