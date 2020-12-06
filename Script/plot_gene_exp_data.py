import matplotlib.pyplot as plt
import numpy as np
import sklearn as sk
import pandas as pd
import kmapper as km
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
import seaborn as sns
from sklearn import ensemble

URL = "https://raw.githubusercontent.com/sods/ods/master/datasets/guo_qpcr.csv"

df = pd.read_csv(URL, index_col=0)
print("Data shape: {}\n{}\n".format(df.shape, "-" * 21))
print("Data labels: {}\n{}\n".format(df.index.unique().tolist(), "-" * 86))
print("Show a small subset of the data:")
df.head()

capture_time = np.array([int(cell_name.split(" ")[0]) for cell_name in df.index.values])
y = df.values

# pca analysis
y_pre = scale(y, axis=0)
pca_res = PCA(n_components=3)
y_pca = pca_res.fit_transform(y_pre)
y_pca_df = pd.DataFrame(y_pca, columns=[f"PC{i}" for i in range(1, 4)])
y_pca_df["time"] = df.index

sns.scatterplot(data=y_pca_df, x="PC1", y="PC2", hue="time")

# Applying mapper algorithm
mapper = km.KeplerMapper(verbose=3)
lens2 = mapper.fit_transform(y_pre, projection="l2norm")
graph = mapper.map(y_pca,
                   y_pre,
                   cover=km.Cover(n_cubes=15, perc_overlap=0.4),
                   clusterer=sk.cluster.KMeans(n_clusters=2,
                                               random_state=1618033))
mapper.visualize(graph,
                 path_html="Analysis/output/gene-exp.html",
                 title="Gene Expression Dataset",
                 custom_tooltips=df.index.values)

graph2 = mapper.map(y_pca,
                   y_pre,
                   cover=km.Cover(n_cubes=10, perc_overlap=0.4),
                   clusterer=sk.cluster.KMeans(n_clusters=2,
                                               random_state=1618033))
mapper.visualize(graph2,
                 path_html="Analysis/output/gene-exp_low_resolution.html",
                 title="Gene Expression Dataset",
                 custom_tooltips=df.index.values)

graph3 = mapper.map(y_pca,
                   y_pre,
                   cover=km.Cover(n_cubes=8, perc_overlap=0.4),
                   clusterer=sk.cluster.KMeans(n_clusters=2,
                                               random_state=1618033))
mapper.visualize(graph3,
                 path_html="Analysis/output/gene-exp_low_resolution2.html",
                 title="Gene Expression Dataset",
                 custom_tooltips=df.index.values)

mapper.visualize(graph3,
                 path_html="Analysis/output/gene-exp_low_resolution2_2.html",
                 title="Gene Expression Dataset",
                 custom_tooltips=df.iloc[:, 1])

# try other mapper
# We create a custom 1-D lens with Isolation Forest
model = sk.ensemble.IsolationForest(random_state=1729)
model.fit(y_pre)
lens1 = model.decision_function(y_pre).reshape((y_pre.shape[0], 1))

# We create another 1-D lens with L2-norm
mapper = km.KeplerMapper(verbose=3)
lens2 = mapper.fit_transform(y_pre, projection="l2norm")
lens=np.c_[lens1, lens2]
graph_ot = mapper.map(lens,
                   y_pre,
                   cover=km.Cover(n_cubes=15, perc_overlap=0.4),
                   clusterer=sk.cluster.KMeans(n_clusters=2,
                                               random_state=1618033))
mapper.visualize(graph_ot,
                 path_html="Analysis/output/gene-exp_other.html",
                 title="Gene Expression Dataset",
                 custom_tooltips=df.index.values,
                 color_values=df.iloc[:, 1].values)

mapper.visualize(graph_ot,
                 path_html="Analysis/output/gene-exp_other_2.html",
                 title="Gene Expression Dataset",
                 custom_tooltips=df.index.values, color_values=df.iloc[:, 1].values)

# Applying persistent homology
