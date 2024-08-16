import pandas as pd
import seaborn as sns
import umap
from gtda.diagrams import Amplitude
from gtda.diagrams import NumberOfPoints
from gtda.diagrams import PersistenceEntropy
from gtda.diagrams import Scaler
from gtda.homology import CubicalPersistence
from gtda.images import Binarizer
from gtda.images import SignedDistanceFiltration
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import normalize


def main():
    """
    This function fetches the mnist_784 dataset using fetch_openml and prints the shapes of X and y.

    Returns:
    None
    """

    X_pre, y = fetch_openml("mnist_784", version=1, return_X_y=True)
    X = X_pre.values

    X = X.reshape((-1, 28, 28))

    metric_list = [
        {"metric": "bottleneck", "metric_params": {}},
        {"metric": "betti", "metric_params": {"p": 1, "n_bins": 100}},
        {"metric": "betti", "metric_params": {"p": 2, "n_bins": 100}},
        {"metric": "persistence_image", "metric_params": {"n_bins": 100}},
        {"metric": "landscape", "metric_params": {"p": 1, "n_layers": 1, "n_bins": 100}},
        {"metric": "landscape", "metric_params": {"p": 1, "n_layers": 2, "n_bins": 100}},
        {"metric": "landscape", "metric_params": {"p": 2, "n_layers": 1, "n_bins": 100}},
        {"metric": "landscape", "metric_params": {"p": 2, "n_layers": 2, "n_bins": 100}},
    ]

    train_size, test_size = 1000, 400
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=train_size, test_size=test_size, stratify=y, random_state=666
    )

    diagram_steps = [
        Binarizer(threshold=0.4, n_jobs=-1),
        SignedDistanceFiltration(n_jobs=-1),
        CubicalPersistence(n_jobs=-1),
        Scaler(n_jobs=-1)
    ]

    feature_union = make_union(
        *[NumberOfPoints(n_jobs=-1)] + [PersistenceEntropy(nan_fill_value=-1)] + [
            Amplitude(**metric, n_jobs=-1) for metric in metric_list
        ]
    )

    tda_union = make_pipeline(*diagram_steps, feature_union)

    X_train_tda = tda_union.fit_transform(X_train)
    print(X_train_tda.shape)

    reducer = umap.UMAP()

    X_train_tda_n = normalize(X_train_tda)
    embedding = reducer.fit_transform(X_train_tda_n)

    df_summary = pd.DataFrame(
        dict(var1=embedding[:, 0], var2=embedding[:, 1],
             label=y_train)
    )

    fig1 = sns.scatterplot(data=df_summary, x="var1", y="var2", hue="label", alpha=0.7)
    fig1.legend(loc='center left', bbox_to_anchor=(1.05, 0.5), ncol=1)
