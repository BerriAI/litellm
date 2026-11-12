"""UMAP + HDBSCAN clustering for task discovery, run only by the analysis worker image.

Clustering runs on a UMAP reduction tuned for density, and the map uses a separate 2D UMAP, as the UMAP docs advise
(https://umap-learn.readthedocs.io/en/latest/clustering.html), since a projection tuned for display distorts density.
HDBSCAN is scikit-learn's maintained implementation
(https://scikit-learn.org/stable/modules/generated/sklearn.cluster.HDBSCAN.html). As a check on the reduction, HDBSCAN
also runs on the unreduced embeddings and the provenance records how far the two partitions agree.

The libraries are untyped and absent from the proxy image, so they load on use and their outputs are validated here.
"""

from importlib import import_module
from typing import Final

from pydantic import TypeAdapter

from litellm.task_insights.discovery import ClusterLayout

SEED: Final = 42
_LABELS: Final = TypeAdapter(tuple[int, ...])
_POINTS: Final = TypeAdapter(tuple[tuple[float, float], ...])
_SCORE: Final = TypeAdapter(float)


class UmapHdbscanClusterer:
    def __init__(self, *, cluster_dimensions: int = 10, neighbors: int = 15, seed: int = SEED) -> None:
        self._cluster_dimensions = cluster_dimensions
        self._neighbors = neighbors
        self._seed = seed

    def cluster(self, vectors: tuple[tuple[float, ...], ...]) -> ClusterLayout:
        numpy: Final = import_module("numpy")
        umap: Final = import_module("umap")
        cluster_module: Final = import_module("sklearn.cluster")
        metrics: Final = import_module("sklearn.metrics")
        size: Final = len(vectors)
        neighbors: Final = max(2, min(self._neighbors, size - 1))
        dimensions: Final = max(2, min(self._cluster_dimensions, size - 2))
        min_cluster_size: Final = max(5, size // 25)
        matrix: Final = numpy.asarray(vectors, dtype="float32")  # pyright: ignore[reportAny]  # untyped numpy module
        reduced: Final = umap.UMAP(  # pyright: ignore[reportAny]  # umap-learn ships no type information
            n_components=dimensions, n_neighbors=neighbors, min_dist=0.0, metric="cosine", random_state=self._seed
        ).fit_transform(matrix)
        display: Final = umap.UMAP(  # pyright: ignore[reportAny]  # umap-learn ships no type information
            n_components=2, n_neighbors=neighbors, min_dist=0.3, metric="cosine", random_state=self._seed
        ).fit_transform(matrix)
        labels: Final = _LABELS.validate_python(
            cluster_module.HDBSCAN(min_cluster_size=min_cluster_size)  # pyright: ignore[reportAny]  # untyped sklearn
            .fit_predict(reduced)
            .tolist()
        )
        raw_labels: Final = _LABELS.validate_python(
            cluster_module.HDBSCAN(  # pyright: ignore[reportAny]  # untyped sklearn
                min_cluster_size=min_cluster_size, metric="cosine", algorithm="brute"
            )
            .fit_predict(matrix)
            .tolist()
        )
        agreement: Final = _SCORE.validate_python(
            metrics.adjusted_rand_score(raw_labels, labels)  # pyright: ignore[reportAny]  # untyped sklearn
        )
        return ClusterLayout(
            labels=labels,
            points=_POINTS.validate_python(display.tolist()),  # pyright: ignore[reportAny]  # numpy array from umap
            provenance={
                "clusterer": "umap+hdbscan",
                "umap_cluster_dimensions": dimensions,
                "umap_neighbors": neighbors,
                "hdbscan_min_cluster_size": min_cluster_size,
                "seed": self._seed,
                "unreduced_cluster_count": len(frozenset(raw_labels) - {-1}),
                "unreduced_noise_count": sum(1 for label in raw_labels if label == -1),
                "reduced_vs_unreduced_adjusted_rand": round(agreement, 4),
            },
        )
