from types import SimpleNamespace

from litellm.task_insights.clustering import UmapHdbscanClusterer


class _Array(list):
    def tolist(self):
        return list(self)


def _modules(calls: list[tuple[str, dict[str, object]]], reduced_labels, raw_labels):
    def umap(**kwargs):
        calls.append(("UMAP", kwargs))
        dimensions = kwargs["n_components"]
        return SimpleNamespace(
            fit_transform=lambda matrix: _Array([[float(index)] * dimensions for index in range(len(matrix))])
        )

    def hdbscan(**kwargs):
        calls.append(("HDBSCAN", kwargs))
        labels = raw_labels if kwargs.get("metric") == "cosine" else reduced_labels
        return SimpleNamespace(fit_predict=lambda matrix: _Array(labels))

    modules = {
        "numpy": SimpleNamespace(asarray=lambda vectors, dtype: list(vectors)),
        "umap": SimpleNamespace(UMAP=umap),
        "sklearn.cluster": SimpleNamespace(HDBSCAN=hdbscan),
        "sklearn.metrics": SimpleNamespace(adjusted_rand_score=lambda a, b: 1.0 if a == b else 0.25),
    }
    return modules.__getitem__


def test_clusters_on_a_reduction_and_draws_the_map_from_a_separate_2d_projection():
    calls: list[tuple[str, dict[str, object]]] = []
    labels = (0, 0, 1, 1, -1, 0)
    clusterer = UmapHdbscanClusterer(load=_modules(calls, labels, (0, 0, 1, 1, 1, 0)))

    layout = clusterer.cluster(tuple((float(index),) for index in range(6)))

    umap_calls = [kwargs for name, kwargs in calls if name == "UMAP"]
    assert [call["n_components"] for call in umap_calls] == [4, 2]
    assert {call["random_state"] for call in umap_calls} == {42}
    assert all(call["n_neighbors"] == 5 for call in umap_calls)
    assert layout.labels == labels
    assert all(len(point) == 2 for point in layout.points)
    assert layout.provenance["unreduced_noise_count"] == 0
    assert layout.provenance["reduced_vs_unreduced_adjusted_rand"] == 0.25


def test_cluster_size_scales_with_the_sample():
    calls: list[tuple[str, dict[str, object]]] = []
    clusterer = UmapHdbscanClusterer(load=_modules(calls, (0,) * 500, (0,) * 500))

    layout = clusterer.cluster(tuple((float(index),) for index in range(500)))

    sizes = {kwargs["min_cluster_size"] for name, kwargs in calls if name == "HDBSCAN"}
    assert sizes == {20}
    assert layout.provenance["umap_cluster_dimensions"] == 10
