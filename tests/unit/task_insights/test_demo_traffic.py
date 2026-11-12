from litellm.task_insights.demo_traffic import DISCOVERY_PROMPTS, HELDOUT_PROMPTS, agreement, batch_prompts


def test_agreement_counts_unknown_labels_and_ignores_unlabeled_prompts():
    labels = {"a": "Debug", "b": "Docs", "c": "unknown", "d": ""}
    assigned = {"a": "Debug", "b": "Debug", "c": None, "d": "Debug"}

    assert agreement(labels, assigned) == (3, 2, 2)


def test_batches_have_stable_distinct_ids_and_no_overlap_between_discovery_and_heldout():
    discovery = batch_prompts("discovery")
    heldout = batch_prompts("heldout")

    assert batch_prompts("discovery") == discovery
    assert len({request_id for request_id, _ in discovery + heldout}) == len(discovery) + len(heldout)
    assert not {prompt for _, prompt in HELDOUT_PROMPTS} & set(DISCOVERY_PROMPTS)
