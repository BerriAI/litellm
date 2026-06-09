"""Pure key builders.

Every key a single ``EVAL`` touches together carries the same Redis Cluster
hash tag ``{policy_id:subject_kind:subject_id}`` so ``counter`` and ``window``
colocate on one slot; without it a two-key ``EVAL`` throws ``CROSSSLOT`` under
Redis Cluster.
"""

from litellm.integrations.governor.model.subjects import Subject

_NAMESPACE = "gov"


def _subject_tag(policy_id: str, subject: Subject) -> str:
    return f"{{{policy_id}:{subject.kind}:{subject.id}}}"


def counter_key(policy_id: str, subject: Subject) -> str:
    return f"{_NAMESPACE}:{_subject_tag(policy_id, subject)}:counter"


def window_key(policy_id: str, subject: Subject) -> str:
    return f"{_NAMESPACE}:{_subject_tag(policy_id, subject)}:window"


def gcra_key(policy_id: str, subject: Subject) -> str:
    return f"{_NAMESPACE}:{_subject_tag(policy_id, subject)}:gcra"


def inflight_key(policy_id: str, subject: Subject) -> str:
    return f"{_NAMESPACE}:{_subject_tag(policy_id, subject)}:inflight"


def reconciled_key(policy_id: str, request_id: str) -> str:
    return f"{_NAMESPACE}:reconciled:{{{policy_id}:{request_id}}}"
