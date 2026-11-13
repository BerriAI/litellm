from typing import Final

from litellm.types.proxy.auth.customer_hierarchy import CustomerHierarchyIds, customer_hierarchy_ids_from_request

_IDS: Final = {
    "user_api_key_customer_member_id": "m1",
    "user_api_key_customer_team_id": "t1",
    "user_api_key_customer_org_id": "o1",
}


def test_ids_are_read_from_whichever_metadata_bucket_the_proxy_stamped() -> None:
    expected: Final = CustomerHierarchyIds("m1", "t1", "o1")
    assert customer_hierarchy_ids_from_request({"litellm_params": {"metadata": _IDS}}) == expected
    assert (
        customer_hierarchy_ids_from_request({"litellm_params": {"metadata": {"note": "x"}, "litellm_metadata": _IDS}})
        == expected
    )
    assert customer_hierarchy_ids_from_request({"litellm_params": {"metadata": {"note": "x"}}}) is None
