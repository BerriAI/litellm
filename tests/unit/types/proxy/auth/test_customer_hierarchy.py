from typing import Final

from litellm.types.proxy.auth.customer_hierarchy import CustomerHierarchyIds, customer_hierarchy_ids_from_request

_IDS: Final = {
    "user_api_key_customer_member_id": "m1",
    "user_api_key_customer_team_id": "t1",
    "user_api_key_customer_org_id": "o1",
}


def test_ids_are_read_from_whichever_metadata_bucket_the_proxy_stamped() -> None:
    expected: Final = CustomerHierarchyIds(customer_org_id="o1", customer_team_id="t1", customer_member_id="m1")
    assert customer_hierarchy_ids_from_request({"litellm_params": {"metadata": _IDS}}) == expected
    assert (
        customer_hierarchy_ids_from_request({"litellm_params": {"metadata": {"note": "x"}, "litellm_metadata": _IDS}})
        == expected
    )
    assert customer_hierarchy_ids_from_request({"litellm_params": {"metadata": {"note": "x"}}}) is None


def test_a_request_that_named_only_an_organization_charges_the_organization_alone() -> None:
    ids: Final = customer_hierarchy_ids_from_request(
        {
            "litellm_params": {
                "metadata": {"user_api_key_customer_org_id": "o1", "user_api_key_customer_member_id": None}
            }
        }
    )
    assert ids == CustomerHierarchyIds(customer_org_id="o1")
    assert ids is not None
    assert tuple(entity_id for _, entity_id in ids.spend_entities()) == ("o1",)
    assert ids.entities() == {"customer_member_id": None, "customer_team_id": None, "customer_org_id": "o1"}
    member_only: Final = {"litellm_params": {"metadata": {"user_api_key_customer_member_id": "m1"}}}
    assert customer_hierarchy_ids_from_request(member_only) is None
