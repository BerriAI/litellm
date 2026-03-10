from litellm import Router


class NoItemsAliasDict(dict):
    def items(self):
        raise AssertionError("Unexpected full alias iteration via items()")


def test_get_model_list_from_model_alias_should_not_iterate_for_non_alias_lookup():
    router = Router(
        model_list=[
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {"model": "gpt-3.5-turbo"},
            }
        ],
        model_group_alias={"alias-1": "gpt-4"},
    )
    router.model_group_alias = NoItemsAliasDict(
        {f"alias-{idx}": "gpt-4" for idx in range(200)}
    )

    model_alias_list = router.get_model_list_from_model_alias(
        model_name="gpt-3.5-turbo"
    )
    assert model_alias_list == []


def test_map_team_model_should_not_iterate_aliases_for_non_alias_team_model_name():
    router = Router(
        model_list=[
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {"model": "gpt-3.5-turbo"},
                "model_info": {
                    "team_id": "team-1",
                    "team_public_model_name": "team-model",
                },
            }
        ],
        model_group_alias={"alias-1": "gpt-4"},
    )
    router.model_group_alias = NoItemsAliasDict(
        {f"alias-{idx}": "gpt-4" for idx in range(200)}
    )

    assert (
        router.map_team_model(team_model_name="team-model", team_id="team-1")
        == "gpt-3.5-turbo"
    )
