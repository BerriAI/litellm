import { render, waitFor } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { KeyResponse } from "../key_team_helpers/key_list";
import KeyInfoView from "./key_info_view";

describe("KeyInfoView", () => {
  const MOCK_KEY_DATA: KeyResponse = {
    token: "40b7608ea43423400d5b82bb5ee11042bfb2ed4655f05b5992b5abbc2f294931",
    token_id: "40b7608ea43423400d5b82bb5ee11042bfb2ed4655f05b5992b5abbc2f294931",
    key_name: "sk-...TUuw",
    key_alias: "asdasdas",
    spend: 0,
    max_budget: 0,
    expires: "null",
    models: [],
    aliases: {},
    config: {},
    user_id: "default_user_id",
    team_id: null,
    max_parallel_requests: 10,
    metadata: {
      logging: [],
      tags: ["test-tag"],
    },
    tpm_limit: 10,
    rpm_limit: 10,
    duration: "30d",
    budget_duration: "30d",
    budget_reset_at: "never",
    allowed_cache_controls: [],
    allowed_routes: [],
    permissions: {},
    model_spend: {},
    model_max_budget: {},
    soft_budget_cooldown: false,
    blocked: false,
    litellm_budget_table: {},
    organization_id: null,
    created_at: "2025-10-29T01:26:41.613000Z",
    updated_at: "2025-10-29T01:47:33.980000Z",
    team_spend: 100,
    team_alias: "",
    team_tpm_limit: 100,
    team_rpm_limit: 100,
    team_max_budget: 100,
    team_models: [],
    team_blocked: false,
    soft_budget: 200,
    team_model_aliases: {},
    team_member_spend: 0,
    team_metadata: {},
    end_user_id: "default_user_id",
    end_user_tpm_limit: 10,
    end_user_rpm_limit: 10,
    end_user_max_budget: 0,
    last_refreshed_at: Date.now(),
    api_key: "sk-...TUuw",
    user_role: "user",
    rpm_limit_per_model: {},
    tpm_limit_per_model: {},
    user_tpm_limit: 10,
    user_rpm_limit: 10,
    user_email: "test@example.com",
    object_permission: {
      object_permission_id: "067002ed-3b01-4bb3-b942-cefa400f0049",
      mcp_servers: [],
      mcp_access_groups: [],
      mcp_tool_permissions: {},
      vector_stores: [],
    },
    auto_rotate: false,
    rotation_interval: undefined,
    last_rotation_at: undefined,
    key_rotation_at: undefined,
  };

  it("should render tags", async () => {
    const { getByText } = render(
      <KeyInfoView
        keyData={MOCK_KEY_DATA}
        onClose={() => {}}
        keyId={"test-key-id"}
        onKeyDataUpdate={() => {}}
        accessToken={"test-token"}
        userID={"test-user"}
        userRole={"admin"}
        premiumUser={true}
        teams={[]}
      />,
    );
    await waitFor(() => {
      expect(getByText("test-tag")).toBeInTheDocument();
    });
  });

  it("should not render tags in metadata textarea", async () => {
    const { container, getByText } = render(
      <KeyInfoView
        keyData={MOCK_KEY_DATA}
        onClose={() => {}}
        keyId={"test-key-id"}
        onKeyDataUpdate={() => {}}
        accessToken={"test-token"}
        userID={"test-user"}
        userRole={"admin"}
        premiumUser={true}
        teams={[]}
      />,
    );
    await waitFor(() => {
      expect(getByText("Metadata")).toBeInTheDocument();
      const metadataBlock = container.querySelector("pre");
      expect(metadataBlock).toBeInTheDocument();
      expect(metadataBlock?.textContent?.trim()).toBe("{}");
    });
  });
});
