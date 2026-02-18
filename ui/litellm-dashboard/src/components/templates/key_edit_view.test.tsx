import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { renderWithProviders } from "../../../tests/test-utils";
import { KeyResponse } from "../key_team_helpers/key_list";
import { KeyEditView } from "./key_edit_view";

vi.mock("../networking", async () => {
  const actual = await vi.importActual("../networking");
  return {
    ...actual,
    getPromptsList: vi.fn().mockResolvedValue({
      prompts: [{ prompt_id: "prompt-1" }, { prompt_id: "prompt-2" }],
    }),
    modelAvailableCall: vi.fn().mockResolvedValue({
      data: [{ id: "gpt-4" }, { id: "gpt-3.5-turbo" }],
    }),
    tagListCall: vi.fn().mockResolvedValue({
      tag1: { name: "tag1", description: "Test tag 1" },
      tag2: { name: "tag2", description: "Test tag 2" },
    }),
    getGuardrailsList: vi.fn().mockResolvedValue({
      guardrails: [{ guardrail_name: "guardrail-1" }],
    }),
    getPoliciesList: vi.fn().mockResolvedValue({
      policies: [{ policy_name: "policy-1" }],
    }),
    getPassThroughEndpointsCall: vi.fn().mockResolvedValue({
      endpoints: [],
    }),
    vectorStoreListCall: vi.fn().mockResolvedValue({
      data: [],
    }),
    mcpToolsCall: vi.fn().mockResolvedValue({
      data: [],
    }),
    agentListCall: vi.fn().mockResolvedValue({
      data: [],
    }),
    fetchMCPServers: vi.fn().mockResolvedValue([]),
    fetchMCPAccessGroups: vi.fn().mockResolvedValue([]),
    listMCPTools: vi.fn().mockResolvedValue({
      tools: [],
      error: null,
      message: null,
      stack_trace: null,
    }),
    getAgentsList: vi.fn().mockResolvedValue({
      agents: [],
    }),
    getAgentAccessGroups: vi.fn().mockResolvedValue([]),
  };
});

vi.mock("../organisms/create_key_button", () => ({
  fetchTeamModels: vi.fn().mockResolvedValue(["team-model-1", "team-model-2"]),
}));

vi.mock("@/app/(dashboard)/hooks/accessGroups/useAccessGroups", () => ({
  useAccessGroups: vi.fn().mockReturnValue({
    data: [
      { access_group_id: "ag-1", access_group_name: "Group 1" },
      { access_group_id: "ag-2", access_group_name: "Group 2" },
    ],
    isLoading: false,
    isError: false,
  }),
}));

vi.mock("../common_components/AccessGroupSelector", () => ({
  default: ({ value = [], onChange }: { value?: string[]; onChange?: (v: string[]) => void }) => (
    <input
      data-testid="access-group-selector"
      value={Array.isArray(value) ? value.join(",") : ""}
      onChange={(e) => onChange?.(e.target.value ? e.target.value.split(",").map((s) => s.trim()) : [])}
    />
  ),
}));

describe("KeyEditView", () => {
  const MOCK_KEY_DATA: KeyResponse = {
    token: "test-token-123",
    token_id: "test-token-123",
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
  it("should render", async () => {
    const { getByText } = renderWithProviders(
      <KeyEditView
        keyData={MOCK_KEY_DATA}
        onCancel={() => { }}
        onSubmit={async () => { }}
        accessToken={""}
        userID={""}
        userRole={""}
        premiumUser={false}
      />,
    );

    await waitFor(() => {
      expect(getByText("Save Changes")).toBeInTheDocument();
    });
  });

  it("should render tags", async () => {
    const { getByText } = renderWithProviders(
      <KeyEditView
        keyData={MOCK_KEY_DATA}
        onCancel={() => { }}
        onSubmit={async () => { }}
        accessToken={""}
        userID={""}
        userRole={""}
        premiumUser={false}
      />,
    );

    await waitFor(() => {
      expect(getByText("test-tag")).toBeInTheDocument();
    });
  });

  it("should not render tags in metadata textarea", async () => {
    const { getByLabelText } = renderWithProviders(
      <KeyEditView
        keyData={MOCK_KEY_DATA}
        onCancel={() => { }}
        onSubmit={async () => { }}
        accessToken={""}
        userID={""}
        userRole={""}
        premiumUser={false}
      />,
    );

    const metadataTextarea = getByLabelText("Metadata") as HTMLTextAreaElement;
    await waitFor(() => {
      expect(metadataTextarea).toHaveValue("{}");
    });
  });

  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("should call onCancel when cancel button is clicked", async () => {
    const onCancelMock = vi.fn();
    renderWithProviders(
      <KeyEditView
        keyData={MOCK_KEY_DATA}
        onCancel={onCancelMock}
        onSubmit={async () => { }}
        accessToken={""}
        userID={""}
        userRole={""}
        premiumUser={false}
      />,
    );

    await waitFor(() => {
      expect(screen.getByRole("button", { name: /cancel/i })).toBeInTheDocument();
    });

    const cancelButton = screen.getByRole("button", { name: /cancel/i });
    await userEvent.click(cancelButton);

    expect(onCancelMock).toHaveBeenCalledTimes(1);
  });

  it("should display key alias input field", async () => {
    renderWithProviders(
      <KeyEditView
        keyData={MOCK_KEY_DATA}
        onCancel={() => { }}
        onSubmit={async () => { }}
        accessToken={""}
        userID={""}
        userRole={""}
        premiumUser={false}
      />,
    );

    await waitFor(() => {
      expect(screen.getByLabelText("Key Alias")).toBeInTheDocument();
    });
  });

  it("should display models select field", async () => {
    renderWithProviders(
      <KeyEditView
        keyData={MOCK_KEY_DATA}
        onCancel={() => { }}
        onSubmit={async () => { }}
        accessToken={""}
        userID={""}
        userRole={""}
        premiumUser={false}
      />,
    );

    await waitFor(() => {
      expect(screen.getByText("Models")).toBeInTheDocument();
    });
  });

  it("should display max budget input field", async () => {
    renderWithProviders(
      <KeyEditView
        keyData={MOCK_KEY_DATA}
        onCancel={() => { }}
        onSubmit={async () => { }}
        accessToken={""}
        userID={""}
        userRole={""}
        premiumUser={false}
      />,
    );

    await waitFor(() => {
      expect(screen.getByLabelText("Max Budget (USD)")).toBeInTheDocument();
    });
  });

  it("should display allowed routes input field", async () => {
    renderWithProviders(
      <KeyEditView
        keyData={MOCK_KEY_DATA}
        onCancel={() => { }}
        onSubmit={async () => { }}
        accessToken={""}
        userID={""}
        userRole={""}
        premiumUser={false}
      />,
    );

    await waitFor(() => {
      expect(screen.getByLabelText(/allowed routes/i)).toBeInTheDocument();
    });
  });

  it("should call onSubmit with form values when form is submitted", async () => {
    const onSubmitMock = vi.fn().mockResolvedValue(undefined);
    renderWithProviders(
      <KeyEditView
        keyData={MOCK_KEY_DATA}
        onCancel={() => { }}
        onSubmit={onSubmitMock}
        accessToken={"test-token"}
        userID={"test-user"}
        userRole={"admin"}
        premiumUser={false}
      />,
    );

    await waitFor(() => {
      expect(screen.getByRole("button", { name: /save changes/i })).toBeInTheDocument();
    });

    const submitButton = screen.getByRole("button", { name: /save changes/i });
    await userEvent.click(submitButton);

    await waitFor(() => {
      expect(onSubmitMock).toHaveBeenCalled();
    });
  });

  it("should disable models field when management routes are selected", async () => {
    const keyDataWithManagementRoutes = {
      ...MOCK_KEY_DATA,
      allowed_routes: ["management_routes"],
    };

    renderWithProviders(
      <KeyEditView
        keyData={keyDataWithManagementRoutes}
        onCancel={() => { }}
        onSubmit={async () => { }}
        accessToken={""}
        userID={""}
        userRole={""}
        premiumUser={false}
      />,
    );

    await waitFor(() => {
      expect(screen.getByText("Models field is disabled for this key type")).toBeInTheDocument();
    });
  });

  it("should disable models field when info routes are selected", async () => {
    const keyDataWithInfoRoutes = {
      ...MOCK_KEY_DATA,
      allowed_routes: ["info_routes"],
    };

    renderWithProviders(
      <KeyEditView
        keyData={keyDataWithInfoRoutes}
        onCancel={() => { }}
        onSubmit={async () => { }}
        accessToken={""}
        userID={""}
        userRole={""}
        premiumUser={false}
      />,
    );

    await waitFor(() => {
      expect(screen.getByText("Models field is disabled for this key type")).toBeInTheDocument();
    });
  });

  it("should disable guardrails selector when user is not premium", async () => {
    renderWithProviders(
      <KeyEditView
        keyData={MOCK_KEY_DATA}
        onCancel={() => { }}
        onSubmit={async () => { }}
        accessToken={"test-token"}
        userID={""}
        userRole={""}
        premiumUser={false}
      />,
    );

    await waitFor(() => {
      expect(screen.getByText("Guardrails")).toBeInTheDocument();
    });
  });

  it("should parse comma-separated allowed routes on submit", async () => {
    const onSubmitMock = vi.fn().mockResolvedValue(undefined);
    renderWithProviders(
      <KeyEditView
        keyData={MOCK_KEY_DATA}
        onCancel={() => { }}
        onSubmit={onSubmitMock}
        accessToken={"test-token"}
        userID={"test-user"}
        userRole={"admin"}
        premiumUser={false}
      />,
    );

    await waitFor(() => {
      expect(screen.getByLabelText(/allowed routes/i)).toBeInTheDocument();
    });

    const allowedRoutesInput = screen.getByLabelText(/allowed routes/i);
    await userEvent.clear(allowedRoutesInput);
    await userEvent.type(allowedRoutesInput, "route1, route2, route3");

    const submitButton = screen.getByRole("button", { name: /save changes/i });
    await userEvent.click(submitButton);

    await waitFor(() => {
      expect(onSubmitMock).toHaveBeenCalled();
      const callArgs = onSubmitMock.mock.calls[0][0];
      expect(Array.isArray(callArgs.allowed_routes)).toBe(true);
      expect(callArgs.allowed_routes).toEqual(["route1", "route2", "route3"]);
    });
  });

  it("should handle empty allowed routes string on submit", async () => {
    const onSubmitMock = vi.fn().mockResolvedValue(undefined);
    renderWithProviders(
      <KeyEditView
        keyData={MOCK_KEY_DATA}
        onCancel={() => { }}
        onSubmit={onSubmitMock}
        accessToken={"test-token"}
        userID={"test-user"}
        userRole={"admin"}
        premiumUser={false}
      />,
    );

    await waitFor(() => {
      expect(screen.getByLabelText(/allowed routes/i)).toBeInTheDocument();
    });

    const allowedRoutesInput = screen.getByLabelText(/allowed routes/i);
    await userEvent.clear(allowedRoutesInput);

    const submitButton = screen.getByRole("button", { name: /save changes/i });
    await userEvent.click(submitButton);

    await waitFor(() => {
      expect(onSubmitMock).toHaveBeenCalled();
      const callArgs = onSubmitMock.mock.calls[0][0];
      expect(callArgs.allowed_routes).toEqual([]);
    });
  });


  it("should pass access_group_ids to onSubmit when saving key with access groups", async () => {
    const onSubmitMock = vi.fn().mockResolvedValue(undefined);
    const keyDataWithAccessGroups = {
      ...MOCK_KEY_DATA,
      access_group_ids: ["ag-1"],
    };

    renderWithProviders(
      <KeyEditView
        keyData={keyDataWithAccessGroups}
        onCancel={() => {}}
        onSubmit={onSubmitMock}
        accessToken="test-token"
        userID="test-user"
        userRole="admin"
        premiumUser={false}
      />,
    );

    await waitFor(() => {
      expect(screen.getByTestId("access-group-selector")).toBeInTheDocument();
    });

    const accessGroupInput = screen.getByTestId("access-group-selector");
    await userEvent.clear(accessGroupInput);
    await userEvent.type(accessGroupInput, "ag-1,ag-2");

    const submitButton = screen.getByRole("button", { name: /save changes/i });
    await userEvent.click(submitButton);

    await waitFor(() => {
      expect(onSubmitMock).toHaveBeenCalled();
      const callArgs = onSubmitMock.mock.calls[0][0];
      expect(callArgs.access_group_ids).toEqual(["ag-1", "ag-2"]);
    });
  });

  it("should disable cancel button during submission", async () => {
    let resolveSubmit: (() => void) | undefined;
    const submitPromise = new Promise<void>((resolve) => {
      resolveSubmit = resolve;
    });
    const onSubmitMock = vi.fn(() => submitPromise);

    renderWithProviders(
      <KeyEditView
        keyData={MOCK_KEY_DATA}
        onCancel={() => { }}
        onSubmit={onSubmitMock}
        accessToken={"test-token"}
        userID={"test-user"}
        userRole={"admin"}
        premiumUser={false}
      />,
    );

    await waitFor(() => {
      expect(screen.getByRole("button", { name: /cancel/i })).toBeInTheDocument();
    });

    const submitButton = screen.getByRole("button", { name: /save changes/i });
    await userEvent.click(submitButton);

    // Wait for onSubmit to be called, which means handleSubmit has started and isKeySaving should be true
    await waitFor(() => {
      expect(onSubmitMock).toHaveBeenCalled();
    });

    // Wait for the cancel button to actually be disabled (state update may take a moment)
    await waitFor(() => {
      const cancelButton = screen.getByRole("button", { name: /cancel/i });
      expect(cancelButton).toBeDisabled();
    }, { timeout: 3000 });

    // Clean up: resolve the promise to allow the form to complete
    if (resolveSubmit) {
      resolveSubmit();
    }
  });
});
