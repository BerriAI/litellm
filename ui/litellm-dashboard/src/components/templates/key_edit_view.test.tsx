import { fireEvent, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { renderWithProviders } from "../../../tests/test-utils";
import { KeyResponse } from "../key_team_helpers/key_list";
import { MODEL_MAX_BUDGET_PREMIUM_HINT } from "../key_team_helpers/ModelMaxBudgetEditor";
import {
  getPassThroughEndpointsCall,
  getPoliciesList,
  getPromptsList,
  modelAvailableCall,
  vectorStoreListCall,
} from "../networking";
import { KeyEditView } from "./key_edit_view";

const can = vi.fn();
vi.mock("@/app/(dashboard)/hooks/useCan", () => ({
  default: (...args: unknown[]) => can(...args),
}));

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

const routerSettingsMocks = vi.hoisted(() => ({
  receivedValue: undefined as { router_settings: Record<string, unknown> } | undefined,
  editedValue: null as Record<string, unknown> | null,
}));

vi.mock("../common_components/RouterSettingsAccordion", async () => {
  const { forwardRef, useImperativeHandle } = await import("react");
  return {
    default: forwardRef(({ value }: { value?: { router_settings: Record<string, unknown> } }, ref) => {
      routerSettingsMocks.receivedValue = value;
      useImperativeHandle(ref, () => ({
        getValue: () => ({ router_settings: routerSettingsMocks.editedValue ?? value?.router_settings ?? {} }),
      }));
      return <div data-testid="router-settings-accordion" />;
    }),
  };
});

vi.mock("@/app/(dashboard)/hooks/organizations/useOrganizations", () => ({
  useOrganizations: vi.fn().mockReturnValue({
    data: [
      { organization_id: "org-1", organization_alias: "Engineering" },
      { organization_id: "org-2", organization_alias: "Sales" },
    ],
    isLoading: false,
  }),
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

vi.mock("../mcp_server_management/MCPServerSelector", () => ({
  default: ({
    value,
    onChange,
  }: {
    value?: { servers?: string[]; accessGroups?: string[]; toolsets?: string[] };
    onChange?: (v: { servers: string[]; accessGroups: string[]; toolsets: string[] }) => void;
  }) => (
    <button
      type="button"
      data-testid="mcp-server-selector"
      onClick={() => onChange?.({ servers: ["mcp-1"], accessGroups: [], toolsets: value?.toolsets ?? [] })}
    >
      pick mcp server
    </button>
  ),
}));

vi.mock("../agent_management/AgentSelector", () => ({
  default: ({ onChange }: { onChange?: (v: { agents: string[]; accessGroups: string[] }) => void }) => (
    <button
      type="button"
      data-testid="agent-selector"
      onClick={() => onChange?.({ agents: ["agent-1"], accessGroups: [] })}
    >
      pick agent
    </button>
  ),
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

const visibleOptions = (): HTMLElement[] => screen.queryAllByRole("option");

const isOptionDisabled = (option: HTMLElement): boolean => option.getAttribute("aria-disabled") === "true";

const optionByContent = (label: string): HTMLElement | undefined =>
  visibleOptions().find((el) => el.textContent === label);

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
  describe("router settings", () => {
    const UNSUPPORTED_STORED_FIELD = { tag_routing_prefix: "team-" };
    const STORED_ROUTER_SETTINGS = {
      num_retries: 2,
      fallbacks: [{ "gpt-4": ["gpt-4o"] }],
      ...UNSUPPORTED_STORED_FIELD,
    };

    const renderWithRouterSettings = (onSubmit: (values: Record<string, unknown>) => Promise<void>) =>
      renderWithProviders(
        <KeyEditView
          keyData={{ ...MOCK_KEY_DATA, router_settings: STORED_ROUTER_SETTINGS }}
          onCancel={() => {}}
          onSubmit={onSubmit}
          accessToken="test-token"
          userID="test-user"
          userRole="proxy_admin"
          premiumUser={true}
        />,
      );

    beforeEach(() => {
      routerSettingsMocks.receivedValue = undefined;
      routerSettingsMocks.editedValue = null;
    });

    it("should load the fields it renders into the editor and withhold the ones it does not", async () => {
      renderWithRouterSettings(async () => {});

      await waitFor(() => {
        expect(routerSettingsMocks.receivedValue).toStrictEqual({
          router_settings: { num_retries: 2, fallbacks: [{ "gpt-4": ["gpt-4o"] }] },
        });
      });
    });

    it("should submit edited fallbacks alongside routing fields the editor cannot show", async () => {
      const onSubmit = vi.fn().mockResolvedValue(undefined);
      renderWithRouterSettings(onSubmit);
      routerSettingsMocks.editedValue = { num_retries: 2, fallbacks: [{ "gpt-4": ["gpt-4o", "gpt-4o-mini"] }] };

      fireEvent.click(screen.getByText("Save Changes"));

      await waitFor(() => {
        expect(onSubmit).toHaveBeenCalledWith(
          expect.objectContaining({
            router_settings: expect.objectContaining({
              ...UNSUPPORTED_STORED_FIELD,
              num_retries: 2,
              fallbacks: [{ "gpt-4": ["gpt-4o", "gpt-4o-mini"] }],
            }),
          }),
        );
      });
    });

    it("should submit cleared router settings so removing every fallback is persisted", async () => {
      const onSubmit = vi.fn().mockResolvedValue(undefined);
      renderWithRouterSettings(onSubmit);
      routerSettingsMocks.editedValue = { num_retries: null, fallbacks: null };

      fireEvent.click(screen.getByText("Save Changes"));

      await waitFor(() => {
        expect(onSubmit).toHaveBeenCalledWith(
          expect.objectContaining({
            router_settings: expect.objectContaining({
              ...UNSUPPORTED_STORED_FIELD,
              num_retries: null,
              fallbacks: null,
            }),
          }),
        );
      });
    });
  });

  it("should render", async () => {
    const { getByText } = renderWithProviders(
      <KeyEditView
        keyData={MOCK_KEY_DATA}
        onCancel={() => {}}
        onSubmit={async () => {}}
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
        onCancel={() => {}}
        onSubmit={async () => {}}
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
        onCancel={() => {}}
        onSubmit={async () => {}}
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
    can.mockReturnValue(true);
  });

  describe("policy and prompt fields", () => {
    const renderAs = (userRole: string) =>
      renderWithProviders(
        <KeyEditView
          keyData={MOCK_KEY_DATA}
          onCancel={() => {}}
          onSubmit={async () => {}}
          accessToken="test-token"
          userID="user-123"
          userRole={userRole}
          premiumUser={true}
        />,
      );

    it("locks the prompts control for a non-premium admin so an unsavable value cannot be entered", async () => {
      renderWithProviders(
        <KeyEditView
          keyData={MOCK_KEY_DATA}
          onCancel={() => {}}
          onSubmit={async () => {}}
          accessToken={"test-token"}
          userID={"test-user"}
          userRole={"Admin"}
          premiumUser={false}
        />,
      );

      const prompts = await screen.findByLabelText(/Prompts/);
      expect(prompts).toBeDisabled();

      await userEvent.type(prompts, "sneaky-prompt{Enter}");

      expect(screen.queryByLabelText("sneaky-prompt")).not.toBeInTheDocument();
    });

    it("leaves the prompts control usable for a premium admin", async () => {
      renderWithProviders(
        <KeyEditView
          keyData={MOCK_KEY_DATA}
          onCancel={() => {}}
          onSubmit={async () => {}}
          accessToken={"test-token"}
          userID={"test-user"}
          userRole={"Admin"}
          premiumUser={true}
        />,
      );

      const prompts = await screen.findByLabelText(/Prompts/);
      expect(prompts).toBeEnabled();

      await userEvent.type(prompts, "allowed-prompt{Enter}");

      expect(await screen.findByLabelText("allowed-prompt")).toBeInTheDocument();
    });

    it("should render both fields and load prompts for an admin", async () => {
      renderAs("Admin");

      await waitFor(() => {
        expect(getPromptsList).toHaveBeenCalledWith("test-token");
      });
      expect(screen.getByText("Prompts", { selector: "label" })).toBeInTheDocument();
      expect(screen.getByText("Policies")).toBeInTheDocument();
    });

    it("should omit both fields and fire neither admin-only request for an internal user", async () => {
      renderAs("Internal User");

      await waitFor(() => {
        expect(modelAvailableCall).toHaveBeenCalled();
      });

      expect(getPromptsList).not.toHaveBeenCalled();
      expect(getPoliciesList).not.toHaveBeenCalled();
      expect(screen.queryByText("Prompts", { selector: "label" })).not.toBeInTheDocument();
      expect(screen.queryByText("Policies")).not.toBeInTheDocument();
    });
  });

  it("should call onCancel without submitting the form when cancel button is clicked", async () => {
    const onCancelMock = vi.fn();
    const onSubmitMock = vi.fn().mockResolvedValue(undefined);
    renderWithProviders(
      <KeyEditView
        keyData={MOCK_KEY_DATA}
        onCancel={onCancelMock}
        onSubmit={onSubmitMock}
        accessToken={""}
        userID={""}
        userRole={""}
        premiumUser={false}
      />,
    );

    const cancelButton = await screen.findByRole("button", { name: /cancel/i });
    await userEvent.click(cancelButton);

    expect(onCancelMock).toHaveBeenCalledTimes(1);
    expect(onSubmitMock).not.toHaveBeenCalled();
  });

  it("should display key alias input field", async () => {
    renderWithProviders(
      <KeyEditView
        keyData={MOCK_KEY_DATA}
        onCancel={() => {}}
        onSubmit={async () => {}}
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
        onCancel={() => {}}
        onSubmit={async () => {}}
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
        onCancel={() => {}}
        onSubmit={async () => {}}
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
        onCancel={() => {}}
        onSubmit={async () => {}}
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
        onCancel={() => {}}
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

  it("should initialize and submit throttle_on_budget_exceeded from key metadata", async () => {
    const onSubmitMock = vi.fn().mockResolvedValue(undefined);
    const keyDataWithThrottle = {
      ...MOCK_KEY_DATA,
      metadata: { ...MOCK_KEY_DATA.metadata, throttle_on_budget_exceeded: true },
    };

    renderWithProviders(
      <KeyEditView
        keyData={keyDataWithThrottle}
        onCancel={() => {}}
        onSubmit={onSubmitMock}
        accessToken={"test-token"}
        userID={"test-user"}
        userRole={"admin"}
        premiumUser={false}
      />,
    );

    await waitFor(() => {
      expect(screen.getByText("Throttle on budget exceeded")).toBeInTheDocument();
    });

    await userEvent.click(screen.getByRole("button", { name: /save changes/i }));

    await waitFor(() => {
      expect(onSubmitMock).toHaveBeenCalledWith(expect.objectContaining({ throttle_on_budget_exceeded: true }));
    });
  });

  it("should initialize and submit enable_prompt_caching from key metadata", async () => {
    const onSubmitMock = vi.fn().mockResolvedValue(undefined);
    const keyDataWithPromptCaching = {
      ...MOCK_KEY_DATA,
      metadata: { ...MOCK_KEY_DATA.metadata, enable_prompt_caching: true },
    };

    renderWithProviders(
      <KeyEditView
        keyData={keyDataWithPromptCaching}
        onCancel={() => {}}
        onSubmit={onSubmitMock}
        accessToken={"test-token"}
        userID={"test-user"}
        userRole={"admin"}
        premiumUser={false}
      />,
    );

    await waitFor(() => {
      expect(screen.getByText("Enable Prompt Caching")).toBeInTheDocument();
    });

    await userEvent.click(screen.getByRole("button", { name: /save changes/i }));

    await waitFor(() => {
      expect(onSubmitMock).toHaveBeenCalledWith(expect.objectContaining({ enable_prompt_caching: true }));
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
        onCancel={() => {}}
        onSubmit={async () => {}}
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
        onCancel={() => {}}
        onSubmit={async () => {}}
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

  it("should disable guardrails selector when user is not premium and has no write access role", async () => {
    renderWithProviders(
      <KeyEditView
        keyData={MOCK_KEY_DATA}
        onCancel={() => {}}
        onSubmit={async () => {}}
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
        onCancel={() => {}}
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
    const keyDataWithRoutes = {
      ...MOCK_KEY_DATA,
      allowed_routes: ["llm_api_routes"],
    };
    renderWithProviders(
      <KeyEditView
        keyData={keyDataWithRoutes}
        onCancel={() => {}}
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

  it("should omit allowed_routes from submit when value is unchanged", async () => {
    const onSubmitMock = vi.fn().mockResolvedValue(undefined);
    const aiApisKeyData = {
      ...MOCK_KEY_DATA,
      allowed_routes: ["llm_api_routes"],
    };
    renderWithProviders(
      <KeyEditView
        keyData={aiApisKeyData}
        onCancel={() => {}}
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
      const callArgs = onSubmitMock.mock.calls[0][0];
      expect("allowed_routes" in callArgs).toBe(false);
    });
  });

  it("should omit allowed_routes from submit when keyData.allowed_routes is null and form is untouched", async () => {
    const onSubmitMock = vi.fn().mockResolvedValue(undefined);
    const keyDataNullRoutes = {
      ...MOCK_KEY_DATA,
      allowed_routes: null as unknown as string[],
    };
    renderWithProviders(
      <KeyEditView
        keyData={keyDataNullRoutes}
        onCancel={() => {}}
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
      const callArgs = onSubmitMock.mock.calls[0][0];
      expect("allowed_routes" in callArgs).toBe(false);
    });
  });

  it("should omit allowed_routes from submit when server returned routes in a different order", async () => {
    const onSubmitMock = vi.fn().mockResolvedValue(undefined);
    const keyDataReordered = {
      ...MOCK_KEY_DATA,
      allowed_routes: ["beta_routes", "alpha_routes"],
    };
    renderWithProviders(
      <KeyEditView
        keyData={keyDataReordered}
        onCancel={() => {}}
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
      const callArgs = onSubmitMock.mock.calls[0][0];
      expect("allowed_routes" in callArgs).toBe(false);
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

  it("should keep mcp_toolsets when saving an edit that does not touch the MCP selector", async () => {
    const onSubmitMock = vi.fn().mockResolvedValue(undefined);
    const keyDataWithToolset = {
      ...MOCK_KEY_DATA,
      object_permission: {
        ...MOCK_KEY_DATA.object_permission!,
        mcp_toolsets: ["ts-1"],
      },
    };

    renderWithProviders(
      <KeyEditView
        keyData={keyDataWithToolset}
        onCancel={() => {}}
        onSubmit={onSubmitMock}
        accessToken="test-token"
        userID="test-user"
        userRole="admin"
        premiumUser={false}
      />,
    );

    await waitFor(() => {
      expect(screen.getByText("Save Changes")).toBeInTheDocument();
    });

    await userEvent.click(screen.getByRole("button", { name: /save changes/i }));

    await waitFor(() => {
      expect(onSubmitMock).toHaveBeenCalled();
    });
    expect(onSubmitMock.mock.calls[0][0].mcp_servers_and_groups.toolsets).toEqual(["ts-1"]);
  });

  it("should submit budget_limits: [] when the last budget window is deleted", async () => {
    const onSubmitMock = vi.fn().mockResolvedValue(undefined);
    const keyDataWithWindow = {
      ...MOCK_KEY_DATA,
      budget_limits: [{ budget_duration: "30d", max_budget: 100 }],
    };
    renderWithProviders(
      <KeyEditView
        keyData={keyDataWithWindow}
        onCancel={() => {}}
        onSubmit={onSubmitMock}
        accessToken={"test-token"}
        userID={"test-user"}
        userRole={"admin"}
        premiumUser={false}
      />,
    );

    const deleteWindowButton = await screen.findByRole("button", { name: "✕" });
    await userEvent.click(deleteWindowButton);

    const submitButton = screen.getByRole("button", { name: /save changes/i });
    await userEvent.click(submitButton);

    await waitFor(() => {
      expect(onSubmitMock).toHaveBeenCalled();
      const callArgs = onSubmitMock.mock.calls[0][0];
      expect(callArgs.budget_limits).toEqual([]);
    });
  });

  it("should persist a canonical budget_duration value, not a word-form the backend cannot parse", async () => {
    const onSubmitMock = vi.fn().mockResolvedValue(undefined);
    renderWithProviders(
      <KeyEditView
        keyData={MOCK_KEY_DATA}
        onCancel={() => {}}
        onSubmit={onSubmitMock}
        accessToken={"test-token"}
        userID={"test-user"}
        userRole={"admin"}
        premiumUser={false}
      />,
    );

    await userEvent.click(await screen.findByLabelText("Reset Budget"));

    const weeklyOption = await screen.findByText("weekly");
    await userEvent.click(weeklyOption);

    const submitButton = screen.getByRole("button", { name: /save changes/i });
    await userEvent.click(submitButton);

    await waitFor(() => {
      expect(onSubmitMock).toHaveBeenCalled();
      const callArgs = onSubmitMock.mock.calls[0][0];
      expect(callArgs.budget_duration).toBe("7d");
    });
  });

  it("should keep an existing canonical budget_duration canonical when saved untouched", async () => {
    const onSubmitMock = vi.fn().mockResolvedValue(undefined);
    renderWithProviders(
      <KeyEditView
        keyData={MOCK_KEY_DATA}
        onCancel={() => {}}
        onSubmit={onSubmitMock}
        accessToken={"test-token"}
        userID={"test-user"}
        userRole={"admin"}
        premiumUser={false}
      />,
    );

    const submitButton = await screen.findByRole("button", { name: /save changes/i });
    await userEvent.click(submitButton);

    await waitFor(() => {
      expect(onSubmitMock).toHaveBeenCalled();
      const callArgs = onSubmitMock.mock.calls[0][0];
      expect(callArgs.budget_duration).toBe("30d");
    });
  });

  it("should heal a legacy word-form budget_duration to canonical when saved untouched", async () => {
    const onSubmitMock = vi.fn().mockResolvedValue(undefined);
    const legacyKeyData = { ...MOCK_KEY_DATA, budget_duration: "monthly" };
    renderWithProviders(
      <KeyEditView
        keyData={legacyKeyData}
        onCancel={() => {}}
        onSubmit={onSubmitMock}
        accessToken={"test-token"}
        userID={"test-user"}
        userRole={"admin"}
        premiumUser={false}
      />,
    );

    const submitButton = await screen.findByRole("button", { name: /save changes/i });
    await userEvent.click(submitButton);

    await waitFor(() => {
      expect(onSubmitMock).toHaveBeenCalled();
      const callArgs = onSubmitMock.mock.calls[0][0];
      expect(callArgs.budget_duration).toBe("30d");
    });
  });

  it("should send an explicit null budget_duration when a previously-set Reset Budget is cleared", async () => {
    const onSubmitMock = vi.fn().mockResolvedValue(undefined);
    renderWithProviders(
      <KeyEditView
        keyData={MOCK_KEY_DATA}
        onCancel={() => {}}
        onSubmit={onSubmitMock}
        accessToken={"test-token"}
        userID={"test-user"}
        userRole={"admin"}
        premiumUser={false}
      />,
    );

    const resetBudget = await screen.findByLabelText("Reset Budget");
    await userEvent.click(resetBudget);
    await userEvent.click(await screen.findByText("Never resets"));

    await waitFor(() => {
      expect(resetBudget).toHaveTextContent("Never resets");
    });

    await userEvent.click(screen.getByRole("button", { name: /save changes/i }));

    await waitFor(() => {
      expect(onSubmitMock).toHaveBeenCalled();
    });
    const callArgs = onSubmitMock.mock.calls[0][0];
    expect(callArgs.budget_duration).toBeNull();
    expect(JSON.stringify({ ...callArgs })).toContain('"budget_duration":null');
  });

  it("should send an explicit null budget_duration when a legacy word-form Reset Budget is cleared", async () => {
    const onSubmitMock = vi.fn().mockResolvedValue(undefined);
    const legacyKeyData = { ...MOCK_KEY_DATA, budget_duration: "monthly" };
    renderWithProviders(
      <KeyEditView
        keyData={legacyKeyData}
        onCancel={() => {}}
        onSubmit={onSubmitMock}
        accessToken={"test-token"}
        userID={"test-user"}
        userRole={"admin"}
        premiumUser={false}
      />,
    );

    await userEvent.click(await screen.findByLabelText("Reset Budget"));
    await userEvent.click(await screen.findByText("Never resets"));

    await userEvent.click(screen.getByRole("button", { name: /save changes/i }));

    await waitFor(() => {
      expect(onSubmitMock).toHaveBeenCalled();
    });
    expect(onSubmitMock.mock.calls[0][0].budget_duration).toBeNull();
  });

  it("should omit budget_limits when existing windows are left untouched (issue #33246)", async () => {
    // The backend treats any budget_limits in the payload as an admin-only
    // budget change, so re-sending untouched windows 403s a non-admin owner.
    // Leaving the field off keeps the stored windows and passes the gate.
    const onSubmitMock = vi.fn().mockResolvedValue(undefined);
    const keyDataWithWindow = {
      ...MOCK_KEY_DATA,
      budget_limits: [{ budget_duration: "30d", max_budget: 100, reset_at: "2026-08-01T00:00:00" }],
    };
    renderWithProviders(
      <KeyEditView
        keyData={keyDataWithWindow}
        onCancel={() => {}}
        onSubmit={onSubmitMock}
        accessToken={"test-token"}
        userID={"test-user"}
        userRole={"admin"}
        premiumUser={false}
      />,
    );

    const submitButton = await screen.findByRole("button", { name: /save changes/i });
    await userEvent.click(submitButton);

    await waitFor(() => {
      expect(onSubmitMock).toHaveBeenCalled();
      const callArgs = onSubmitMock.mock.calls[0][0];
      expect(callArgs.budget_limits).toBeUndefined();
    });
  });

  it("should omit budget_limits on a key that has no windows (issue #33246 repro)", async () => {
    // Core repro: a non-admin owner edits a non-budget field on a key with no
    // budget windows. The form previously always sent budget_limits: [], which
    // the backend read as a budget change and rejected.
    const onSubmitMock = vi.fn().mockResolvedValue(undefined);
    renderWithProviders(
      <KeyEditView
        keyData={MOCK_KEY_DATA} // no budget_limits
        onCancel={() => {}}
        onSubmit={onSubmitMock}
        accessToken={"test-token"}
        userID={"test-user"}
        userRole={"admin"}
        premiumUser={false}
      />,
    );

    const submitButton = await screen.findByRole("button", { name: /save changes/i });
    await userEvent.click(submitButton);

    await waitFor(() => {
      expect(onSubmitMock).toHaveBeenCalled();
      const callArgs = onSubmitMock.mock.calls[0][0];
      expect(callArgs.budget_limits).toBeUndefined();
    });
  });

  it("should send budget_limits when a window's cap is changed", async () => {
    const onSubmitMock = vi.fn().mockResolvedValue(undefined);
    const keyDataWithWindow = {
      ...MOCK_KEY_DATA,
      budget_limits: [{ budget_duration: "30d", max_budget: 100 }],
    };
    renderWithProviders(
      <KeyEditView
        keyData={keyDataWithWindow}
        onCancel={() => {}}
        onSubmit={onSubmitMock}
        accessToken={"test-token"}
        userID={"test-user"}
        userRole={"admin"}
        premiumUser={false}
      />,
    );

    const maxBudgetInput = await screen.findByPlaceholderText("Max spend ($)");
    await userEvent.clear(maxBudgetInput);
    await userEvent.type(maxBudgetInput, "200");

    const submitButton = screen.getByRole("button", { name: /save changes/i });
    await userEvent.click(submitButton);

    await waitFor(() => {
      expect(onSubmitMock).toHaveBeenCalled();
      const callArgs = onSubmitMock.mock.calls[0][0];
      expect(callArgs.budget_limits).toEqual([{ budget_duration: "30d", max_budget: 200 }]);
    });
  });

  it("should omit budget_limits (not clear stored windows) when a window is left incomplete", async () => {
    const onSubmitMock = vi.fn().mockResolvedValue(undefined);
    const keyDataWithWindow = {
      ...MOCK_KEY_DATA,
      budget_limits: [{ budget_duration: "30d", max_budget: 100 }],
    };
    renderWithProviders(
      <KeyEditView
        keyData={keyDataWithWindow}
        onCancel={() => {}}
        onSubmit={onSubmitMock}
        accessToken={"test-token"}
        userID={"test-user"}
        userRole={"admin"}
        premiumUser={false}
      />,
    );

    const maxBudgetInput = await screen.findByPlaceholderText("Max spend ($)");
    await userEvent.clear(maxBudgetInput);

    const submitButton = screen.getByRole("button", { name: /save changes/i });
    await userEvent.click(submitButton);

    await waitFor(() => {
      expect(onSubmitMock).toHaveBeenCalled();
      const callArgs = onSubmitMock.mock.calls[0][0];
      expect(callArgs.budget_limits).toBeUndefined();
    });
  });

  describe("per-model budgets", () => {
    const keyDataWithBudgets = {
      ...MOCK_KEY_DATA,
      model_max_budget: { "gpt-4": { budget_limit: 5, time_period: "30d" } },
    };

    const renderWith = (premiumUser: boolean) => {
      const onSubmit = vi.fn().mockResolvedValue(undefined);
      renderWithProviders(
        <KeyEditView
          keyData={keyDataWithBudgets}
          onCancel={() => {}}
          onSubmit={onSubmit}
          accessToken={"test-token"}
          userID={"test-user"}
          userRole={"admin"}
          premiumUser={premiumUser}
        />,
      );
      return onSubmit;
    };

    // Same hazard as the user edit form: keyData changes while this component
    // stays mounted (the effect re-seeds the form for exactly that reason), and
    // the editor's rows live in state seeded once.
    it("re-seeds the editor when a different key is loaded", async () => {
      const withBudget = (limit: number, token: string) => ({
        ...MOCK_KEY_DATA,
        token,
        model_max_budget: { "gpt-4": { budget_limit: limit, time_period: "1h" } },
      });

      const { rerender } = renderWithProviders(
        <KeyEditView
          keyData={withBudget(5, "tok-a")}
          onCancel={() => {}}
          onSubmit={vi.fn().mockResolvedValue(undefined)}
          accessToken={"test-token"}
          userID={"test-user"}
          userRole={"admin"}
          premiumUser={true}
        />,
      );
      expect(await screen.findByPlaceholderText("Max spend ($)")).toHaveValue(5);

      rerender(
        <KeyEditView
          keyData={withBudget(99, "tok-b")}
          onCancel={() => {}}
          onSubmit={vi.fn().mockResolvedValue(undefined)}
          accessToken={"test-token"}
          userID={"test-user"}
          userRole={"admin"}
          premiumUser={true}
        />,
      );

      expect(await screen.findByPlaceholderText("Max spend ($)")).toHaveValue(99);
    });

    it("should say why the editor is locked when the proxy has no enterprise license", async () => {
      renderWith(false);

      expect(await screen.findByText(MODEL_MAX_BUDGET_PREMIUM_HINT)).toBeInTheDocument();
    });

    it("should leave the editor usable when the proxy has one", async () => {
      renderWith(true);

      expect(await screen.findByText(/Cap spend per model over its own window/)).toBeInTheDocument();
      expect(screen.queryByText(MODEL_MAX_BUDGET_PREMIUM_HINT)).not.toBeInTheDocument();
    });

    // /key/update validates model_max_budget whenever the field is present and
    // rejects it without a license, so re-sending an untouched budget would turn
    // every unrelated edit into a 400.
    it("should leave model_max_budget out of an edit that did not touch it", async () => {
      const onSubmit = renderWith(true);

      await userEvent.click(await screen.findByRole("button", { name: /save changes/i }));

      await waitFor(() => {
        expect(onSubmit).toHaveBeenCalled();
      });
      expect(onSubmit.mock.calls[0][0]).not.toHaveProperty("model_max_budget");
    });
  });

  it("should display 'AI APIs' label for the llm_api key type option", async () => {
    const keyDataWithLlmApiRoutes = {
      ...MOCK_KEY_DATA,
      allowed_routes: ["llm_api_routes"],
    };

    renderWithProviders(
      <KeyEditView
        keyData={keyDataWithLlmApiRoutes}
        onCancel={() => {}}
        onSubmit={async () => {}}
        accessToken={""}
        userID={""}
        userRole={""}
        premiumUser={false}
      />,
    );

    await waitFor(() => {
      expect(screen.getByText("Key Type")).toBeInTheDocument();
    });

    // The selected key type label should show "AI APIs" (not "LLM API")
    await userEvent.click(screen.getByLabelText("Key Type"));

    await waitFor(() => {
      // Verify "AI APIs" appears as an option label
      const optionTexts = visibleOptions().map((el) => el.textContent);
      const hasAIAPIs = optionTexts.some((text) => text?.includes("AI APIs"));
      expect(hasAIAPIs).toBe(true);

      // Verify old "LLM API" label does NOT appear
      const hasLLMAPI = optionTexts.some((text) => text?.includes("LLM API"));
      expect(hasLLMAPI).toBe(false);
    });
  });

  it("should display cancel button during submission", async () => {
    let resolveSubmit: (() => void) | undefined;
    const submitPromise = new Promise<void>((resolve) => {
      resolveSubmit = resolve;
    });
    const onSubmitMock = vi.fn(() => submitPromise);

    renderWithProviders(
      <KeyEditView
        keyData={MOCK_KEY_DATA}
        onCancel={() => {}}
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
    await waitFor(
      () => {
        const cancelButton = screen.getByRole("button", { name: /cancel/i });
        expect(cancelButton).toBeDisabled();
      },
      { timeout: 3000 },
    );

    // Clean up: resolve the promise to allow the form to complete
    if (resolveSubmit) {
      resolveSubmit();
    }
  });

  describe("organization dropdown", () => {
    it("should render the organization dropdown", async () => {
      renderWithProviders(
        <KeyEditView
          keyData={MOCK_KEY_DATA}
          onCancel={() => {}}
          onSubmit={async () => {}}
          accessToken=""
          userID=""
          userRole="Admin"
          premiumUser={false}
        />,
      );

      await waitFor(() => {
        expect(screen.getByText("Organization")).toBeInTheDocument();
      });
    });

    it("should disable the organization dropdown for non-admin users", async () => {
      renderWithProviders(
        <KeyEditView
          keyData={MOCK_KEY_DATA}
          onCancel={() => {}}
          onSubmit={async () => {}}
          accessToken=""
          userID=""
          userRole="Internal User"
          premiumUser={false}
        />,
      );

      await waitFor(() => {
        expect(screen.getByText("Organization")).toBeInTheDocument();
      });

      await userEvent.click(screen.getByLabelText("Organization"));

      expect(screen.queryByText("Engineering")).not.toBeInTheDocument();
    });

    it("should not disable the organization dropdown for admin users", async () => {
      renderWithProviders(
        <KeyEditView
          keyData={MOCK_KEY_DATA}
          onCancel={() => {}}
          onSubmit={async () => {}}
          accessToken=""
          userID=""
          userRole="Admin"
          premiumUser={false}
        />,
      );

      await waitFor(() => {
        expect(screen.getByText("Organization")).toBeInTheDocument();
      });

      await userEvent.click(screen.getByLabelText("Organization"));

      expect(await screen.findByText("Engineering")).toBeInTheDocument();
    });

    it("should initialize organization from keyData", async () => {
      const keyWithOrg = {
        ...MOCK_KEY_DATA,
        organization_id: "org-1",
      };

      renderWithProviders(
        <KeyEditView
          keyData={keyWithOrg}
          onCancel={() => {}}
          onSubmit={async () => {}}
          accessToken=""
          userID=""
          userRole="Admin"
          premiumUser={false}
        />,
      );

      await waitFor(() => {
        expect(screen.getByLabelText("Organization")).toHaveValue("Engineering");
      });
    });
  });

  describe("models dropdown team gating", () => {
    const openModelsDropdown = async () => {
      await userEvent.click(screen.getByLabelText("Models"));
    };

    it("should offer all-proxy-models but not all-team-models for a teamless key", async () => {
      renderWithProviders(
        <KeyEditView
          keyData={MOCK_KEY_DATA}
          onCancel={() => {}}
          onSubmit={async () => {}}
          accessToken="test-token"
          userID="user-123"
          userRole="Admin"
          premiumUser={false}
        />,
      );

      await waitFor(() => {
        expect(screen.getByText("Models", { selector: "label" })).toBeInTheDocument();
      });

      await openModelsDropdown();

      await waitFor(() => {
        expect(screen.getAllByText("gpt-4").length).toBeGreaterThan(0);
      });

      expect(screen.getAllByText("All Proxy Models").length).toBeGreaterThan(0);
      expect(screen.queryAllByText("All Team Models")).toHaveLength(0);
    });

    it("should offer all-team-models but hide all-proxy-models for a team key", async () => {
      const teamKeyData = { ...MOCK_KEY_DATA, team_id: "team-1" };
      const teams = [{ team_id: "team-1", models: ["all-proxy-models", "team-model-1"] }];

      renderWithProviders(
        <KeyEditView
          keyData={teamKeyData}
          teams={teams}
          onCancel={() => {}}
          onSubmit={async () => {}}
          accessToken="test-token"
          userID="user-123"
          userRole="Admin"
          premiumUser={false}
        />,
      );

      await waitFor(() => {
        expect(screen.getByText("Models", { selector: "label" })).toBeInTheDocument();
      });

      await openModelsDropdown();

      await waitFor(() => {
        expect(screen.getAllByText("team-model-1").length).toBeGreaterThan(0);
      });

      expect(screen.getAllByText("All Team Models").length).toBeGreaterThan(0);
      expect(screen.queryAllByText("All Proxy Models")).toHaveLength(0);
      expect(screen.queryAllByText("all-proxy-models")).toHaveLength(0);
    });

    it("should not offer all-team-models for a team key whose team has not loaded yet", async () => {
      const teamKeyData = { ...MOCK_KEY_DATA, team_id: "team-1" };

      renderWithProviders(
        <KeyEditView
          keyData={teamKeyData}
          teams={[]}
          onCancel={() => {}}
          onSubmit={async () => {}}
          accessToken="test-token"
          userID="user-123"
          userRole="Admin"
          premiumUser={false}
        />,
      );

      await waitFor(() => {
        expect(screen.getByText("Models", { selector: "label" })).toBeInTheDocument();
      });

      await openModelsDropdown();

      expect(screen.queryAllByText("All Team Models")).toHaveLength(0);
      expect(screen.queryAllByText("All Proxy Models")).toHaveLength(0);
    });

    it("should not duplicate the all-proxy-models option when the teamless model list already carries the sentinel", async () => {
      vi.mocked(modelAvailableCall).mockResolvedValueOnce({
        data: [{ id: "all-proxy-models" }, { id: "gpt-4" }],
      });

      renderWithProviders(
        <KeyEditView
          keyData={MOCK_KEY_DATA}
          onCancel={() => {}}
          onSubmit={async () => {}}
          accessToken="test-token"
          userID="user-123"
          userRole="Admin"
          premiumUser={false}
        />,
      );

      await waitFor(() => {
        expect(screen.getByText("Models", { selector: "label" })).toBeInTheDocument();
      });

      await openModelsDropdown();

      const proxyOptionLabels = () => visibleOptions().map((option) => option.textContent);

      await waitFor(() => {
        expect(proxyOptionLabels()).toContain("gpt-4");
      });

      const labels = proxyOptionLabels();
      expect(labels.filter((label) => label === "All Proxy Models")).toHaveLength(1);
      expect(labels).not.toContain("all-proxy-models");
    });

    it("should collapse the selection to all-proxy-models when the sentinel is picked alongside a model", async () => {
      const onSubmitMock = vi.fn().mockResolvedValue(undefined);

      renderWithProviders(
        <KeyEditView
          keyData={MOCK_KEY_DATA}
          onCancel={() => {}}
          onSubmit={onSubmitMock}
          accessToken="test-token"
          userID="user-123"
          userRole="Admin"
          premiumUser={false}
        />,
      );

      await waitFor(() => {
        expect(screen.getByText("Models", { selector: "label" })).toBeInTheDocument();
      });

      await openModelsDropdown();

      const clickOption = async (label: string) => {
        const option = await waitFor(() => {
          const match = optionByContent(label);
          expect(match).toBeTruthy();
          return match as HTMLElement;
        });
        fireEvent.click(option);
      };

      await clickOption("gpt-4");
      await clickOption("All Proxy Models");
      await userEvent.keyboard("{Escape}");

      await userEvent.click(screen.getByRole("button", { name: /save changes/i }));

      await waitFor(() => {
        expect(onSubmitMock).toHaveBeenCalled();
      });
      expect(onSubmitMock.mock.calls[0][0].models).toEqual(["all-proxy-models"]);
    });

    it("should disable the individual model options once all-proxy-models is selected", async () => {
      renderWithProviders(
        <KeyEditView
          keyData={MOCK_KEY_DATA}
          onCancel={() => {}}
          onSubmit={async () => {}}
          accessToken="test-token"
          userID="user-123"
          userRole="Admin"
          premiumUser={false}
        />,
      );

      await waitFor(() => {
        expect(screen.getByText("Models", { selector: "label" })).toBeInTheDocument();
      });

      await openModelsDropdown();

      const findOption = (label: string) => optionByContent(label);

      const gpt4Before = await waitFor(() => {
        const match = findOption("gpt-4");
        expect(match).toBeTruthy();
        return match!;
      });
      expect(isOptionDisabled(gpt4Before)).toBe(false);

      fireEvent.click(
        await waitFor(() => {
          const match = findOption("All Proxy Models");
          expect(match).toBeTruthy();
          return match!;
        }),
      );

      await waitFor(() => {
        expect(isOptionDisabled(findOption("gpt-4")!)).toBe(true);
      });
    });
  });

  describe("estimated output tokens", () => {
    const renderEditView = (
      keyData: KeyResponse,
      onSubmit: (values: any) => Promise<void>,
      userRole: string = "Admin",
    ) =>
      renderWithProviders(
        <KeyEditView
          keyData={keyData}
          onCancel={() => {}}
          onSubmit={onSubmit}
          accessToken={"test-token"}
          userID={"test-user"}
          userRole={userRole}
          premiumUser={false}
        />,
      );

    it("refuses to save an invalid per-model estimate, and saves once it is corrected", async () => {
      const onSubmitMock = vi.fn().mockResolvedValue(undefined);
      renderEditView(MOCK_KEY_DATA, onSubmitMock);
      await screen.findByRole("button", { name: /save changes/i });
      const perModel = screen.getByLabelText("Estimated Output Tokens Per Model");

      fireEvent.change(perModel, { target: { value: "not json" } });
      await userEvent.click(screen.getByRole("button", { name: /save changes/i }));

      expect(await screen.findByText(/JSON object of positive integers/)).toBeInTheDocument();
      expect(onSubmitMock).not.toHaveBeenCalled();

      fireEvent.change(perModel, { target: { value: '{"gpt-4": 4096}' } });
      await userEvent.click(screen.getByRole("button", { name: /save changes/i }));

      await waitFor(() => {
        expect(onSubmitMock).toHaveBeenCalled();
      });
    });

    it("refuses to save a fractional estimate, and saves once it is corrected", async () => {
      const onSubmitMock = vi.fn().mockResolvedValue(undefined);
      renderEditView(MOCK_KEY_DATA, onSubmitMock);
      await screen.findByRole("button", { name: /save changes/i });
      const estimate = screen.getByLabelText("Estimated Output Tokens");

      fireEvent.change(estimate, { target: { value: "12.5" } });
      await userEvent.click(screen.getByRole("button", { name: /save changes/i }));

      await waitFor(() => {
        expect(onSubmitMock).not.toHaveBeenCalled();
      });

      fireEvent.change(estimate, { target: { value: "2048" } });
      await userEvent.click(screen.getByRole("button", { name: /save changes/i }));

      await waitFor(() => {
        expect(onSubmitMock).toHaveBeenCalled();
      });
    });

    it("loads the estimates from key metadata and resubmits them unchanged", async () => {
      const onSubmitMock = vi.fn().mockResolvedValue(undefined);
      renderEditView(
        {
          ...MOCK_KEY_DATA,
          metadata: {
            ...MOCK_KEY_DATA.metadata,
            default_estimated_output_tokens: 512,
            default_estimated_output_tokens_per_model: { "gpt-4": 4096 },
          },
        },
        onSubmitMock,
      );

      await waitFor(() => {
        expect(screen.getByLabelText("Estimated Output Tokens")).toHaveValue(512);
      });
      expect(screen.getByLabelText("Estimated Output Tokens Per Model")).toHaveValue('{"gpt-4":4096}');

      await userEvent.click(screen.getByRole("button", { name: /save changes/i }));

      await waitFor(() => {
        expect(onSubmitMock).toHaveBeenCalled();
      });
      const callArgs = onSubmitMock.mock.calls[0][0];
      expect(callArgs.default_estimated_output_tokens).toBe(512);
      expect(callArgs.default_estimated_output_tokens_per_model).toEqual({ "gpt-4": 4096 });
    });

    it("submits edited estimates as a number and a parsed object", async () => {
      const onSubmitMock = vi.fn().mockResolvedValue(undefined);
      renderEditView(MOCK_KEY_DATA, onSubmitMock);

      await waitFor(() => {
        expect(screen.getByLabelText("Estimated Output Tokens")).toBeInTheDocument();
      });

      fireEvent.change(screen.getByLabelText("Estimated Output Tokens"), { target: { value: "2048" } });
      fireEvent.change(screen.getByLabelText("Estimated Output Tokens Per Model"), {
        target: { value: '{"gpt-5": 8192}' },
      });

      await userEvent.click(screen.getByRole("button", { name: /save changes/i }));

      await waitFor(() => {
        expect(onSubmitMock).toHaveBeenCalled();
      });
      const callArgs = onSubmitMock.mock.calls[0][0];
      expect(callArgs.default_estimated_output_tokens).toBe(2048);
      expect(callArgs.default_estimated_output_tokens_per_model).toEqual({ "gpt-5": 8192 });
    });

    it("omits both estimates from the payload when the controls are blank", async () => {
      const onSubmitMock = vi.fn().mockResolvedValue(undefined);
      renderEditView(MOCK_KEY_DATA, onSubmitMock);

      await waitFor(() => {
        expect(screen.getByLabelText("Estimated Output Tokens Per Model")).toHaveValue("");
      });

      await userEvent.click(screen.getByRole("button", { name: /save changes/i }));

      await waitFor(() => {
        expect(onSubmitMock).toHaveBeenCalled();
      });
      const callArgs = onSubmitMock.mock.calls[0][0];
      expect(callArgs).not.toHaveProperty("default_estimated_output_tokens");
      expect(callArgs).not.toHaveProperty("default_estimated_output_tokens_per_model");
    });

    it.each(["Internal User", "Admin Viewer", "org_admin"])(
      "leaves both controls read-only for %s and still resubmits the stored values",
      async (userRole) => {
        const onSubmitMock = vi.fn().mockResolvedValue(undefined);
        renderEditView(
          {
            ...MOCK_KEY_DATA,
            metadata: {
              ...MOCK_KEY_DATA.metadata,
              default_estimated_output_tokens: 512,
              default_estimated_output_tokens_per_model: { "gpt-4": 4096 },
            },
          },
          onSubmitMock,
          userRole,
        );

        await waitFor(() => {
          expect(screen.getByLabelText("Estimated Output Tokens")).toBeDisabled();
        });
        expect(screen.getByLabelText("Estimated Output Tokens Per Model")).toBeDisabled();

        await userEvent.click(screen.getByRole("button", { name: /save changes/i }));

        await waitFor(() => {
          expect(onSubmitMock).toHaveBeenCalled();
        });
        const callArgs = onSubmitMock.mock.calls[0][0];
        expect(callArgs.default_estimated_output_tokens).toBe(512);
        expect(callArgs.default_estimated_output_tokens_per_model).toEqual({ "gpt-4": 4096 });
      },
    );

    it.each(["Admin", "proxy_admin"])("leaves both controls editable for %s", async (userRole) => {
      renderEditView(MOCK_KEY_DATA, vi.fn().mockResolvedValue(undefined), userRole);

      await waitFor(() => {
        expect(screen.getByLabelText("Estimated Output Tokens")).toBeEnabled();
      });
      expect(screen.getByLabelText("Estimated Output Tokens Per Model")).toBeEnabled();
    });
  });

  const UNTOUCHED_SAVE_PAYLOAD = {
    key_alias: "asdasdas",
    models: [],
    max_budget: 0,
    budget_duration: "30d",
    tpm_limit: 10,
    tpm_limit_type: null,
    rpm_limit: 10,
    rpm_limit_type: null,
    throttle_on_budget_exceeded: false,
    enable_prompt_caching: false,
    max_parallel_requests: 10,
    model_tpm_limit: undefined,
    model_rpm_limit: undefined,
    guardrails: undefined,
    disable_global_guardrails: false,
    policies: undefined,
    tags: ["test-tag"],
    prompts: undefined,
    access_group_ids: [],
    allowed_passthrough_routes: undefined,
    vector_stores: [],
    mcp_servers_and_groups: { servers: [], accessGroups: [], toolsets: [] },
    mcp_tool_permissions: {},
    agents_and_groups: { agents: [], accessGroups: [] },
    organization_id: null,
    team_id: null,
    logging_settings: [],
    metadata: "{}",
    duration: "30d",
    token: "test-token-123",
    disabled_callbacks: [],
    auto_rotate: false,
    rotation_interval: undefined,
    tag_rpm_limit: {},
  };

  describe("submit payload contract", () => {
    const renderForPayload = (
      onSubmit: (values: Record<string, unknown>) => Promise<void>,
      keyData: KeyResponse = MOCK_KEY_DATA,
    ) =>
      renderWithProviders(
        <KeyEditView
          keyData={keyData}
          onCancel={() => {}}
          onSubmit={onSubmit}
          accessToken={"test-token"}
          userID={"test-user"}
          userRole={"Admin"}
          premiumUser={true}
        />,
      );

    it("sends exactly the bound form fields on an untouched save, and no server-only key data", async () => {
      const onSubmitMock = vi.fn().mockResolvedValue(undefined);
      renderForPayload(onSubmitMock);
      await screen.findByRole("button", { name: /save changes/i });

      await userEvent.click(screen.getByRole("button", { name: /save changes/i }));

      await waitFor(() => {
        expect(onSubmitMock).toHaveBeenCalled();
      });
      expect(onSubmitMock.mock.calls[0][0]).toStrictEqual(UNTOUCHED_SAVE_PAYLOAD);
    });

    it("drops the policy and prompt keys entirely for a role that cannot see those fields", async () => {
      const onSubmitMock = vi.fn().mockResolvedValue(undefined);
      renderWithProviders(
        <KeyEditView
          keyData={MOCK_KEY_DATA}
          onCancel={() => {}}
          onSubmit={onSubmitMock}
          accessToken={"test-token"}
          userID={"test-user"}
          userRole={"Internal User"}
          premiumUser={true}
        />,
      );
      await screen.findByRole("button", { name: /save changes/i });

      await userEvent.click(screen.getByRole("button", { name: /save changes/i }));

      await waitFor(() => {
        expect(onSubmitMock).toHaveBeenCalled();
      });
      const payload = onSubmitMock.mock.calls[0][0];
      expect(payload).not.toHaveProperty("policies");
      expect(payload).not.toHaveProperty("prompts");
      expect(payload).toHaveProperty("guardrails");
    });

    it("routes the shared lifecycle and rate-limit-type controls into their own payload keys", async () => {
      const onSubmitMock = vi.fn().mockResolvedValue(undefined);
      renderForPayload(onSubmitMock);
      await screen.findByRole("button", { name: /save changes/i });

      const duration = screen.getByPlaceholderText("e.g., 30d");
      await userEvent.clear(duration);
      await userEvent.type(duration, "45d");

      await userEvent.click(screen.getByLabelText(/TPM Rate Limit Type/));
      await userEvent.click(await screen.findByTitle("Guaranteed throughput"));

      await userEvent.click(screen.getByRole("button", { name: /save changes/i }));

      await waitFor(() => {
        expect(onSubmitMock).toHaveBeenCalled();
      });
      const payload = onSubmitMock.mock.calls[0][0];
      expect(payload.duration).toBe("45d");
      expect(payload.tpm_limit_type).toBe("guaranteed_throughput");
      expect(payload.rpm_limit_type).toBeNull();
    });

    it("blanks duration rather than dropping the key when Never Expire is ticked", async () => {
      const onSubmitMock = vi.fn().mockResolvedValue(undefined);
      renderForPayload(onSubmitMock, { ...MOCK_KEY_DATA, expires: "2026-01-01T00:00:00Z" });
      await screen.findByRole("button", { name: /save changes/i });

      await userEvent.click(screen.getByRole("checkbox", { name: /never expire/i }));
      await userEvent.click(screen.getByRole("button", { name: /save changes/i }));

      await waitFor(() => {
        expect(onSubmitMock).toHaveBeenCalled();
      });
      expect(onSubmitMock.mock.calls[0][0]).toHaveProperty("duration", null);
    });

    it("carries a typed value from every free-text and numeric control into the payload", async () => {
      const onSubmitMock = vi.fn().mockResolvedValue(undefined);
      renderForPayload(onSubmitMock);
      await screen.findByRole("button", { name: /save changes/i });

      const retype = async (label: RegExp | string, text: string) => {
        const control = screen.getByLabelText(label);
        await userEvent.clear(control);
        await userEvent.type(control, text);
      };

      await retype("Key Alias", "typed-alias");
      await retype("Max Budget (USD)", "12.5");
      await retype("TPM Limit", "111");
      await retype("RPM Limit", "222");
      await retype("Max Parallel Requests", "3");
      await retype("Model TPM Limit", '{{"gpt-4": 7}');
      await retype("Model RPM Limit", '{{"gpt-4": 8}');
      await retype("Metadata", '{{"typed": true}');

      await userEvent.click(screen.getByRole("button", { name: /save changes/i }));

      await waitFor(() => {
        expect(onSubmitMock).toHaveBeenCalled();
      });
      expect(onSubmitMock.mock.calls[0][0]).toMatchObject({
        key_alias: "typed-alias",
        max_budget: "12.5",
        tpm_limit: "111",
        rpm_limit: "222",
        max_parallel_requests: "3",
        model_tpm_limit: '{"gpt-4": 7}',
        model_rpm_limit: '{"gpt-4": 8}',
        metadata: '{"typed": true}',
      });
    });

    it("carries every toggle driven off its default into the payload", async () => {
      const onSubmitMock = vi.fn().mockResolvedValue(undefined);
      renderForPayload(onSubmitMock);
      await screen.findByRole("button", { name: /save changes/i });

      await userEvent.click(screen.getByRole("switch", { name: /throttle on budget exceeded/i }));
      await userEvent.click(screen.getByRole("switch", { name: /enable prompt caching/i }));
      await userEvent.click(screen.getByRole("switch", { name: /disable global guardrails/i }));

      await userEvent.click(screen.getByRole("button", { name: /save changes/i }));

      await waitFor(() => {
        expect(onSubmitMock).toHaveBeenCalled();
      });
      expect(onSubmitMock.mock.calls[0][0]).toMatchObject({
        throttle_on_budget_exceeded: true,
        enable_prompt_caching: true,
        disable_global_guardrails: true,
      });
    });

    it("carries a tag typed into the tags control into the payload", async () => {
      const onSubmitMock = vi.fn().mockResolvedValue(undefined);
      renderForPayload(onSubmitMock);
      await screen.findByRole("button", { name: /save changes/i });

      await userEvent.type(screen.getByLabelText("Tags"), "typed-tag{Enter}");

      await userEvent.click(screen.getByRole("button", { name: /save changes/i }));

      await waitFor(() => {
        expect(onSubmitMock).toHaveBeenCalled();
      });
      expect(onSubmitMock.mock.calls[0][0].tags).toEqual(["test-tag", "typed-tag"]);
    });

    const pickFromCombobox = async (inputLabel: RegExp | string, optionName: RegExp | string) => {
      await userEvent.click(screen.getByLabelText(inputLabel));
      await userEvent.click(await screen.findByRole("option", { name: optionName }));
      await userEvent.keyboard("{Escape}");
    };

    it("carries a picked guardrail into the payload", async () => {
      const onSubmitMock = vi.fn().mockResolvedValue(undefined);
      renderForPayload(onSubmitMock);
      await screen.findByRole("button", { name: /save changes/i });

      await pickFromCombobox("Select guardrails", "guardrail-1");
      await userEvent.click(screen.getByRole("button", { name: /save changes/i }));

      await waitFor(() => {
        expect(onSubmitMock).toHaveBeenCalled();
      });
      expect(onSubmitMock.mock.calls[0][0].guardrails).toEqual(["guardrail-1"]);
    });

    it("carries a picked policy into the payload", async () => {
      vi.mocked(getPoliciesList).mockResolvedValueOnce({
        policies: [{ policy_name: "policy-1", version_number: 1, version_status: "production" }],
      });
      const onSubmitMock = vi.fn().mockResolvedValue(undefined);
      renderForPayload(onSubmitMock);
      await screen.findByRole("button", { name: /save changes/i });

      await pickFromCombobox(/Select policies/, /policy-1/);
      await userEvent.click(screen.getByRole("button", { name: /save changes/i }));

      await waitFor(() => {
        expect(onSubmitMock).toHaveBeenCalled();
      });
      expect(onSubmitMock.mock.calls[0][0].policies).toEqual(["policy-1"]);
    });

    it("carries a typed prompt into the payload", async () => {
      const onSubmitMock = vi.fn().mockResolvedValue(undefined);
      renderForPayload(onSubmitMock);
      await screen.findByRole("button", { name: /save changes/i });

      await userEvent.type(screen.getByLabelText("Prompts"), "prompt-1{Enter}");
      await userEvent.click(screen.getByRole("button", { name: /save changes/i }));

      await waitFor(() => {
        expect(onSubmitMock).toHaveBeenCalled();
      });
      expect(onSubmitMock.mock.calls[0][0].prompts).toEqual(["prompt-1"]);
    });

    it("carries the RPM rate limit type into its own payload key", async () => {
      const onSubmitMock = vi.fn().mockResolvedValue(undefined);
      renderForPayload(onSubmitMock);
      await screen.findByRole("button", { name: /save changes/i });

      await userEvent.click(screen.getByLabelText(/RPM Rate Limit Type/));
      await userEvent.click(await screen.findByTitle("Guaranteed throughput"));

      await userEvent.click(screen.getByRole("button", { name: /save changes/i }));

      await waitFor(() => {
        expect(onSubmitMock).toHaveBeenCalled();
      });
      const payload = onSubmitMock.mock.calls[0][0];
      expect(payload.rpm_limit_type).toBe("guaranteed_throughput");
      expect(payload.tpm_limit_type).toBeNull();
    });

    it("carries a picked vector store into the payload", async () => {
      vi.mocked(vectorStoreListCall).mockResolvedValueOnce({
        data: [{ vector_store_id: "vs-1", vector_store_name: "VS One" }],
      });
      const onSubmitMock = vi.fn().mockResolvedValue(undefined);
      renderForPayload(onSubmitMock);
      await screen.findByRole("button", { name: /save changes/i });

      await pickFromCombobox("Select vector stores", /VS One/);
      await userEvent.click(screen.getByRole("button", { name: /save changes/i }));

      await waitFor(() => {
        expect(onSubmitMock).toHaveBeenCalled();
      });
      expect(onSubmitMock.mock.calls[0][0].vector_stores).toEqual(["vs-1"]);
    });

    it("carries a picked pass through route into the payload", async () => {
      vi.mocked(getPassThroughEndpointsCall).mockResolvedValueOnce({
        endpoints: [{ path: "/bria", methods: ["POST"] }],
      });
      const onSubmitMock = vi.fn().mockResolvedValue(undefined);
      renderForPayload(onSubmitMock);
      await screen.findByRole("button", { name: /save changes/i });

      await pickFromCombobox(/allowed pass through routes/, /\/bria/);
      await userEvent.click(screen.getByRole("button", { name: /save changes/i }));

      await waitFor(() => {
        expect(onSubmitMock).toHaveBeenCalled();
      });
      expect(onSubmitMock.mock.calls[0][0].allowed_passthrough_routes).toEqual(["/bria"]);
    });

    it("carries a picked team into the payload", async () => {
      const onSubmitMock = vi.fn().mockResolvedValue(undefined);
      renderWithProviders(
        <KeyEditView
          keyData={MOCK_KEY_DATA}
          teams={[{ team_id: "team-9", team_alias: "Team Nine" }]}
          onCancel={() => {}}
          onSubmit={onSubmitMock}
          accessToken={"test-token"}
          userID={"test-user"}
          userRole={"Admin"}
          premiumUser={true}
        />,
      );
      await screen.findByRole("button", { name: /save changes/i });

      await userEvent.click(screen.getByLabelText("Team ID"));
      await userEvent.click(await screen.findByRole("option", { name: /Team Nine/ }));

      await userEvent.click(screen.getByRole("button", { name: /save changes/i }));

      await waitFor(() => {
        expect(onSubmitMock).toHaveBeenCalled();
      });
      expect(onSubmitMock.mock.calls[0][0].team_id).toBe("team-9");
    });

    it("carries a picked MCP server into the payload", async () => {
      const onSubmitMock = vi.fn().mockResolvedValue(undefined);
      renderForPayload(onSubmitMock);
      await screen.findByRole("button", { name: /save changes/i });

      await userEvent.click(screen.getByRole("button", { name: "pick mcp server" }));
      await userEvent.click(screen.getByRole("button", { name: /save changes/i }));

      await waitFor(() => {
        expect(onSubmitMock).toHaveBeenCalled();
      });
      expect(onSubmitMock.mock.calls[0][0].mcp_servers_and_groups.servers).toEqual(["mcp-1"]);
    });

    it("carries a picked agent into the payload", async () => {
      const onSubmitMock = vi.fn().mockResolvedValue(undefined);
      renderForPayload(onSubmitMock);
      await screen.findByRole("button", { name: /save changes/i });

      await userEvent.click(screen.getByRole("button", { name: "pick agent" }));
      await userEvent.click(screen.getByRole("button", { name: /save changes/i }));

      await waitFor(() => {
        expect(onSubmitMock).toHaveBeenCalled();
      });
      expect(onSubmitMock.mock.calls[0][0].agents_and_groups.agents).toEqual(["agent-1"]);
    });

    it("carries an added logging integration into the payload", async () => {
      const onSubmitMock = vi.fn().mockResolvedValue(undefined);
      renderForPayload(onSubmitMock);
      await screen.findByRole("button", { name: /save changes/i });

      await userEvent.click(screen.getByRole("button", { name: /add integration/i }));
      await userEvent.click(screen.getByRole("button", { name: /save changes/i }));

      await waitFor(() => {
        expect(onSubmitMock).toHaveBeenCalled();
      });
      expect(onSubmitMock.mock.calls[0][0].logging_settings).toEqual([
        { callback_name: "", callback_type: "success", callback_vars: {} },
      ]);
    });

    it("resends stored budget fallbacks on an untouched save", async () => {
      const onSubmitMock = vi.fn().mockResolvedValue(undefined);
      renderForPayload(onSubmitMock, {
        ...MOCK_KEY_DATA,
        budget_fallbacks: { "gpt-4": ["gpt-4o-mini"] },
      } as KeyResponse);
      await screen.findByRole("button", { name: /save changes/i });

      await userEvent.click(screen.getByRole("button", { name: /save changes/i }));

      await waitFor(() => {
        expect(onSubmitMock).toHaveBeenCalled();
      });
      expect(onSubmitMock.mock.calls[0][0]).toHaveProperty("budget_fallbacks", { "gpt-4": ["gpt-4o-mini"] });
    });

    it("omits budget fallbacks entirely for a key that has none", async () => {
      const onSubmitMock = vi.fn().mockResolvedValue(undefined);
      renderForPayload(onSubmitMock);
      await screen.findByRole("button", { name: /save changes/i });

      await userEvent.click(screen.getByRole("button", { name: /save changes/i }));

      await waitFor(() => {
        expect(onSubmitMock).toHaveBeenCalled();
      });
      expect(onSubmitMock.mock.calls[0][0]).not.toHaveProperty("budget_fallbacks");
    });

    it("resends the stored per-tag rpm limits on an untouched save", async () => {
      const onSubmitMock = vi.fn().mockResolvedValue(undefined);
      renderForPayload(onSubmitMock, {
        ...MOCK_KEY_DATA,
        metadata: { ...MOCK_KEY_DATA.metadata, tag_rpm_limit: { "test-tag": 7 } },
      } as KeyResponse);
      await screen.findByRole("button", { name: /save changes/i });

      await userEvent.click(screen.getByRole("button", { name: /save changes/i }));

      await waitFor(() => {
        expect(onSubmitMock).toHaveBeenCalled();
      });
      expect(onSubmitMock.mock.calls[0][0]).toHaveProperty("tag_rpm_limit", { "test-tag": 7 });
    });
  });
});
