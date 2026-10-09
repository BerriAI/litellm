import {
  chooseSelectOption,
  fireEvent,
  renderHook,
  screen,
  waitFor,
  within,
  renderWithProviders,
} from "../../../tests/test-utils";
import userEvent, { PointerEventsCheckLevel } from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import type { Team } from "../key_team_helpers/key_list";
import { credentialCreateCall, modelCreateCall, type CredentialItem } from "../networking";
import { Providers } from "../provider_info_helpers";
import { projectMountedValues, useMountRegistry, type MountedFormValues } from "../common_components/MountedFormField";
import { useForm } from "react-hook-form";
import AddModelForm from "./AddModelForm";
import { handleAddModelSubmit } from "./handle_add_model_submit";
import { toast } from "@/lib/toast";

vi.mock("../molecules/models/ProviderLogo", () => ({
  ProviderLogo: ({ provider, className }: { provider: string; className?: string }) => (
    <div className={className} data-testid={`provider-logo-${provider}`}>
      {provider}
    </div>
  ),
}));

vi.mock("../networking", async () => {
  const actual = await vi.importActual("../networking");
  return {
    ...actual,
    getGuardrailsList: vi.fn().mockResolvedValue({
      guardrails: [{ guardrail_name: "test-guardrail-1" }, { guardrail_name: "test-guardrail-2" }],
    }),
    tagListCall: vi.fn().mockResolvedValue({}),
    modelAvailableCall: vi.fn().mockResolvedValue({
      data: [{ id: "model-group-1" }, { id: "model-group-2" }],
    }),
    modelHubCall: vi.fn().mockResolvedValue({
      data: [
        { model_group: "gpt-4", mode: "chat" },
        { model_group: "gpt-3.5-turbo", mode: "chat" },
      ],
    }),
    testConnectionRequest: vi.fn().mockResolvedValue({ status: "success" }),
    credentialCreateCall: vi.fn().mockResolvedValue({ success: true }),
    modelCreateCall: vi.fn().mockResolvedValue({ success: true }),
    getProviderCreateMetadata: vi.fn().mockResolvedValue([
      {
        provider: "OpenAI",
        provider_display_name: "OpenAI",
        litellm_provider: "openai",
        default_model_placeholder: "gpt-3.5-turbo",
        credential_fields: [],
      },
      {
        provider: "MICROSOFT_365_COPILOT",
        provider_display_name: "Microsoft 365 Copilot",
        litellm_provider: "microsoft_365_copilot",
        default_model_placeholder: "microsoft_365_copilot/chat",
        credential_fields: [
          { key: "token_exchange_endpoint", label: "Token Endpoint URL", field_type: "text" },
          {
            key: "token_exchange_profile",
            label: "Exchange Grant",
            field_type: "select",
            options: ["jwt_bearer_obo", "rfc8693"],
            default_value: "jwt_bearer_obo",
          },
          { key: "client_id", label: "Client ID", field_type: "text" },
          { key: "client_secret", label: "Client Secret", field_type: "password" },
          { key: "token_exchange_scope", label: "Scope", field_type: "text" },
          { key: "token_exchange_audience", label: "Audience", field_type: "text" },
          { key: "api_key", label: "Delegated Access Token", field_type: "password" },
        ],
      },
      {
        provider: "GITHUB_COPILOT",
        provider_display_name: "Github Copilot",
        litellm_provider: "github_copilot",
        default_model_placeholder: "github_copilot/chat",
        credential_fields: [
          { key: "api_base", label: "API Base", field_type: "text" },
          { key: "api_key", label: "API Key", field_type: "password" },
        ],
      },
    ]),
  };
});

vi.mock("@/app/(dashboard)/hooks/providers/useProviderFields", () => ({
  useProviderFields: vi.fn().mockReturnValue({
    data: [
      {
        provider: "OpenAI",
        provider_display_name: "OpenAI",
        litellm_provider: "openai",
        default_model_placeholder: "gpt-3.5-turbo",
        credential_fields: [],
      },
      {
        provider: "MICROSOFT_365_COPILOT",
        provider_display_name: "Microsoft 365 Copilot",
        litellm_provider: "microsoft_365_copilot",
        default_model_placeholder: "microsoft_365_copilot/chat",
        credential_fields: [
          { key: "token_exchange_endpoint", label: "Token Endpoint URL", field_type: "text" },
          {
            key: "token_exchange_profile",
            label: "Exchange Grant",
            field_type: "select",
            options: ["jwt_bearer_obo", "rfc8693"],
            default_value: "jwt_bearer_obo",
          },
          { key: "client_id", label: "Client ID", field_type: "text" },
          { key: "client_secret", label: "Client Secret", field_type: "password" },
          { key: "token_exchange_scope", label: "Scope", field_type: "text" },
          { key: "token_exchange_audience", label: "Audience", field_type: "text" },
          { key: "api_key", label: "Delegated Access Token", field_type: "password" },
        ],
      },
      {
        provider: "GITHUB_COPILOT",
        provider_display_name: "Github Copilot",
        litellm_provider: "github_copilot",
        default_model_placeholder: "github_copilot/chat",
        credential_fields: [
          { key: "api_base", label: "API Base", field_type: "text" },
          { key: "api_key", label: "API Key", field_type: "password" },
        ],
      },
    ],
    isLoading: false,
    error: null,
  }),
}));

vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({
  default: vi.fn(),
}));

vi.mock("@/app/(dashboard)/hooks/teams/useTeams", () => ({
  useInfiniteTeams: () => ({
    data: {
      pages: [
        {
          teams: [{ team_id: "team-1", team_alias: "Test Team", organization_id: "org-1" }],
          total: 1,
          page: 1,
          page_size: 20,
          total_pages: 1,
        },
      ],
    },
    fetchNextPage: vi.fn(),
    hasNextPage: false,
    isFetchingNextPage: false,
    isLoading: false,
  }),
}));

vi.mock("@/app/(dashboard)/hooks/guardrails/useGuardrails", () => ({
  useGuardrails: vi.fn().mockReturnValue({
    data: {
      guardrails: [{ guardrail_name: "test-guardrail" }],
      globalGuardrailNames: new Set<string>(),
      optionalGuardrailNames: new Set<string>(["test-guardrail"]),
    },
    isLoading: false,
    error: null,
  }),
}));

vi.mock("@/app/(dashboard)/hooks/tags/useTags", () => ({
  useTags: vi.fn().mockReturnValue({
    data: { tag1: ["model1", "model2"] },
    isLoading: false,
    error: null,
  }),
}));

const mockAuthorizedUser = (userRole: string, userId: string, premiumUser: boolean) => ({
  token: "test-token",
  accessToken: "test-access-token",
  userId,
  userEmail: "test@example.com",
  userRole,
  premiumUser,
  disabledPersonalKeyCreation: false,
  showSSOBanner: false,
});

const testTeam: Team = {
  team_id: "team-1",
  team_alias: "Test Team",
  models: ["gpt-4"],
  max_budget: 100,
  budget_duration: "monthly",
  tpm_limit: null,
  rpm_limit: null,
  organization_id: "org-1",
  created_at: "2024-01-01T00:00:00Z",
  keys: [],
  members_with_roles: [],
};

const createTestProps = (userRole = "proxy_admin", userId = "user-1", isTeamAdmin = false) => {
  const { result } = renderHook(() => {
    const form = useForm<MountedFormValues>({ mode: "onChange" });
    const registry = useMountRegistry();
    return { form, registry };
  });
  const { form, registry } = result.current;

  const teams = [
    {
      ...testTeam,
      members_with_roles: isTeamAdmin ? [{ user_id: userId, role: "admin" }] : [],
    },
  ];

  const credentials: CredentialItem[] = [
    {
      credential_name: "test-credential",
      display_name: "Prod OpenAI",
      credential_values: {},
      credential_info: {
        custom_llm_provider: "openai",
        description: "Test credential",
      },
    },
  ];

  return {
    form,
    registry,
    mountedValues: () => projectMountedValues(registry, form.getValues),
    handleOk: vi.fn().mockResolvedValue(true),
    setSelectedProvider: vi.fn(),
    getPlaceholder: vi.fn((provider: string) => `Enter ${provider} model name`),
    setShowAdvancedSettings: vi.fn(),
    selectedProvider: Providers.OpenAI,
    providerModels: ["gpt-4", "gpt-3.5-turbo"],
    showAdvancedSettings: false,
    teams,
    credentials,
    userRole,
    userId,
  };
};

describe("AddModelForm", () => {
  it("should render", async () => {
    const mockUseAuthorized = vi.mocked(await import("@/app/(dashboard)/hooks/useAuthorized"));
    mockUseAuthorized.default.mockReturnValue(mockAuthorizedUser("proxy_admin", "user-1", true));

    const props = createTestProps();

    renderWithProviders(<AddModelForm {...props} />);

    expect(await screen.findByRole("heading", { name: "Add Model" })).toBeInTheDocument();
  });

  it("should show proxy admin only (not team admin) - should not see Select Team dropdown unless switch is toggled", async () => {
    const mockUseAuthorized = vi.mocked(await import("@/app/(dashboard)/hooks/useAuthorized"));
    mockUseAuthorized.default.mockReturnValue(mockAuthorizedUser("proxy_admin", "user-1", true));

    const props = createTestProps("proxy_admin", "user-1", false);

    renderWithProviders(<AddModelForm {...props} />);

    await screen.findByText("Provider");

    expect(screen.queryByText("Team Selection Required")).not.toBeInTheDocument();
    expect(screen.queryByText("Select Team")).not.toBeInTheDocument();

    const teamSwitch = screen.getByRole("switch");
    expect(teamSwitch).toBeInTheDocument();

    expect(screen.queryByText("Select Team")).not.toBeInTheDocument();

    await userEvent.click(teamSwitch);

    expect(await screen.findByText("Select Team")).toBeInTheDocument();
  });

  it("should show proxy admin who is also team admin - should not see Select Team dropdown unless switch is toggled", async () => {
    const mockUseAuthorized = vi.mocked(await import("@/app/(dashboard)/hooks/useAuthorized"));
    mockUseAuthorized.default.mockReturnValue(mockAuthorizedUser("proxy_admin", "user-1", true));

    const props = createTestProps("proxy_admin", "user-1", true);

    renderWithProviders(<AddModelForm {...props} />);

    await screen.findByText("Provider");

    expect(screen.queryByText("Team Selection Required")).not.toBeInTheDocument();
    expect(screen.queryByText("Select Team")).not.toBeInTheDocument();

    const teamSwitch = screen.getByRole("switch");
    expect(teamSwitch).toBeInTheDocument();

    expect(screen.queryByText("Select Team")).not.toBeInTheDocument();

    await userEvent.click(teamSwitch);

    expect(await screen.findByText("Select Team")).toBeInTheDocument();
  });

  it("should show team admin (not proxy admin) - should see alert and team select, must select team before seeing remaining fields", async () => {
    const mockUseAuthorized = vi.mocked(await import("@/app/(dashboard)/hooks/useAuthorized"));
    mockUseAuthorized.default.mockReturnValue(mockAuthorizedUser("team_member", "user-1", true));

    const props = createTestProps("team_member", "user-1", true);

    renderWithProviders(<AddModelForm {...props} />);

    await screen.findByRole("heading", { name: "Add Model" });

    expect(screen.getByText("Team Selection Required")).toBeInTheDocument();

    expect(screen.getByText("Select Team")).toBeInTheDocument();

    expect(screen.queryByText("Provider")).not.toBeInTheDocument();

    const teamSelect = screen.getByRole("combobox");
    await userEvent.click(teamSelect);
    await userEvent.click(screen.getByText("Test Team"));

    await waitFor(() => {
      expect(screen.getByText("Provider")).toBeInTheDocument();
    });
  });

  it("should show team admin (not proxy admin) - should not see team-BYOK switch", async () => {
    const mockUseAuthorized = vi.mocked(await import("@/app/(dashboard)/hooks/useAuthorized"));
    mockUseAuthorized.default.mockReturnValue(mockAuthorizedUser("team_member", "user-1", true));

    const props = createTestProps("team_member", "user-1", true);

    renderWithProviders(<AddModelForm {...props} />);

    await screen.findByText("Select Team");

    const teamSelect = screen.getByRole("combobox");
    await userEvent.click(teamSelect);
    await userEvent.click(screen.getByText("Test Team"));

    await waitFor(() => {
      expect(screen.getByText("Provider")).toBeInTheDocument();
    });

    expect(screen.queryByRole("switch")).not.toBeInTheDocument();
  });

  it("should handle non-admin, non-team-admin users - should not see team selection or switch", async () => {
    const mockUseAuthorized = vi.mocked(await import("@/app/(dashboard)/hooks/useAuthorized"));
    mockUseAuthorized.default.mockReturnValue(mockAuthorizedUser("user", "user-1", false));

    const props = createTestProps("user", "user-1", false);

    renderWithProviders(<AddModelForm {...props} />);

    await screen.findByRole("heading", { name: "Add Model" });

    expect(screen.queryByText("Team Selection Required")).not.toBeInTheDocument();

    expect(screen.queryByText("Select Team")).not.toBeInTheDocument();

    expect(screen.queryByText("Provider")).not.toBeInTheDocument();

    expect(screen.queryByRole("switch")).not.toBeInTheDocument();
  });

  describe("the existing-credentials picker", () => {
    const openPicker = async () => {
      const mockUseAuthorized = vi.mocked(await import("@/app/(dashboard)/hooks/useAuthorized"));
      mockUseAuthorized.default.mockReturnValue(mockAuthorizedUser("proxy_admin", "user-1", true));
      const user = userEvent.setup({ pointerEventsCheck: PointerEventsCheckLevel.Never });
      const props = createTestProps();
      renderWithProviders(<AddModelForm {...props} />);
      const input = await screen.findByPlaceholderText("Select or search for existing credentials");
      await user.click(input);
      return { user, input, form: props.form };
    };

    it("shows each credential's display name and credential name together", async () => {
      await openPicker();

      const option = await screen.findByRole("option", { name: /test-credential/ });
      expect(option).toHaveTextContent("test-credential");
      expect(option).toHaveTextContent("Prod OpenAI");
    });

    it("filters by the display name, not only the credential name", async () => {
      const { user, input } = await openPicker();

      await user.type(input, "prod open");

      expect(await screen.findByRole("option", { name: /test-credential/ })).toBeInTheDocument();
    });

    it("selecting by display name sets litellm_credential_name to the credential name", async () => {
      const { user, input, form } = await openPicker();
      expect(screen.getByText("OR")).toBeInTheDocument();

      await user.type(input, "prod open");
      await user.click(await screen.findByRole("option", { name: /test-credential/ }));

      expect(input).toHaveValue("Prod OpenAI");
      expect(form.getValues("litellm_credential_name")).toBe("test-credential");
      expect(screen.queryByText("OR")).not.toBeInTheDocument();
    });
  });

  it("should display the provider field and the Test Connect / Add Model buttons", async () => {
    const mockUseAuthorized = vi.mocked(await import("@/app/(dashboard)/hooks/useAuthorized"));
    mockUseAuthorized.default.mockReturnValue(mockAuthorizedUser("proxy_admin", "user-1", true));

    const props = createTestProps();

    renderWithProviders(<AddModelForm {...props} />);

    expect(await screen.findByText("Provider")).toBeInTheDocument();
    expect((await screen.findAllByRole("button", { name: "Test Connect" })).length).toBeGreaterThan(0);
    expect(await screen.findByRole("button", { name: "Add Model" })).toBeInTheDocument();
  });

  it("offers the Evaluation decisions mode", async () => {
    const mockUseAuthorized = vi.mocked(await import("@/app/(dashboard)/hooks/useAuthorized"));
    mockUseAuthorized.default.mockReturnValue(mockAuthorizedUser("proxy_admin", "user-1", true));

    renderWithProviders(<AddModelForm {...createTestProps()} />);

    await screen.findByText("Provider");
    await userEvent.click(screen.getByRole("combobox", { name: "Mode" }));

    expect(await screen.findByRole("option", { name: "Evaluation - /v1/decisions", exact: true })).toBeInTheDocument();
  });

  it("shows only the Close button in the connection test dialog footer", async () => {
    const mockUseAuthorized = vi.mocked(await import("@/app/(dashboard)/hooks/useAuthorized"));
    mockUseAuthorized.default.mockReturnValue(mockAuthorizedUser("proxy_admin", "user-1", true));

    renderWithProviders(<AddModelForm {...createTestProps()} />);

    await userEvent.click(await screen.findByTestId("test-connect-btn"));

    const dialog = await screen.findByRole("dialog");
    const footer = dialog.querySelector('[data-slot="dialog-footer"]');
    expect(footer).not.toBeNull();
    expect(footer!.textContent?.trim()).toBe("Close");
  });

  describe("the enterprise gate on the Team-BYOK switch", () => {
    const renderForm = async (premiumUser: boolean) => {
      const mockUseAuthorized = vi.mocked(await import("@/app/(dashboard)/hooks/useAuthorized"));
      mockUseAuthorized.default.mockReturnValue(mockAuthorizedUser("proxy_admin", "user-1", premiumUser));
      renderWithProviders(<AddModelForm {...createTestProps()} />);
      return screen.findByRole("switch", { name: "Team-BYOK Model" });
    };

    it("explains the gate on hover even though the switch it sits on is disabled", async () => {
      const user = userEvent.setup({ pointerEventsCheck: PointerEventsCheckLevel.Never });
      const teamOnlySwitch = await renderForm(false);
      expect(teamOnlySwitch).toHaveAttribute("aria-disabled", "true");

      await user.hover(teamOnlySwitch);

      expect(await screen.findByText(/enterprise-only feature/)).toBeInTheDocument();
    });

    it("says nothing on hover once the user is premium", async () => {
      const user = userEvent.setup();
      const teamOnlySwitch = await renderForm(true);
      expect(teamOnlySwitch).not.toHaveAttribute("aria-disabled", "true");

      await user.hover(teamOnlySwitch);

      expect(screen.queryByText(/enterprise-only feature/)).not.toBeInTheDocument();
    });
  });

  describe("workload identity federation entry point", () => {
    const renderAsRole = async (userRole: string, selectedProvider: Providers) => {
      const mockUseAuthorized = vi.mocked(await import("@/app/(dashboard)/hooks/useAuthorized"));
      mockUseAuthorized.default.mockReturnValue(mockAuthorizedUser(userRole, "user-1", true));
      const props = { ...createTestProps(userRole, "user-1", false), selectedProvider };
      renderWithProviders(<AddModelForm {...props} />);
      await screen.findByText("Existing Credentials");
      return props;
    };

    const fill = (label: string | RegExp, value: string) =>
      fireEvent.change(screen.getByLabelText(label), { target: { value } });

    it("saves the federation fields as a credential and attaches it to the model", async () => {
      const user = userEvent.setup();
      const props = await renderAsRole("proxy_admin", Providers.Anthropic);

      await user.click(screen.getByRole("button", { name: "Use workload identity federation" }));
      expect(await screen.findByRole("combobox", { name: /^Authentication:/ })).toHaveTextContent(
        "Workload identity federation",
      );
      fill("Credential Name:", "anthropic-federated");
      fill(/Federation Rule ID/, "fdrl_new");
      fill(/Organization ID/, "org-new");
      fill(/Identity Token File/, "/var/run/secrets/anthropic/token");
      await user.click(screen.getByRole("button", { name: "Add Credential" }));

      await waitFor(() => {
        expect(credentialCreateCall).toHaveBeenCalledWith("test-access-token", {
          credential_name: "anthropic-federated",
          credential_values: {
            anthropic_federation_rule_id: "fdrl_new",
            anthropic_organization_id: "org-new",
            anthropic_identity_token_file: "/var/run/secrets/anthropic/token",
          },
          credential_info: { custom_llm_provider: Providers.Anthropic },
        });
      });
      await waitFor(() => {
        expect(props.form.getValues("litellm_credential_name")).toBe("anthropic-federated");
      });
      expect(props.mountedValues()).not.toHaveProperty("anthropic_federation_rule_id");
    });

    it("keeps the credential dialog on the model's provider", async () => {
      const user = userEvent.setup();
      await renderAsRole("proxy_admin", Providers.Anthropic);

      await user.click(screen.getByRole("button", { name: "Use workload identity federation" }));

      const providerSelect = within(await screen.findByRole("dialog")).getByPlaceholderText("Select a provider");
      expect(providerSelect).toHaveValue("Anthropic");
      expect(providerSelect).toBeDisabled();
    });

    it("saves an OpenAI federated credential and attaches it to the model", async () => {
      const user = userEvent.setup();
      const props = await renderAsRole("proxy_admin", Providers.OpenAI);

      await user.click(screen.getByRole("button", { name: "Use workload identity federation" }));
      expect(await screen.findByRole("combobox", { name: /^Authentication:/ })).toHaveTextContent(
        "Workload identity federation",
      );
      fill("Credential Name:", "openai-federated");
      fill(/Service Account ID/, "svc_new");
      await user.click(screen.getByRole("button", { name: "Add Credential" }));

      await waitFor(() => {
        expect(credentialCreateCall).toHaveBeenCalledWith("test-access-token", {
          credential_name: "openai-federated",
          credential_values: { openai_service_account_id: "svc_new" },
          credential_info: { custom_llm_provider: Providers.OpenAI },
        });
      });
      await waitFor(() => {
        expect(props.form.getValues("litellm_credential_name")).toBe("openai-federated");
      });
    });

    it("is not offered for a provider without federation support", async () => {
      await renderAsRole("proxy_admin", Providers.OpenAI_Compatible);

      expect(screen.queryByRole("button", { name: "Use workload identity federation" })).not.toBeInTheDocument();
    });

    it("is not offered to an admin who is not a proxy admin", async () => {
      await renderAsRole("org_admin", Providers.Anthropic);

      expect(screen.queryByRole("button", { name: "Use workload identity federation" })).not.toBeInTheDocument();
    });
  });

  describe("credential-only provider auth types", () => {
    const renderAsAdmin = async (selectedProvider = Providers.MICROSOFT_365_COPILOT) => {
      const mockUseAuthorized = vi.mocked(await import("@/app/(dashboard)/hooks/useAuthorized"));
      mockUseAuthorized.default.mockReturnValue(mockAuthorizedUser("proxy_admin", "user-1", true));
      const props = createTestProps();
      props.selectedProvider = selectedProvider;
      renderWithProviders(<AddModelForm {...props} />);
      await screen.findByText("Existing Credentials");
      return props;
    };

    const submitModel = async (props: ReturnType<typeof createTestProps>) => {
      const credentialName = props.form.getValues("litellm_credential_name");
      const apiKey = props.form.getValues("api_key");
      const modelValues = {
        ...(credentialName ? { litellm_credential_name: credentialName } : {}),
        ...(apiKey ? { api_key: apiKey } : {}),
        custom_llm_provider: "MICROSOFT_365_COPILOT",
        model_mappings: [
          {
            public_name: "copilot-model",
            litellm_model: "microsoft_365_copilot/chat",
          },
        ],
      };
      await handleAddModelSubmit(modelValues, "test-access-token", { resetFields: vi.fn() });
    };

    it("creates and attaches an OAuth exchange credential without mounting its fields on the model", async () => {
      const user = userEvent.setup();
      vi.mocked(credentialCreateCall).mockClear();
      vi.mocked(modelCreateCall).mockClear();
      const props = await renderAsAdmin();

      await user.click(screen.getByRole("button", { name: "Create credential" }));
      const dialog = await screen.findByRole("dialog");
      expect(within(dialog).getByPlaceholderText("Select a provider")).toHaveValue(Providers.MICROSOFT_365_COPILOT);
      expect(within(dialog).getByRole("combobox", { name: "Auth Type:" })).toHaveTextContent(
        "OAuth token exchange (on-behalf-of)",
      );
      await chooseSelectOption(
        user,
        within(dialog).getByRole("combobox", { name: "Exchange Grant" }),
        "jwt_bearer_obo",
      );
      fireEvent.change(within(dialog).getByLabelText("Credential Name:"), { target: { value: "m365-obo" } });
      fireEvent.change(within(dialog).getByLabelText("Token Endpoint URL"), {
        target: { value: "https://login.example.test/token" },
      });
      fireEvent.change(within(dialog).getByLabelText("Client ID"), { target: { value: "client-id" } });
      fireEvent.change(within(dialog).getByLabelText("Client Secret"), { target: { value: "client-secret" } });
      fireEvent.change(within(dialog).getByLabelText("Scope"), { target: { value: "scope" } });
      fireEvent.change(within(dialog).getByLabelText("Audience"), { target: { value: "audience" } });
      await user.click(within(dialog).getByRole("button", { name: "Add Credential" }));

      await waitFor(() =>
        expect(credentialCreateCall).toHaveBeenCalledWith("test-access-token", {
          credential_name: "m365-obo",
          credential_values: {
            token_exchange_endpoint: "https://login.example.test/token",
            token_exchange_profile: "jwt_bearer_obo",
            client_id: "client-id",
            client_secret: "client-secret",
            token_exchange_scope: "scope",
            token_exchange_audience: "audience",
          },
          credential_info: { custom_llm_provider: Providers.MICROSOFT_365_COPILOT },
        }),
      );
      expect(props.form.getValues("litellm_credential_name")).toBe("m365-obo");
      expect(props.mountedValues()).not.toHaveProperty("token_exchange_endpoint");

      props.handleOk.mockImplementation(async () => {
        await submitModel(props);
        return true;
      });
      await user.click(screen.getByRole("button", { name: "Add Model" }));
      expect(props.handleOk).toHaveBeenCalledOnce();
      await waitFor(() => expect(modelCreateCall).toHaveBeenCalledOnce());
      const modelPayload = vi.mocked(modelCreateCall).mock.calls[0][1];
      expect(modelPayload.litellm_params).toMatchObject({ litellm_credential_name: "m365-obo" });
      expect(Object.keys(modelPayload.litellm_params).some((key) => key.startsWith("token_exchange_"))).toBe(false);
    });

    it("blocks model creation when OAuth exchange has no credential", async () => {
      const props = await renderAsAdmin();
      const errorToast = vi.spyOn(toast, "error");

      await userEvent.click(screen.getByRole("button", { name: "Add Model" }));

      expect(props.handleOk).not.toHaveBeenCalled();
      expect(errorToast).toHaveBeenCalledWith("Create or select a credential for this auth type");
      errorToast.mockRestore();
    });

    it("keeps static delegated tokens inline in the model request", async () => {
      const user = userEvent.setup();
      vi.mocked(modelCreateCall).mockClear();
      const props = await renderAsAdmin();
      await chooseSelectOption(
        user,
        screen.getByRole("combobox", { name: "Auth Type:" }),
        "Static delegated access token",
      );
      await waitFor(() =>
        expect(screen.getByRole("combobox", { name: "Auth Type:" })).toHaveTextContent("Static delegated access token"),
      );
      fireEvent.change(screen.getByLabelText("Delegated Access Token"), { target: { value: "delegated-token" } });
      props.handleOk.mockImplementation(async () => {
        await submitModel(props);
        return true;
      });

      await user.click(screen.getByRole("button", { name: "Add Model" }));
      expect(props.handleOk).toHaveBeenCalledOnce();
      await waitFor(() => expect(modelCreateCall).toHaveBeenCalledOnce());
      expect(vi.mocked(modelCreateCall).mock.calls[0][1].litellm_params).toMatchObject({
        api_key: "delegated-token",
      });
    });

    it("creates a per-user GitHub credential and attaches it without sending its auth type inline", async () => {
      const user = userEvent.setup();
      vi.mocked(credentialCreateCall).mockClear();
      vi.mocked(modelCreateCall).mockClear();
      const props = await renderAsAdmin(Providers.GITHUB_COPILOT);

      await chooseSelectOption(user, screen.getByRole("combobox", { name: "Auth Type:" }), "Per-user GitHub OAuth");
      await user.click(screen.getByRole("button", { name: "Create credential" }));
      const dialog = await screen.findByRole("dialog");
      const providerSelect = within(dialog).getByPlaceholderText("Select a provider");
      expect(providerSelect).toHaveValue(Providers.GITHUB_COPILOT);
      expect(providerSelect).toBeDisabled();
      expect(within(dialog).getByRole("combobox", { name: "Auth Type:" })).toHaveTextContent("Per-user GitHub OAuth");
      fireEvent.change(within(dialog).getByLabelText("Credential Name:"), {
        target: { value: "github-per-user" },
      });
      await user.click(within(dialog).getByRole("button", { name: "Add Credential" }));

      await waitFor(() =>
        expect(credentialCreateCall).toHaveBeenCalledWith("test-access-token", {
          credential_name: "github-per-user",
          credential_values: { github_copilot_auth_type: "per_user_oauth" },
          credential_info: { custom_llm_provider: Providers.GITHUB_COPILOT },
        }),
      );
      expect(props.form.getValues("litellm_credential_name")).toBe("github-per-user");

      props.handleOk.mockImplementation(async () => {
        await handleAddModelSubmit(
          {
            litellm_credential_name: props.form.getValues("litellm_credential_name"),
            custom_llm_provider: Providers.GITHUB_COPILOT,
            model_mappings: [
              {
                public_name: "copilot-model",
                litellm_model: "github_copilot/chat",
              },
            ],
          },
          "test-access-token",
          { resetFields: vi.fn() },
        );
        return true;
      });
      await user.click(screen.getByRole("button", { name: "Add Model" }));

      await waitFor(() => expect(modelCreateCall).toHaveBeenCalledOnce());
      const modelParams = vi.mocked(modelCreateCall).mock.calls[0][1].litellm_params;
      expect(modelParams).toMatchObject({ litellm_credential_name: "github-per-user" });
      expect(modelParams).not.toHaveProperty("github_copilot_auth_type");
    });

    it("keeps shared GitHub device login inline when adding a model", async () => {
      const user = userEvent.setup();
      vi.mocked(credentialCreateCall).mockClear();
      vi.mocked(modelCreateCall).mockClear();
      const props = await renderAsAdmin(Providers.GITHUB_COPILOT);

      expect(await screen.findByRole("combobox", { name: "Auth Type:" })).toHaveTextContent("Shared device login");
      expect(screen.queryByRole("button", { name: "Create credential" })).not.toBeInTheDocument();
      fireEvent.change(screen.getByLabelText("API Key"), { target: { value: "shared-device-token" } });
      props.handleOk.mockImplementation(async () => {
        await handleAddModelSubmit(
          {
            api_key: "shared-device-token",
            custom_llm_provider: Providers.GITHUB_COPILOT,
            model_mappings: [
              {
                public_name: "copilot-model",
                litellm_model: "github_copilot/chat",
              },
            ],
          },
          "test-access-token",
          { resetFields: vi.fn() },
        );
        return true;
      });

      await user.click(screen.getByRole("button", { name: "Add Model" }));

      await waitFor(() => expect(modelCreateCall).toHaveBeenCalledOnce());
      expect(credentialCreateCall).not.toHaveBeenCalled();
      expect(vi.mocked(modelCreateCall).mock.calls[0][1].litellm_params).toMatchObject({
        api_key: "shared-device-token",
      });
    });
  });

  describe("cache control bindings reach the parent form store", () => {
    const renderWithForm = async () => {
      const mockUseAuthorized = vi.mocked(await import("@/app/(dashboard)/hooks/useAuthorized"));
      mockUseAuthorized.default.mockReturnValue(mockAuthorizedUser("proxy_admin", "user-1", true));
      const props = createTestProps();
      const user = userEvent.setup({ pointerEventsCheck: PointerEventsCheckLevel.Never });
      renderWithProviders(<AddModelForm {...props} />);
      await screen.findByText("Provider");

      return {
        user,
        openCacheControl: async () => {
          await user.click(await screen.findByText("Advanced Settings"));
          await user.click(screen.getByRole("switch", { name: "Cache Control Injection Points" }));
          await screen.findByText("Add Injection Point");
        },
        closeCacheControl: async () => {
          await user.click(screen.getByRole("switch", { name: "Cache Control Injection Points" }));
          await waitFor(() => expect(screen.queryByText("Add Injection Point")).not.toBeInTheDocument());
        },
        mountedValues: async (): Promise<Record<string, unknown>> => props.mountedValues(),
      };
    };

    it("omits both cache control keys while the section is untouched", async () => {
      const { mountedValues } = await renderWithForm();
      const values = await mountedValues();
      expect(values).not.toHaveProperty("cache_control_injection_points");
      expect(values.cache_control).toBeUndefined();
    });

    it("sends the seeded injection point once the toggle is on", async () => {
      const { openCacheControl, mountedValues } = await renderWithForm();
      await openCacheControl();
      const values = await mountedValues();
      expect(values.cache_control).toBe(true);
      expect(values.cache_control_injection_points).toEqual([{ location: "message" }]);
    });

    it("carries an edited role and keeps the index a string, as the antd control did", async () => {
      const { user, openCacheControl, mountedValues } = await renderWithForm();
      await openCacheControl();

      await user.click(screen.getByText("Select a role"));
      await user.click(await screen.findByText("System"));
      await user.type(screen.getByPlaceholderText("Optional"), "3");

      const values = await mountedValues();
      expect(values.cache_control_injection_points).toEqual([{ location: "message", role: "system", index: "3" }]);
    });

    it("adds a second injection point row", async () => {
      const { user, openCacheControl, mountedValues } = await renderWithForm();
      await openCacheControl();

      await user.click(screen.getByText("Add Injection Point"));
      await waitFor(() => expect(screen.getAllByPlaceholderText("Optional")).toHaveLength(2));
      await user.type(screen.getAllByPlaceholderText("Optional")[1], "7");

      const values = await mountedValues();
      expect(values.cache_control_injection_points).toEqual([
        { location: "message" },
        { location: "message", index: "7" },
      ]);
    });

    it("drops the injection points again when the toggle goes back off", async () => {
      const { openCacheControl, closeCacheControl, mountedValues } = await renderWithForm();
      await openCacheControl();
      await closeCacheControl();

      const values = await mountedValues();
      expect(values.cache_control).toBe(false);
      expect(values).not.toHaveProperty("cache_control_injection_points");
    });
  });
});
