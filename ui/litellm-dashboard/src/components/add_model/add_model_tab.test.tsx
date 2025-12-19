import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, renderHook, screen, waitFor } from "@testing-library/react";
import { Form } from "antd";
import type { UploadProps } from "antd/es/upload";
import { describe, expect, it, vi } from "vitest";
import type { Team } from "../key_team_helpers/key_list";
import type { CredentialItem } from "../networking";
import { Providers } from "../provider_info_helpers";
import AddModelTab from "./add_model_tab";

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
    getProviderCreateMetadata: vi.fn().mockResolvedValue([
      {
        provider: "OpenAI",
        provider_display_name: "OpenAI",
        litellm_provider: "openai",
        default_model_placeholder: "gpt-3.5-turbo",
        credential_fields: [],
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
    ],
    isLoading: false,
    error: null,
  }),
}));

const createQueryClient = () =>
  new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
        staleTime: Infinity,
        gcTime: Infinity,
        refetchOnWindowFocus: false,
        refetchOnReconnect: false,
        refetchOnMount: false,
      },
    },
  });

const createTestProps = () => {
  const { result } = renderHook(() => Form.useForm());
  const [form] = result.current;

  const handleOk = vi.fn();
  const setSelectedProvider = vi.fn();
  const setProviderModelsFn = vi.fn();
  const getPlaceholder = vi.fn((provider: Providers) => `Enter ${provider} model name`);
  const setShowAdvancedSettings = vi.fn();

  const selectedProvider = Providers.OpenAI;
  const providerModels = ["gpt-4", "gpt-3.5-turbo"];
  const showAdvancedSettings = false;

  const teams: Team[] = [
    {
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
    },
  ];

  const credentials: CredentialItem[] = [
    {
      credential_name: "test-credential",
      credential_values: {},
      credential_info: {
        custom_llm_provider: "openai",
        description: "Test credential",
      },
    },
  ];

  const uploadProps: UploadProps = {
    beforeUpload: () => false,
    showUploadList: false,
  };

  return {
    form,
    handleOk,
    setSelectedProvider,
    setProviderModelsFn,
    getPlaceholder,
    setShowAdvancedSettings,
    selectedProvider,
    providerModels,
    showAdvancedSettings,
    teams,
    credentials,
    uploadProps,
    accessToken: "test-access-token",
    userRole: "Admin",
    premiumUser: true,
  };
};

describe("Add Model Tab", () => {
  it("should render", async () => {
    const props = createTestProps();
    const queryClient = createQueryClient();

    render(
      <QueryClientProvider client={queryClient}>
        <AddModelTab
          form={props.form}
          handleOk={props.handleOk}
          selectedProvider={props.selectedProvider}
          setSelectedProvider={props.setSelectedProvider}
          providerModels={props.providerModels}
          setProviderModelsFn={props.setProviderModelsFn}
          getPlaceholder={props.getPlaceholder}
          uploadProps={props.uploadProps}
          showAdvancedSettings={props.showAdvancedSettings}
          setShowAdvancedSettings={props.setShowAdvancedSettings}
          teams={props.teams}
          credentials={props.credentials}
          accessToken={props.accessToken}
          userRole={props.userRole}
          premiumUser={props.premiumUser}
        />
      </QueryClientProvider>,
    );

    expect(await screen.findByRole("tab", { name: "Add Model" })).toBeInTheDocument();
  }, 10000); // This test is flaky, adding a timeout until we find a better solution

  it("should display both Add Model and Add Auto Router tabs", async () => {
    const props = createTestProps();
    const queryClient = createQueryClient();

    render(
      <QueryClientProvider client={queryClient}>
        <AddModelTab
          form={props.form}
          handleOk={props.handleOk}
          selectedProvider={props.selectedProvider}
          setSelectedProvider={props.setSelectedProvider}
          providerModels={props.providerModels}
          setProviderModelsFn={props.setProviderModelsFn}
          getPlaceholder={props.getPlaceholder}
          uploadProps={props.uploadProps}
          showAdvancedSettings={props.showAdvancedSettings}
          setShowAdvancedSettings={props.setShowAdvancedSettings}
          teams={props.teams}
          credentials={props.credentials}
          accessToken={props.accessToken}
          userRole={props.userRole}
          premiumUser={props.premiumUser}
        />
      </QueryClientProvider>,
    );

    expect(await screen.findByRole("tab", { name: "Add Model" })).toBeInTheDocument();
    expect(await screen.findByRole("tab", { name: "Add Auto Router" })).toBeInTheDocument();
  });

  it("should display provider selection field", async () => {
    const props = createTestProps();
    const queryClient = createQueryClient();

    render(
      <QueryClientProvider client={queryClient}>
        <AddModelTab
          form={props.form}
          handleOk={props.handleOk}
          selectedProvider={props.selectedProvider}
          setSelectedProvider={props.setSelectedProvider}
          providerModels={props.providerModels}
          setProviderModelsFn={props.setProviderModelsFn}
          getPlaceholder={props.getPlaceholder}
          uploadProps={props.uploadProps}
          showAdvancedSettings={props.showAdvancedSettings}
          setShowAdvancedSettings={props.setShowAdvancedSettings}
          teams={props.teams}
          credentials={props.credentials}
          accessToken={props.accessToken}
          userRole={props.userRole}
          premiumUser={props.premiumUser}
        />
      </QueryClientProvider>,
    );

    expect(await screen.findByText("Provider")).toBeInTheDocument();
  });

  it("should display Test Connect and Add Model buttons", async () => {
    const props = createTestProps();
    const queryClient = createQueryClient();

    render(
      <QueryClientProvider client={queryClient}>
        <AddModelTab
          form={props.form}
          handleOk={props.handleOk}
          selectedProvider={props.selectedProvider}
          setSelectedProvider={props.setSelectedProvider}
          providerModels={props.providerModels}
          setProviderModelsFn={props.setProviderModelsFn}
          getPlaceholder={props.getPlaceholder}
          uploadProps={props.uploadProps}
          showAdvancedSettings={props.showAdvancedSettings}
          setShowAdvancedSettings={props.setShowAdvancedSettings}
          teams={props.teams}
          credentials={props.credentials}
          accessToken={props.accessToken}
          userRole={props.userRole}
          premiumUser={props.premiumUser}
        />
      </QueryClientProvider>,
    );

    // Wait for async operations to complete and buttons to appear
    await waitFor(
      async () => {
        const testConnectButtons = await screen.findAllByRole("button", { name: "Test Connect" });
        expect(testConnectButtons.length).toBeGreaterThan(0);
        const addModelButton = await screen.findByRole("button", { name: "Add Model" });
        expect(addModelButton).toBeInTheDocument();
      },
      { timeout: 10000 },
    );
  }, 15000); // 15 second timeout to allow waitFor to complete
});
