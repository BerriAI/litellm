import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import React, { ReactNode } from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import ModelInfoView from "./model_info_view";
import { toast } from "@/lib/toast";
import * as networking from "./networking";
vi.mock(
  "@/app/(dashboard)/hooks/autoRouter/useComplexityScorerDefaults",
  async () => await import("../../tests/mocks/complexityScorerDefaults"),
);

vi.mock("../../utils/dataUtils", () => ({
  copyToClipboard: vi.fn().mockResolvedValue(true),
}));

vi.mock("./networking", () => ({
  modelInfoV1Call: vi.fn(),
  credentialGetCall: vi.fn(),
  credentialListCall: vi.fn(),
  getGuardrailsList: vi.fn(),
  tagListCall: vi.fn(),
  testConnectionRequest: vi.fn(),
  testModelGroupConnection: vi.fn(),
  modelPatchUpdateCall: vi.fn(),
  modelDeleteCall: vi.fn(),
  credentialCreateCall: vi.fn(),
  vectorStoreListCall: vi.fn(),
}));

const mockUseModelsInfo = vi.fn();
const mockUseModelHub = vi.fn();

vi.mock("@/app/(dashboard)/hooks/models/useModels", () => ({
  useModelsInfo: (...args: any[]) => mockUseModelsInfo(...args),
  useModelHub: (...args: any[]) => mockUseModelHub(...args),
}));

const mockUseModelCostMap = vi.fn();
vi.mock("@/app/(dashboard)/hooks/models/useModelCostMap", () => ({
  useModelCostMap: (...args: any[]) => mockUseModelCostMap(...args),
}));

const mockUsePtuCostAttributionEnabled = vi.fn();
vi.mock("@/app/(dashboard)/hooks/uiSettings/usePtuCostAttributionEnabled", () => ({
  usePtuCostAttributionEnabled: () => mockUsePtuCostAttributionEnabled(),
}));

const mockToast = vi.mocked(toast);
const mockModelInfoV1Call = vi.mocked(networking.modelInfoV1Call);
const mockCredentialGetCall = vi.mocked(networking.credentialGetCall);
const mockCredentialListCall = vi.mocked(networking.credentialListCall);
const mockGetGuardrailsList = vi.mocked(networking.getGuardrailsList);
const mockTagListCall = vi.mocked(networking.tagListCall);
const mockTestConnectionRequest = vi.mocked(networking.testConnectionRequest);
const mockTestModelGroupConnection = vi.mocked(networking.testModelGroupConnection);
const mockModelPatchUpdateCall = vi.mocked(networking.modelPatchUpdateCall);
const mockModelDeleteCall = vi.mocked(networking.modelDeleteCall);
const mockCredentialCreateCall = vi.mocked(networking.credentialCreateCall);
const mockVectorStoreListCall = vi.mocked(networking.vectorStoreListCall);

describe("ModelInfoView", () => {
  let queryClient: QueryClient;

  const defaultModelData = {
    model_name: "GPT-4",
    litellm_params: {
      model: "gpt-4",
      api_base: "https://api.openai.com/v1",
      custom_llm_provider: "openai",
      litellm_credential_name: "selected-credential",
    },
    model_info: {
      id: "123",
      created_by: "123",
      created_at: "2024-01-01T00:00:00Z",
      db_model: true,
      input_cost_per_token: 0.00003,
      output_cost_per_token: 0.00006,
    },
  };

  const DEFAULT_ADMIN_PROPS = {
    modelId: "123",
    onClose: vi.fn(),
    accessToken: "test-token",
    userID: "123",
    userRole: "Admin",
    onModelUpdate: vi.fn(),
    modelAccessGroups: ["group1", "group2"],
  };

  beforeEach(() => {
    queryClient = new QueryClient({
      defaultOptions: {
        queries: {
          retry: false,
        },
      },
    });
    vi.clearAllMocks();
    mockUsePtuCostAttributionEnabled.mockReturnValue(false);

    mockUseModelsInfo.mockReturnValue({
      data: {
        data: [defaultModelData],
      },
      isLoading: false,
      error: null,
    });

    mockUseModelHub.mockReturnValue({
      data: {
        data: [],
      },
      isLoading: false,
      error: null,
    });

    mockUseModelCostMap.mockReturnValue({
      data: {},
      isLoading: false,
      error: null,
    });

    mockModelInfoV1Call.mockResolvedValue({
      data: [defaultModelData],
    });

    mockCredentialGetCall.mockResolvedValue({
      credential_name: "test-credential",
      credential_values: {},
      credential_info: {},
    });
    mockCredentialListCall.mockResolvedValue({
      credentials: [
        {
          credential_name: "selected-credential",
          credential_values: {},
          credential_info: {},
        },
      ],
    });

    mockGetGuardrailsList.mockResolvedValue({
      guardrails: [{ guardrail_name: "content_filter" }, { guardrail_name: "toxicity_filter" }],
    });

    mockTagListCall.mockResolvedValue({
      test_tag: {
        name: "test_tag",
        description: "A test tag",
        models: [],
        created_at: "2024-01-01T00:00:00Z",
        updated_at: "2024-01-01T00:00:00Z",
      },
      production_tag: {
        name: "production_tag",
        description: "Production ready models",
        models: [],
        created_at: "2024-01-01T00:00:00Z",
        updated_at: "2024-01-01T00:00:00Z",
      },
    });

    mockTestConnectionRequest.mockResolvedValue({
      status: "success",
    });

    mockVectorStoreListCall.mockResolvedValue({
      data: [
        { vector_store_id: "vs-alpha", vector_store_name: "Alpha" },
        { vector_store_id: "vs-beta", vector_store_name: "Beta" },
      ],
    } as never);
    mockModelPatchUpdateCall.mockResolvedValue({});
    mockModelDeleteCall.mockResolvedValue({});
    mockCredentialCreateCall.mockResolvedValue({});
  });

  const wrapper = ({ children }: { children: ReactNode }) =>
    React.createElement(QueryClientProvider, { client: queryClient }, children);

  it("should render", async () => {
    render(<ModelInfoView {...DEFAULT_ADMIN_PROPS} />, { wrapper });
    await waitFor(() => {
      expect(screen.getByText("Model Settings")).toBeInTheDocument();
    });
  });

  it("should display loading state when model data is loading", () => {
    mockUseModelsInfo.mockReturnValue({
      data: null,
      isLoading: true,
      error: null,
    });

    render(<ModelInfoView {...DEFAULT_ADMIN_PROPS} />, { wrapper });
    expect(screen.getByText("Loading...")).toBeInTheDocument();
  });

  it("should display not found message when model data is not available", async () => {
    mockUseModelsInfo.mockReturnValue({
      data: {
        data: [],
      },
      isLoading: false,
      error: null,
    });

    render(<ModelInfoView {...DEFAULT_ADMIN_PROPS} />, { wrapper });
    await waitFor(() => {
      expect(screen.getByText("Model not found")).toBeInTheDocument();
    });
  });

  it("should display model name in the header", async () => {
    render(<ModelInfoView {...DEFAULT_ADMIN_PROPS} />, { wrapper });
    await waitFor(() => {
      expect(screen.getByText(/Public Model Name:/)).toBeInTheDocument();
    });
  });

  it("should display back button that calls onClose when clicked", async () => {
    const mockOnClose = vi.fn();
    const user = userEvent.setup();
    render(<ModelInfoView {...DEFAULT_ADMIN_PROPS} onClose={mockOnClose} />, { wrapper });

    await waitFor(() => {
      expect(screen.getByText("Model Settings")).toBeInTheDocument();
    });

    const backButton = screen.getByRole("button", { name: /back to models/i });
    await user.click(backButton);

    expect(mockOnClose).toHaveBeenCalledTimes(1);
  });

  it("should display test connection button", async () => {
    render(<ModelInfoView {...DEFAULT_ADMIN_PROPS} />, { wrapper });
    await waitFor(() => {
      expect(screen.getByRole("button", { name: /test connection/i })).toBeInTheDocument();
    });
  });

  it("should test connection when test connection button is clicked", async () => {
    const user = userEvent.setup();
    render(<ModelInfoView {...DEFAULT_ADMIN_PROPS} />, { wrapper });

    await waitFor(() => {
      expect(screen.getByText("Model Settings")).toBeInTheDocument();
    });

    const testButton = screen.getByRole("button", { name: /test connection/i });
    await user.click(testButton);

    await waitFor(() => {
      expect(mockTestConnectionRequest).toHaveBeenCalled();
      expect(mockToast.success).toHaveBeenCalledWith("Connection test successful!");
    });
  });

  it("should pass model_info.id to disambiguate duplicate model_name deployments", async () => {
    // Regression test: when two deployments share `model_name` (e.g.
    // wildcard `openai/*` with different `api_base` values), the UI
    // must forward the clicked row's `model_info.id` to the backend.
    // Otherwise /health/test_connection silently probes deployments[0]
    // instead of the deployment the user actually selected.
    const user = userEvent.setup();
    render(<ModelInfoView {...DEFAULT_ADMIN_PROPS} />, { wrapper });

    await waitFor(() => {
      expect(screen.getByText("Model Settings")).toBeInTheDocument();
    });

    const testButton = screen.getByRole("button", { name: /test connection/i });
    await user.click(testButton);

    await waitFor(() => {
      expect(mockTestConnectionRequest).toHaveBeenCalled();
    });

    const callArgs = mockTestConnectionRequest.mock.calls[0];
    // Signature: (accessToken, litellm_params, model_info, mode)
    const modelInfoArg = callArgs[2] as Record<string, unknown>;
    expect(modelInfoArg).toBeDefined();
    expect(modelInfoArg.id).toBe("123");
  });

  it("should display error notification when connection test fails", async () => {
    const user = userEvent.setup();
    mockTestConnectionRequest.mockRejectedValue(new Error("Connection failed"));

    render(<ModelInfoView {...DEFAULT_ADMIN_PROPS} />, { wrapper });

    await waitFor(() => {
      expect(screen.getByText("Model Settings")).toBeInTheDocument();
    });

    const testButton = screen.getByRole("button", { name: /test connection/i });
    await user.click(testButton);

    await waitFor(() => {
      expect(mockToast.error).toHaveBeenCalled();
    });
  });

  it("should display reuse credentials button for admin users", async () => {
    render(<ModelInfoView {...DEFAULT_ADMIN_PROPS} />, { wrapper });
    await waitFor(() => {
      expect(screen.getByRole("button", { name: /re-use credentials/i })).toBeInTheDocument();
    });
  });

  it("should disable reuse credentials button for non-admin users", async () => {
    render(<ModelInfoView {...DEFAULT_ADMIN_PROPS} userRole="User" />, { wrapper });
    await waitFor(() => {
      const button = screen.getByRole("button", { name: /re-use credentials/i });
      expect(button).toBeDisabled();
    });
  });

  it("should display delete model button", async () => {
    render(<ModelInfoView {...DEFAULT_ADMIN_PROPS} />, { wrapper });
    await waitFor(() => {
      expect(screen.getByRole("button", { name: /delete model/i })).toBeInTheDocument();
    });
  });

  it("should disable delete button when model is not a DB model", async () => {
    const nonDbModelData = {
      ...defaultModelData,
      model_info: {
        ...defaultModelData.model_info,
        db_model: false,
      },
    };

    mockUseModelsInfo.mockReturnValue({
      data: {
        data: [nonDbModelData],
      },
      isLoading: false,
      error: null,
    });

    render(<ModelInfoView {...DEFAULT_ADMIN_PROPS} />, { wrapper });
    await waitFor(() => {
      const deleteButton = screen.getByRole("button", { name: /delete model/i });
      expect(deleteButton).toBeDisabled();
    });
  });

  it("should disable delete button when user is not admin and did not create the model", async () => {
    const nonCreatedByUserModelData = {
      ...defaultModelData,
      model_info: {
        ...defaultModelData.model_info,
        created_by: "456",
      },
    };

    mockUseModelsInfo.mockReturnValue({
      data: {
        data: [nonCreatedByUserModelData],
      },
      isLoading: false,
      error: null,
    });

    render(<ModelInfoView {...DEFAULT_ADMIN_PROPS} userRole="User" />, { wrapper });
    await waitFor(() => {
      const deleteButton = screen.getByRole("button", { name: /delete model/i });
      expect(deleteButton).toBeDisabled();
    });
  });

  it("should display overview and raw JSON tabs", async () => {
    render(<ModelInfoView {...DEFAULT_ADMIN_PROPS} />, { wrapper });
    await waitFor(() => {
      expect(screen.getByRole("tab", { name: /overview/i })).toBeInTheDocument();
      expect(screen.getByRole("tab", { name: /raw json/i })).toBeInTheDocument();
    });
  });

  it("keeps the edit form and its touched fields alive across a tab switch", async () => {
    const user = userEvent.setup();
    render(<ModelInfoView {...DEFAULT_ADMIN_PROPS} />, { wrapper });

    await user.click(await screen.findByRole("button", { name: /edit settings/i }));
    const costInput = screen.getByPlaceholderText("Enter input cost") as HTMLInputElement;
    await user.clear(costInput);
    await user.type(costInput, "5");

    await user.click(screen.getByRole("tab", { name: /raw json/i }));
    await user.click(screen.getByRole("tab", { name: /overview/i }));

    expect(screen.getByPlaceholderText("Enter input cost")).toBe(costInput);
    expect(Number(costInput.value)).toBe(5);
    await user.click(screen.getByRole("button", { name: /save changes/i }));

    await waitFor(() => {
      expect(mockModelPatchUpdateCall).toHaveBeenCalled();
    });
    expect(mockModelPatchUpdateCall.mock.calls[0][1].litellm_params.input_cost_per_token).toBeCloseTo(5 / 1_000_000);
  });

  it("should display model information in overview tab", async () => {
    render(<ModelInfoView {...DEFAULT_ADMIN_PROPS} />, { wrapper });
    await waitFor(() => {
      expect(screen.getByText("Provider")).toBeInTheDocument();
      expect(screen.getByText("LiteLLM Model")).toBeInTheDocument();
      expect(screen.getByText("Pricing")).toBeInTheDocument();
    });
  });

  it("should display edit settings button when user can edit model", async () => {
    render(<ModelInfoView {...DEFAULT_ADMIN_PROPS} />, { wrapper });
    await waitFor(() => {
      expect(screen.getByRole("button", { name: /edit settings/i })).toBeInTheDocument();
    });
  });

  it("should not display edit settings button when model is not a DB model", async () => {
    const nonDbModelData = {
      ...defaultModelData,
      model_info: {
        ...defaultModelData.model_info,
        db_model: false,
      },
    };

    mockUseModelsInfo.mockReturnValue({
      data: {
        data: [nonDbModelData],
      },
      isLoading: false,
      error: null,
    });

    render(<ModelInfoView {...DEFAULT_ADMIN_PROPS} />, { wrapper });
    await waitFor(() => {
      expect(screen.queryByRole("button", { name: /edit settings/i })).not.toBeInTheDocument();
    });
  });

  it("should enter edit mode when edit settings button is clicked", async () => {
    const user = userEvent.setup();
    render(<ModelInfoView {...DEFAULT_ADMIN_PROPS} />, { wrapper });

    await waitFor(() => {
      expect(screen.getByRole("button", { name: /edit settings/i })).toBeInTheDocument();
    });

    const editButton = screen.getByRole("button", { name: /edit settings/i });
    await user.click(editButton);

    await waitFor(() => {
      expect(screen.getByRole("button", { name: /save changes/i })).toBeInTheDocument();
      expect(screen.getByRole("button", { name: /cancel/i })).toBeInTheDocument();
    });
  });

  it("should display form fields in edit mode", async () => {
    const user = userEvent.setup();
    render(<ModelInfoView {...DEFAULT_ADMIN_PROPS} />, { wrapper });

    await waitFor(() => {
      expect(screen.getByRole("button", { name: /edit settings/i })).toBeInTheDocument();
    });

    const editButton = screen.getByRole("button", { name: /edit settings/i });
    await user.click(editButton);

    await waitFor(() => {
      expect(screen.getByPlaceholderText("Enter model name")).toBeInTheDocument();
      expect(screen.getByPlaceholderText("Enter LiteLLM model name")).toBeInTheDocument();
    });
  });

  it("should allow editing model name in edit mode", async () => {
    const user = userEvent.setup();
    render(<ModelInfoView {...DEFAULT_ADMIN_PROPS} />, { wrapper });

    await waitFor(() => {
      expect(screen.getByRole("button", { name: /edit settings/i })).toBeInTheDocument();
    });

    const editButton = screen.getByRole("button", { name: /edit settings/i });
    await user.click(editButton);

    const modelNameInput = await screen.findByPlaceholderText("Enter model name");
    await user.clear(modelNameInput);
    fireEvent.change(modelNameInput, { target: { value: "Updated Model Name" } });

    expect(modelNameInput).toHaveValue("Updated Model Name");
  });

  it("should cancel editing when cancel button is clicked", async () => {
    const user = userEvent.setup();
    render(<ModelInfoView {...DEFAULT_ADMIN_PROPS} />, { wrapper });

    await waitFor(() => {
      expect(screen.getByRole("button", { name: /edit settings/i })).toBeInTheDocument();
    });

    const editButton = screen.getByRole("button", { name: /edit settings/i });
    await user.click(editButton);

    await waitFor(() => {
      expect(screen.getByRole("button", { name: /cancel/i })).toBeInTheDocument();
    });

    const cancelButton = screen.getByRole("button", { name: /cancel/i });
    await user.click(cancelButton);

    await waitFor(() => {
      expect(screen.getByRole("button", { name: /edit settings/i })).toBeInTheDocument();
      expect(screen.queryByRole("button", { name: /save changes/i })).not.toBeInTheDocument();
    });
  });

  it("should save model changes when save button is clicked", async () => {
    const user = userEvent.setup();
    const mockOnModelUpdate = vi.fn();
    render(<ModelInfoView {...DEFAULT_ADMIN_PROPS} onModelUpdate={mockOnModelUpdate} />, { wrapper });

    await waitFor(() => {
      expect(screen.getByRole("button", { name: /edit settings/i })).toBeInTheDocument();
    });

    const editButton = screen.getByRole("button", { name: /edit settings/i });
    await user.click(editButton);

    await waitFor(() => {
      expect(screen.getByRole("button", { name: /save changes/i })).toBeInTheDocument();
    });

    const saveButton = screen.getByRole("button", { name: /save changes/i });
    await user.click(saveButton);

    await waitFor(() => {
      expect(mockModelPatchUpdateCall).toHaveBeenCalled();
      expect(mockToast.success).toHaveBeenCalledWith("Model settings updated successfully");
      expect(mockOnModelUpdate).toHaveBeenCalled();
    });
  });

  it("should display tags section", async () => {
    render(<ModelInfoView {...DEFAULT_ADMIN_PROPS} />, { wrapper });
    await waitFor(() => {
      expect(screen.getByText("Tags")).toBeInTheDocument();
    });
  });

  it("should display LiteLLM Params section", async () => {
    render(<ModelInfoView {...DEFAULT_ADMIN_PROPS} />, { wrapper });
    await waitFor(() => {
      expect(screen.getByText("LiteLLM Params")).toBeInTheDocument();
    });
  });

  it("should show existing credentials field in edit mode", async () => {
    const user = userEvent.setup();
    render(<ModelInfoView {...DEFAULT_ADMIN_PROPS} />, { wrapper });

    await waitFor(() => {
      expect(screen.getByRole("button", { name: /edit settings/i })).toBeInTheDocument();
    });

    await user.click(screen.getByRole("button", { name: /edit settings/i }));

    await waitFor(() => {
      expect(screen.getByText("Existing Credentials")).toBeInTheDocument();
    });
  });

  it("should keep selector credential and ignore litellm_credential_name from LiteLLM Params json", async () => {
    const user = userEvent.setup();
    render(<ModelInfoView {...DEFAULT_ADMIN_PROPS} />, { wrapper });

    await waitFor(() => {
      expect(screen.getByRole("button", { name: /edit settings/i })).toBeInTheDocument();
    });

    await user.click(screen.getByRole("button", { name: /edit settings/i }));

    const litellmParamsInput = screen
      .getAllByRole("textbox")
      .find(
        (input) =>
          input.tagName === "TEXTAREA" && (input as HTMLTextAreaElement).value.includes('"custom_llm_provider"'),
      );
    expect(litellmParamsInput).toBeDefined();
    if (!litellmParamsInput) {
      return;
    }
    expect((litellmParamsInput as HTMLTextAreaElement).value).not.toContain("litellm_credential_name");
    await user.clear(litellmParamsInput);
    await user.paste(`{"litellm_credential_name":"from-json","timeout":42}`);

    await user.click(screen.getByRole("button", { name: /save changes/i }));

    await waitFor(() => {
      expect(mockModelPatchUpdateCall).toHaveBeenCalled();
    });

    const updatePayload = mockModelPatchUpdateCall.mock.calls[0][1];
    expect(updatePayload.litellm_params.litellm_credential_name).toBe("selected-credential");
    expect(updatePayload.litellm_params.litellm_credential_name).not.toBe("from-json");
  });

  it("should not include vector_store_ids in update payload when model has none", async () => {
    // Regression: editing a model without vector stores used to inject
    // vector_store_ids: [] into litellm_params, which then propagated to
    // inference requests and broke Anthropic calls.
    const user = userEvent.setup();
    render(<ModelInfoView {...DEFAULT_ADMIN_PROPS} />, { wrapper });

    await waitFor(() => {
      expect(screen.getByRole("button", { name: /edit settings/i })).toBeInTheDocument();
    });

    await user.click(screen.getByRole("button", { name: /edit settings/i }));

    await waitFor(() => {
      expect(screen.getByRole("button", { name: /save changes/i })).toBeInTheDocument();
    });

    await user.click(screen.getByRole("button", { name: /save changes/i }));

    await waitFor(() => {
      expect(mockModelPatchUpdateCall).toHaveBeenCalled();
    });

    const updatePayload = mockModelPatchUpdateCall.mock.calls[0][1];
    expect(updatePayload.litellm_params).not.toHaveProperty("vector_store_ids");
  });

  describe("PTU cost attribution gate", () => {
    const ptuModelData = {
      ...defaultModelData,
      // Zero per-token pricing is what the backend stores for a PTU deployment, since the flat
      // cost of its reserved capacity already covers the traffic that capacity serves.
      litellm_params: { ...defaultModelData.litellm_params, input_cost_per_token: 0, output_cost_per_token: 0 },
      model_info: {
        ...defaultModelData.model_info,
        team_id: "team-1",
        input_cost_per_token: 0,
        output_cost_per_token: 0,
        ptu_count: 15,
        cost_per_ptu_per_hour: 2,
        ptu_effective_from: "2026-07-01T00:00:00+00:00",
        ptu_effective_to: "2026-08-01T00:00:00+00:00",
      },
    };

    const renderWithPtuModel = () => {
      mockUseModelsInfo.mockReturnValue({ data: { data: [ptuModelData] }, isLoading: false, error: null });
      mockModelInfoV1Call.mockResolvedValue({ data: [ptuModelData] });
      return render(<ModelInfoView {...DEFAULT_ADMIN_PROPS} />, { wrapper });
    };

    it("hides the PTU fields when disabled, even for a model that already stores PTU config", async () => {
      renderWithPtuModel();

      await waitFor(() => {
        expect(screen.getByText("Model Settings")).toBeInTheDocument();
      });

      expect(screen.queryByText("PTU Count")).not.toBeInTheDocument();
      expect(screen.queryByText("Cost per PTU / Hour (USD)")).not.toBeInTheDocument();
      expect(screen.queryByText("PTU Effective From (UTC)")).not.toBeInTheDocument();
      expect(screen.queryByText("PTU Effective To (UTC)")).not.toBeInTheDocument();
    });

    it("shows the PTU fields when enabled", async () => {
      mockUsePtuCostAttributionEnabled.mockReturnValue(true);
      renderWithPtuModel();

      await waitFor(() => {
        expect(screen.getByText("PTU Count")).toBeInTheDocument();
      });
      expect(screen.getByText("Cost per PTU / Hour (USD)")).toBeInTheDocument();
      expect(screen.getByText("PTU Effective From (UTC)")).toBeInTheDocument();
      expect(screen.getByText("PTU Effective To (UTC)")).toBeInTheDocument();
    });

    it("omits PTU fields from the save payload when disabled, so an unrelated edit cannot clear stored config", async () => {
      const user = userEvent.setup();
      renderWithPtuModel();

      await waitFor(() => {
        expect(screen.getByRole("button", { name: /edit settings/i })).toBeInTheDocument();
      });
      await user.click(screen.getByRole("button", { name: /edit settings/i }));

      await waitFor(() => {
        expect(screen.getByRole("button", { name: /save changes/i })).toBeInTheDocument();
      });
      await user.click(screen.getByRole("button", { name: /save changes/i }));

      await waitFor(() => {
        expect(mockModelPatchUpdateCall).toHaveBeenCalled();
      });

      const modelInfo = mockModelPatchUpdateCall.mock.calls[0][1].model_info;
      expect(modelInfo).not.toHaveProperty("ptu_count");
      expect(modelInfo).not.toHaveProperty("cost_per_ptu_per_hour");
      expect(modelInfo).not.toHaveProperty("ptu_effective_from");
      expect(modelInfo).not.toHaveProperty("ptu_effective_to");
    });

    it("shows a zeroed PTU price as 0.0000 rather than Not Set", async () => {
      mockUsePtuCostAttributionEnabled.mockReturnValue(true);
      renderWithPtuModel();

      await waitFor(() => {
        expect(screen.getByText("Input Cost (per 1M tokens)")).toBeInTheDocument();
      });
      for (const label of ["Input Cost (per 1M tokens)", "Output Cost (per 1M tokens)"]) {
        expect(screen.getByText(label).parentElement).toHaveTextContent("0.0000");
      }
    });

    it("blocks the save once the operator types a non-zero per-token cost alongside PTU config", async () => {
      mockUsePtuCostAttributionEnabled.mockReturnValue(true);
      const user = userEvent.setup();
      renderWithPtuModel();

      await waitFor(() => {
        expect(screen.getByRole("button", { name: /edit settings/i })).toBeInTheDocument();
      });
      await user.click(screen.getByRole("button", { name: /edit settings/i }));

      await waitFor(() => {
        expect(screen.getByPlaceholderText("Enter input cost")).toBeInTheDocument();
      });
      await user.clear(screen.getByPlaceholderText("Enter input cost"));
      fireEvent.change(screen.getByPlaceholderText("Enter input cost"), { target: { value: "2.5" } });
      await user.click(screen.getByRole("button", { name: /save changes/i }));

      await waitFor(() => {
        expect(screen.getByText(/bills by reserved capacity/i)).toBeInTheDocument();
      });
      expect(mockModelPatchUpdateCall).not.toHaveBeenCalled();
    });

    it("lets the operator put a cost-map-priced deployment on PTU without clearing the seeded rate", async () => {
      // A rate the form seeded from /model/info is the server's own, so refusing it blocked
      // every attempt to enable PTU from the dashboard.
      mockUsePtuCostAttributionEnabled.mockReturnValue(true);
      const seededModel = {
        ...defaultModelData,
        model_info: { ...defaultModelData.model_info, team_id: "team-1", input_cost_per_token: 0.0000003 },
      };
      mockUseModelsInfo.mockReturnValue({ data: { data: [seededModel] }, isLoading: false, error: null });
      mockModelInfoV1Call.mockResolvedValue({ data: [seededModel] });
      const user = userEvent.setup();
      render(<ModelInfoView {...DEFAULT_ADMIN_PROPS} />, { wrapper });

      await waitFor(() => {
        expect(screen.getByRole("button", { name: /edit settings/i })).toBeInTheDocument();
      });
      await user.click(screen.getByRole("button", { name: /edit settings/i }));

      await waitFor(() => {
        expect(screen.getByPlaceholderText("e.g. 15")).toBeInTheDocument();
      });
      fireEvent.change(screen.getByPlaceholderText("e.g. 15"), { target: { value: "15" } });
      await user.click(screen.getByRole("button", { name: /save changes/i }));

      await waitFor(() => {
        expect(screen.queryByText(/bills by reserved capacity/i)).not.toBeInTheDocument();
      });
    });

    const enterPtuEdit = async (user: ReturnType<typeof userEvent.setup>) => {
      mockUsePtuCostAttributionEnabled.mockReturnValue(true);
      renderWithPtuModel();
      expect(await screen.findByRole("button", { name: /edit settings/i })).toBeInTheDocument();
      await user.click(screen.getByRole("button", { name: /edit settings/i }));
      expect(await screen.findByPlaceholderText("e.g. 15")).toBeInTheDocument();
    };

    const expectBlocked = async (user: ReturnType<typeof userEvent.setup>, message: RegExp) => {
      await user.click(screen.getByRole("button", { name: /save changes/i }));
      expect(await screen.findAllByText(message)).not.toHaveLength(0);
      expect(mockModelPatchUpdateCall).not.toHaveBeenCalled();
    };

    it("skips PTU validation entirely when the feature is disabled, so a half-set stored record still saves", async () => {
      mockUsePtuCostAttributionEnabled.mockReturnValue(false);
      const halfSetPtuModel = {
        ...ptuModelData,
        model_info: { ...ptuModelData.model_info, cost_per_ptu_per_hour: null, ptu_effective_from: null },
      };
      mockUseModelsInfo.mockReturnValue({ data: { data: [halfSetPtuModel] }, isLoading: false, error: null });
      mockModelInfoV1Call.mockResolvedValue({ data: [halfSetPtuModel] });
      const user = userEvent.setup();
      render(<ModelInfoView {...DEFAULT_ADMIN_PROPS} />, { wrapper });

      expect(await screen.findByRole("button", { name: /edit settings/i })).toBeInTheDocument();
      await user.click(screen.getByRole("button", { name: /edit settings/i }));
      expect(await screen.findByRole("button", { name: /save changes/i })).toBeInTheDocument();
      await user.click(screen.getByRole("button", { name: /save changes/i }));

      await waitFor(() => expect(mockModelPatchUpdateCall).toHaveBeenCalled());
      expect(screen.queryByText(/must be set together/i)).not.toBeInTheDocument();
    });

    it("blocks a PTU count above the backend ceiling", async () => {
      const user = userEvent.setup();
      await enterPtuEdit(user);

      await user.clear(screen.getByPlaceholderText("e.g. 15"));
      await user.type(screen.getByPlaceholderText("e.g. 15"), "1000001");

      await expectBlocked(user, /PTU Count must be a whole number between 1 and 1,000,000/i);
    });

    it("blocks a cost per PTU hour above the backend ceiling", async () => {
      const user = userEvent.setup();
      await enterPtuEdit(user);

      await user.clear(screen.getByPlaceholderText("e.g. 2.00"));
      await user.type(screen.getByPlaceholderText("e.g. 2.00"), "2000000");

      await expectBlocked(user, /Cost per PTU \/ Hour must be between 0 and 1,000,000/i);
    });

    it("blocks a half-set PTU count and rate pair", async () => {
      const user = userEvent.setup();
      await enterPtuEdit(user);

      await user.clear(screen.getByPlaceholderText("e.g. 2.00"));

      await expectBlocked(user, /PTU Count and Cost per PTU \/ Hour must be set together/i);
    });

    it("blocks PTU config with no effective start", async () => {
      mockUsePtuCostAttributionEnabled.mockReturnValue(true);
      const undatedPtuModel = {
        ...ptuModelData,
        model_info: { ...ptuModelData.model_info, ptu_effective_from: null, ptu_effective_to: null },
      };
      mockUseModelsInfo.mockReturnValue({ data: { data: [undatedPtuModel] }, isLoading: false, error: null });
      mockModelInfoV1Call.mockResolvedValue({ data: [undatedPtuModel] });
      const user = userEvent.setup();
      render(<ModelInfoView {...DEFAULT_ADMIN_PROPS} />, { wrapper });

      expect(await screen.findByRole("button", { name: /edit settings/i })).toBeInTheDocument();
      await user.click(screen.getByRole("button", { name: /edit settings/i }));
      expect(await screen.findByPlaceholderText("e.g. 15")).toBeInTheDocument();

      await expectBlocked(user, /PTU Effective From is required when PTU Count is set/i);
    });

    it("blocks a PTU window whose end is not after its start", async () => {
      const user = userEvent.setup();
      await enterPtuEdit(user);

      fireEvent.change(screen.getByLabelText("PTU Effective To (UTC)"), {
        target: { value: "2026-06-01T00:00:00" },
      });

      await expectBlocked(user, /PTU Effective To must be after PTU Effective From/i);
    });

    it("sends the PTU fields on save when enabled", async () => {
      mockUsePtuCostAttributionEnabled.mockReturnValue(true);
      const user = userEvent.setup();
      renderWithPtuModel();

      await waitFor(() => {
        expect(screen.getByRole("button", { name: /edit settings/i })).toBeInTheDocument();
      });
      await user.click(screen.getByRole("button", { name: /edit settings/i }));

      await waitFor(() => {
        expect(screen.getByRole("button", { name: /save changes/i })).toBeInTheDocument();
      });
      await user.click(screen.getByRole("button", { name: /save changes/i }));

      await waitFor(() => {
        expect(mockModelPatchUpdateCall).toHaveBeenCalled();
      });

      const modelInfo = mockModelPatchUpdateCall.mock.calls[0][1].model_info;
      expect(modelInfo.ptu_count).toBe(15);
      expect(modelInfo.cost_per_ptu_per_hour).toBe(2);
    });

    it("routes each edited PTU field into its own model_info key", async () => {
      mockUsePtuCostAttributionEnabled.mockReturnValue(true);
      const user = userEvent.setup();
      renderWithPtuModel();

      expect(await screen.findByRole("button", { name: /edit settings/i })).toBeInTheDocument();
      await user.click(screen.getByRole("button", { name: /edit settings/i }));
      expect(await screen.findByRole("button", { name: /save changes/i })).toBeInTheDocument();

      await user.clear(screen.getByPlaceholderText("e.g. 15"));
      await user.type(screen.getByPlaceholderText("e.g. 15"), "20");
      await user.clear(screen.getByPlaceholderText("e.g. 2.00"));
      await user.type(screen.getByPlaceholderText("e.g. 2.00"), "3.5");

      const from = screen.getByLabelText("PTU Effective From (UTC)");
      const to = screen.getByLabelText("PTU Effective To (UTC)");
      expect(from).toHaveValue("2026-07-01T00:00");
      expect(to).toHaveValue("2026-08-01T00:00");

      fireEvent.change(to, { target: { value: "2026-10-03T02:00:00" } });
      fireEvent.change(from, { target: { value: "2026-09-02T01:00:00" } });

      await user.click(screen.getByRole("button", { name: /save changes/i }));
      await waitFor(() => expect(mockModelPatchUpdateCall).toHaveBeenCalled());

      const modelInfo = mockModelPatchUpdateCall.mock.calls[0][1].model_info;
      expect(modelInfo.ptu_count).toBe(20);
      expect(modelInfo.cost_per_ptu_per_hour).toBe(3.5);
      expect(modelInfo.ptu_effective_from).toBe("2026-09-02T01:00:00.000Z");
      expect(modelInfo.ptu_effective_to).toBe("2026-10-03T02:00:00.000Z");
    });
  });

  it("blocks the save when the LiteLLM Params box does not hold valid JSON", async () => {
    const user = userEvent.setup();
    render(<ModelInfoView {...DEFAULT_ADMIN_PROPS} />, { wrapper });

    expect(await screen.findByRole("button", { name: /edit settings/i })).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: /edit settings/i }));

    const extraParams = screen
      .getAllByRole("textbox")
      .find(
        (input) =>
          input.tagName === "TEXTAREA" && (input as HTMLTextAreaElement).value.includes('"custom_llm_provider"'),
      ) as HTMLTextAreaElement;
    await user.clear(extraParams);
    await user.paste("{not json");

    await user.click(screen.getByRole("button", { name: /save changes/i }));

    expect(await screen.findByText("Please enter valid JSON")).toBeInTheDocument();
    expect(mockModelPatchUpdateCall).not.toHaveBeenCalled();
  });

  it("should not include input_cost_per_token or output_cost_per_token in update payload when user does not touch cost fields", async () => {
    // Regression: editing a model without touching cost fields used to inject
    // input_cost_per_token: 0 and output_cost_per_token: 0 into litellm_params,
    // overriding the built-in pricing table from model_prices_and_context_window.json.
    const user = userEvent.setup();
    render(<ModelInfoView {...DEFAULT_ADMIN_PROPS} />, { wrapper });

    await waitFor(() => {
      expect(screen.getByRole("button", { name: /edit settings/i })).toBeInTheDocument();
    });

    await user.click(screen.getByRole("button", { name: /edit settings/i }));

    await waitFor(() => {
      expect(screen.getByRole("button", { name: /save changes/i })).toBeInTheDocument();
    });

    await user.click(screen.getByRole("button", { name: /save changes/i }));

    await waitFor(() => {
      expect(mockModelPatchUpdateCall).toHaveBeenCalled();
    });

    const updatePayload = mockModelPatchUpdateCall.mock.calls[0][1];
    expect(updatePayload.litellm_params).not.toHaveProperty("input_cost_per_token");
    expect(updatePayload.litellm_params).not.toHaveProperty("output_cost_per_token");
  });

  it("never re-sends a masked secret on save (regression: masked auth value must not overwrite the real secret)", async () => {
    // /model/info redacts secrets by masking (e.g. "azur****BBCC"), not removing them.
    // A plain save re-PATCHes the whole litellm_params blob; if the masked value were
    // sent, the backend would encrypt the asterisks over the real azure_ad_token and
    // silently destroy the credential. The edit form must strip masked values entirely.
    const maskedSecret = "azur********************************************BBCC";
    const maskedModelData = {
      ...defaultModelData,
      litellm_params: {
        model: "azure/gpt-4o",
        api_base: "https://example-az.openai.azure.com",
        custom_llm_provider: "azure",
        azure_ad_token: maskedSecret,
      },
    };
    mockUseModelsInfo.mockReturnValue({
      data: { data: [maskedModelData] },
      isLoading: false,
      error: null,
    });
    mockModelInfoV1Call.mockResolvedValue({ data: [maskedModelData] });

    const user = userEvent.setup();
    render(<ModelInfoView {...DEFAULT_ADMIN_PROPS} />, { wrapper });

    await waitFor(() => {
      expect(screen.getByRole("button", { name: /edit settings/i })).toBeInTheDocument();
    });
    await user.click(screen.getByRole("button", { name: /edit settings/i }));

    await waitFor(() => {
      expect(screen.getByRole("button", { name: /save changes/i })).toBeInTheDocument();
    });
    await user.click(screen.getByRole("button", { name: /save changes/i }));

    await waitFor(() => {
      expect(mockModelPatchUpdateCall).toHaveBeenCalled();
    });

    const updatePayload = mockModelPatchUpdateCall.mock.calls[0][1];
    expect(updatePayload.litellm_params.azure_ad_token).not.toBe(maskedSecret);
    // No masked value may appear anywhere in the outbound params.
    expect(JSON.stringify(updatePayload.litellm_params)).not.toContain("**");
  });

  it("should display health check model field for wildcard models", async () => {
    const wildcardModelData = {
      ...defaultModelData,
      litellm_params: {
        ...defaultModelData.litellm_params,
        model: "openai/gpt-4*",
      },
    };

    mockUseModelsInfo.mockReturnValue({
      data: {
        data: [wildcardModelData],
      },
      isLoading: false,
      error: null,
    });

    render(<ModelInfoView {...DEFAULT_ADMIN_PROPS} />, { wrapper });
    await waitFor(() => {
      expect(screen.getByText("Health Check Model")).toBeInTheDocument();
    });
  });

  it("should not display health check model field for non-wildcard models", async () => {
    render(<ModelInfoView {...DEFAULT_ADMIN_PROPS} />, { wrapper });
    await waitFor(() => {
      expect(screen.getByText("Model Settings")).toBeInTheDocument();
      expect(screen.queryByText("Health Check Model")).not.toBeInTheDocument();
    });
  });

  it("should display edit auto router button for auto router models", async () => {
    const autoRouterModelData = {
      ...defaultModelData,
      litellm_params: {
        ...defaultModelData.litellm_params,
        auto_router_config: {},
      },
    };

    mockUseModelsInfo.mockReturnValue({
      data: {
        data: [autoRouterModelData],
      },
      isLoading: false,
      error: null,
    });

    render(<ModelInfoView {...DEFAULT_ADMIN_PROPS} />, { wrapper });
    await waitFor(() => {
      expect(screen.getByRole("button", { name: /edit auto router/i })).toBeInTheDocument();
    });
  });

  it("does not offer Test Connection for semantic auto router models (no tier-based test exists yet)", async () => {
    const semanticAutoRouterModelData = {
      ...defaultModelData,
      litellm_params: {
        ...defaultModelData.litellm_params,
        auto_router_config: {},
      },
    };

    mockUseModelsInfo.mockReturnValue({
      data: {
        data: [semanticAutoRouterModelData],
      },
      isLoading: false,
      error: null,
    });

    render(<ModelInfoView {...DEFAULT_ADMIN_PROPS} />, { wrapper });
    await waitFor(() => {
      expect(screen.getByText("Model Settings")).toBeInTheDocument();
    });
    expect(screen.queryByTestId("test-connection-button")).not.toBeInTheDocument();
  });

  it("tests each complexity tier's model group instead of sending the router pseudo-model to /health/test_connection (regression: raw test previously threw 'Unmapped LLM provider... model=complexity_router')", async () => {
    const complexityRouterModelData = {
      ...defaultModelData,
      litellm_params: {
        ...defaultModelData.litellm_params,
        model: "auto_router/complexity_router",
        complexity_router_config: {
          tiers: { SIMPLE: ["gpt-4o-mini"], MEDIUM: ["gpt-4o"], COMPLEX: [], REASONING: [] },
        },
      },
    };

    mockUseModelsInfo.mockReturnValue({
      data: {
        data: [complexityRouterModelData],
      },
      isLoading: false,
      error: null,
    });
    mockTestModelGroupConnection.mockResolvedValue({ status: "success" });

    render(<ModelInfoView {...DEFAULT_ADMIN_PROPS} />, { wrapper });
    const testConnectionButton = await screen.findByTestId("test-connection-button");
    await userEvent.click(testConnectionButton);

    await waitFor(() => {
      expect(mockTestModelGroupConnection).toHaveBeenCalledWith("test-token", "gpt-4o-mini", "chat");
    });
    expect(mockTestModelGroupConnection).toHaveBeenCalledWith("test-token", "gpt-4o", "chat");
    expect(mockTestConnectionRequest).not.toHaveBeenCalled();
  });

  it("also tests the configured default model when an unconfigured tier would fall back to it in production", async () => {
    const complexityRouterModelData = {
      ...defaultModelData,
      litellm_params: {
        ...defaultModelData.litellm_params,
        model: "auto_router/complexity_router",
        complexity_router_config: {
          tiers: { SIMPLE: ["gpt-4o-mini"], MEDIUM: [], COMPLEX: [], REASONING: [] },
        },
        complexity_router_default_model: "gpt-4o",
      },
    };

    mockUseModelsInfo.mockReturnValue({
      data: {
        data: [complexityRouterModelData],
      },
      isLoading: false,
      error: null,
    });
    mockTestModelGroupConnection.mockResolvedValue({ status: "success" });

    render(<ModelInfoView {...DEFAULT_ADMIN_PROPS} />, { wrapper });
    const testConnectionButton = await screen.findByTestId("test-connection-button");
    await userEvent.click(testConnectionButton);

    await waitFor(() => {
      expect(mockTestModelGroupConnection).toHaveBeenCalledWith("test-token", "gpt-4o-mini", "chat");
    });
    expect(mockTestModelGroupConnection).toHaveBeenCalledWith("test-token", "gpt-4o", "chat");
  });

  it("does not duplicate the default model as a test target when it is already covered by a configured tier", async () => {
    const complexityRouterModelData = {
      ...defaultModelData,
      litellm_params: {
        ...defaultModelData.litellm_params,
        model: "auto_router/complexity_router",
        complexity_router_config: {
          tiers: { SIMPLE: ["gpt-4o-mini"], MEDIUM: ["gpt-4o"], COMPLEX: [], REASONING: [] },
        },
        complexity_router_default_model: "gpt-4o",
      },
    };

    mockUseModelsInfo.mockReturnValue({
      data: {
        data: [complexityRouterModelData],
      },
      isLoading: false,
      error: null,
    });
    mockTestModelGroupConnection.mockResolvedValue({ status: "success" });

    render(<ModelInfoView {...DEFAULT_ADMIN_PROPS} />, { wrapper });
    const testConnectionButton = await screen.findByTestId("test-connection-button");
    await userEvent.click(testConnectionButton);

    await waitFor(() => {
      expect(mockTestModelGroupConnection).toHaveBeenCalledWith("test-token", "gpt-4o", "chat");
    });
    expect(mockTestModelGroupConnection).toHaveBeenCalledTimes(2);
  });

  it("warns instead of erroring when no complexity tiers are configured to test", async () => {
    const complexityRouterModelData = {
      ...defaultModelData,
      litellm_params: {
        ...defaultModelData.litellm_params,
        model: "auto_router/complexity_router",
        complexity_router_config: {
          tiers: { SIMPLE: [], MEDIUM: [], COMPLEX: [], REASONING: [] },
        },
      },
    };

    mockUseModelsInfo.mockReturnValue({
      data: {
        data: [complexityRouterModelData],
      },
      isLoading: false,
      error: null,
    });

    render(<ModelInfoView {...DEFAULT_ADMIN_PROPS} />, { wrapper });
    const testConnectionButton = await screen.findByTestId("test-connection-button");
    await userEvent.click(testConnectionButton);

    await waitFor(() => {
      expect(mockToast.warning).toHaveBeenCalledWith(
        "No complexity tiers are configured yet, so there is nothing to test.",
      );
    });
    expect(mockTestModelGroupConnection).not.toHaveBeenCalled();
  });

  // Bugbot finding on #36615: complexity_router_config.default_model is a UI-only bookkeeping
  // marker — init_complexity_router_deployment (litellm/router.py) never reads it, falling back
  // to tier-derivation instead when litellm_params.complexity_router_default_model is absent.
  // Probing the blob field here would test a model the running router never calls.
  it("ignores an unused config blob pin when litellm_params has no default, matching the backend's own tier-derivation fallback", async () => {
    const complexityRouterModelData = {
      ...defaultModelData,
      litellm_params: {
        ...defaultModelData.litellm_params,
        model: "auto_router/complexity_router",
        complexity_router_config: {
          tiers: { SIMPLE: ["gpt-4o-mini"], MEDIUM: [], COMPLEX: [], REASONING: [] },
          default_model: "unused-blob-pin",
        },
        // no complexity_router_default_model
      },
    };

    mockUseModelsInfo.mockReturnValue({
      data: {
        data: [complexityRouterModelData],
      },
      isLoading: false,
      error: null,
    });
    mockTestModelGroupConnection.mockResolvedValue({ status: "success" });

    render(<ModelInfoView {...DEFAULT_ADMIN_PROPS} />, { wrapper });
    const testConnectionButton = await screen.findByTestId("test-connection-button");
    await userEvent.click(testConnectionButton);

    await waitFor(() => {
      expect(mockTestModelGroupConnection).toHaveBeenCalledWith("test-token", "gpt-4o-mini", "chat");
    });
    expect(mockTestModelGroupConnection).not.toHaveBeenCalledWith("test-token", "unused-blob-pin", "chat");
    expect(mockTestModelGroupConnection).toHaveBeenCalledTimes(1);
  });

  it("should display model access groups field", async () => {
    render(<ModelInfoView {...DEFAULT_ADMIN_PROPS} />, { wrapper });
    await waitFor(() => {
      expect(screen.getByText("Model Access Groups")).toBeInTheDocument();
    });
  });

  it("should display guardrails field", async () => {
    render(<ModelInfoView {...DEFAULT_ADMIN_PROPS} />, { wrapper });
    await waitFor(() => {
      expect(screen.getByText("Guardrails")).toBeInTheDocument();
    });
  });

  it("should display pricing information", async () => {
    render(<ModelInfoView {...DEFAULT_ADMIN_PROPS} />, { wrapper });
    await waitFor(() => {
      expect(screen.getByText(/Input:/)).toBeInTheDocument();
      expect(screen.getByText(/Output:/)).toBeInTheDocument();
    });
  });

  it("should display created at and created by information", async () => {
    render(<ModelInfoView {...DEFAULT_ADMIN_PROPS} />, { wrapper });
    await waitFor(() => {
      expect(screen.getByText(/Created At/)).toBeInTheDocument();
      expect(screen.getByText(/Created By/)).toBeInTheDocument();
    });
  });

  it("renders the provider card logo from the bundled provider map", async () => {
    render(<ModelInfoView {...DEFAULT_ADMIN_PROPS} />, { wrapper });

    const logo = await screen.findByAltText("openai logo");
    expect(logo).toHaveAttribute("src", expect.stringContaining("openai_small"));
  });

  it("renders a letter avatar instead of an img for an unknown provider slug", async () => {
    mockUseModelsInfo.mockReturnValue({
      data: {
        data: [
          {
            ...defaultModelData,
            litellm_params: {
              ...defaultModelData.litellm_params,
              custom_llm_provider: "zzz-internal",
            },
          },
        ],
      },
      isLoading: false,
      error: null,
    });

    render(<ModelInfoView {...DEFAULT_ADMIN_PROPS} />, { wrapper });

    await waitFor(() => {
      expect(screen.getAllByText("zzz-internal").length).toBeGreaterThan(0);
    });
    expect(screen.queryByAltText("zzz-internal logo")).not.toBeInTheDocument();
    expect(screen.getByText("z")).toBeInTheDocument();
  });

  // EditAutoRouterModal only speaks complexity and semantic. Offering it for an adaptive or
  // quality router lets a save write auto_router_config onto a row that stores its settings
  // elsewhere. These rows stay reachable from Health Status and direct ?model= links even
  // though the Models table now excludes auto-routers, so the button itself has to be gated.
  describe("Edit Auto Router affordance", () => {
    const withRouter = (litellmParams: Record<string, unknown>) => {
      mockUseModelsInfo.mockReturnValue({
        data: { data: [{ ...defaultModelData, litellm_params: { ...litellmParams } }] },
        isLoading: false,
        error: null,
      });
    };

    it.each([
      ["auto_router/adaptive_router", "adaptive"],
      ["auto_router/quality_router", "quality"],
    ])("is absent for a %s router", async (model) => {
      withRouter({ model });
      render(<ModelInfoView {...DEFAULT_ADMIN_PROPS} />, { wrapper });

      expect(await screen.findByText("GPT-4")).toBeInTheDocument();
      expect(screen.queryByRole("button", { name: /edit auto router/i })).not.toBeInTheDocument();
    });

    it("is present for a complexity router, which the modal does understand", async () => {
      withRouter({ model: "auto_router/complexity_router", complexity_router_config: { tiers: {} } });
      render(<ModelInfoView {...DEFAULT_ADMIN_PROPS} />, { wrapper });

      expect(await screen.findByRole("button", { name: /edit auto router/i })).toBeInTheDocument();
    });
  });

  // An auto router has no upstream credential, so the credential actions are meaningless for
  // every strategy, and the destructive action should name what it actually removes.
  describe("auto-router header actions", () => {
    const withParams = (litellmParams: Record<string, unknown>) => {
      mockUseModelsInfo.mockReturnValue({
        data: { data: [{ ...defaultModelData, litellm_params: { ...litellmParams } }] },
        isLoading: false,
        error: null,
      });
    };

    it.each([
      ["auto_router/complexity_router"],
      ["auto_router/adaptive_router"],
      ["auto_router/quality_router"],
      ["auto_router/my-semantic"],
    ])("hides the credential actions and renames delete for %s", async (model) => {
      withParams({ model });
      render(<ModelInfoView {...DEFAULT_ADMIN_PROPS} />, { wrapper });

      expect(await screen.findByTestId("delete-model-button")).toHaveTextContent("Delete Auto-Router");
      expect(screen.queryByTestId("update-api-key-button")).not.toBeInTheDocument();
      expect(screen.queryByTestId("reuse-credentials-button")).not.toBeInTheDocument();
    });

    it("keeps both credential actions and the Delete Model label for an ordinary model", async () => {
      withParams({ model: "gpt-4", api_base: "https://api.openai.com/v1" });
      render(<ModelInfoView {...DEFAULT_ADMIN_PROPS} />, { wrapper });

      expect(await screen.findByTestId("delete-model-button")).toHaveTextContent("Delete Model");
      expect(screen.getByTestId("update-api-key-button")).toBeInTheDocument();
      expect(screen.getByTestId("reuse-credentials-button")).toBeInTheDocument();
    });

    it.each([["auto_router/adaptive_router"], ["auto_router/quality_router"]])(
      "offers no Test Connection for %s, whose targets it cannot build",
      async (model) => {
        withParams({ model });
        render(<ModelInfoView {...DEFAULT_ADMIN_PROPS} />, { wrapper });

        await screen.findByTestId("delete-model-button");
        expect(screen.queryByTestId("test-connection-button")).not.toBeInTheDocument();
      },
    );

    it("keeps Test Connection for a complexity router", async () => {
      withParams({ model: "auto_router/complexity_router", complexity_router_config: { tiers: {} } });
      render(<ModelInfoView {...DEFAULT_ADMIN_PROPS} />, { wrapper });

      expect(await screen.findByTestId("test-connection-button")).toBeInTheDocument();
    });
  });

  describe("payload parity pins", () => {
    const enterEditMode = async (user: ReturnType<typeof userEvent.setup>) => {
      render(<ModelInfoView {...DEFAULT_ADMIN_PROPS} />, { wrapper });
      expect(await screen.findByRole("button", { name: /edit settings/i })).toBeInTheDocument();
      await user.click(screen.getByRole("button", { name: /edit settings/i }));
      expect(await screen.findByRole("button", { name: /save changes/i })).toBeInTheDocument();
    };

    const save = async (user: ReturnType<typeof userEvent.setup>) => {
      await user.click(screen.getByRole("button", { name: /save changes/i }));
      await waitFor(() => expect(mockModelPatchUpdateCall).toHaveBeenCalled());
      return mockModelPatchUpdateCall.mock.calls[0][1] as {
        model_name: string;
        litellm_params: Record<string, unknown>;
        model_info: Record<string, unknown>;
      };
    };

    it("sends the whole edit payload for an untouched save", async () => {
      const user = userEvent.setup();
      await enterEditMode(user);
      const payload = await save(user);

      expect(payload).toEqual({
        model_name: "GPT-4",
        litellm_params: {
          model: "gpt-4",
          api_base: "https://api.openai.com/v1",
          custom_llm_provider: "openai",
          litellm_credential_name: "selected-credential",
          tags: [],
          guardrails: [],
        },
        model_info: {
          id: "123",
          created_by: "123",
          created_at: "2024-01-01T00:00:00Z",
          db_model: true,
          input_cost_per_token: 0.00003,
          output_cost_per_token: 0.00006,
          access_groups: [],
        },
      });
    });

    it("omits health_check_model for a model that is not a wildcard, whose field never renders", async () => {
      const user = userEvent.setup();
      await enterEditMode(user);
      const payload = await save(user);

      expect(payload.model_info).not.toHaveProperty("health_check_model");
    });

    it("routes each edited field into its own payload key", async () => {
      const user = userEvent.setup();
      await enterEditMode(user);

      await user.clear(screen.getByPlaceholderText("Enter model name"));
      await user.type(screen.getByPlaceholderText("Enter model name"), "renamed-model");
      await user.clear(screen.getByPlaceholderText("Enter LiteLLM model name"));
      await user.type(screen.getByPlaceholderText("Enter LiteLLM model name"), "gpt-4o");
      await user.clear(screen.getByPlaceholderText("Enter API base"));
      await user.type(screen.getByPlaceholderText("Enter API base"), "https://example.test/v1");
      await user.clear(screen.getByPlaceholderText("Enter custom LLM provider"));
      await user.type(screen.getByPlaceholderText("Enter custom LLM provider"), "azure");
      await user.type(screen.getByPlaceholderText("Enter organization"), "org-9");
      await user.type(screen.getByPlaceholderText("Enter TPM"), "111");
      await user.type(screen.getByPlaceholderText("Enter RPM"), "222");
      await user.type(screen.getByPlaceholderText("Enter max retries"), "4");
      await user.type(screen.getByPlaceholderText("Enter timeout"), "33");
      await user.type(screen.getByPlaceholderText("Enter stream timeout"), "44");

      const payload = await save(user);

      expect(payload.model_name).toBe("renamed-model");
      expect(payload.litellm_params).toMatchObject({
        model: "gpt-4o",
        api_base: "https://example.test/v1",
        custom_llm_provider: "azure",
        organization: "org-9",
        tpm: "111",
        rpm: "222",
        max_retries: "4",
        timeout: "33",
        stream_timeout: "44",
      });
    });

    it("routes each edited pricing field into its own payload key", async () => {
      const user = userEvent.setup();
      await enterEditMode(user);

      await user.clear(screen.getByPlaceholderText("Enter output cost"));
      await user.type(screen.getByPlaceholderText("Enter output cost"), "12");
      const [cacheRead, cacheWrite] = screen.getAllByPlaceholderText("Defaults to Input Cost if blank");
      await user.type(cacheRead, "5");
      await user.type(cacheWrite, "9");

      const payload = await save(user);

      expect(payload.litellm_params).toMatchObject({
        output_cost_per_token: 0.000012,
        cache_read_input_token_cost: 0.000005,
        cache_creation_input_token_cost: 0.000009,
      });
    });

    const addTag = async (user: ReturnType<typeof userEvent.setup>, placeholder: string, tag: string) => {
      const input = screen.getByPlaceholderText(placeholder);
      await user.type(input, tag);
      await user.keyboard("{Enter}");
    };

    it("routes each typed collection field into its own payload key", async () => {
      const user = userEvent.setup();
      await enterEditMode(user);

      await addTag(user, "Select existing groups or type to create new ones", "beta-testers");
      await addTag(user, "Select existing guardrails or type to create new ones", "content_filter");
      await addTag(user, "Select existing tags or type to create new ones", "production_tag");

      const payload = await save(user);

      expect(payload.model_info.access_groups).toEqual(["beta-testers"]);
      expect(payload.litellm_params.guardrails).toEqual(["content_filter"]);
      expect(payload.litellm_params.tags).toEqual(["production_tag"]);
    });

    it("sends the edited model info JSON", async () => {
      const user = userEvent.setup();
      await enterEditMode(user);

      const modelInfo = screen.getByPlaceholderText('{"gpt-4": 100, "claude-v1": 200}');
      await user.clear(modelInfo);
      await user.paste('{"id":"123","team_id":"team-7"}');

      const payload = await save(user);

      expect(payload.model_info).toMatchObject({ team_id: "team-7" });
    });

    it("sends the edited LiteLLM extra params", async () => {
      const user = userEvent.setup();
      await enterEditMode(user);

      const extraParams = screen
        .getAllByRole("textbox")
        .find(
          (input) =>
            input.tagName === "TEXTAREA" && (input as HTMLTextAreaElement).value.includes('"custom_llm_provider"'),
        ) as HTMLTextAreaElement;
      await user.clear(extraParams);
      await user.paste('{"drop_params":true}');

      const payload = await save(user);

      expect(payload.litellm_params.drop_params).toBe(true);
    });

    it("sends the credential picked in the selector", async () => {
      mockCredentialListCall.mockResolvedValue({
        credentials: [
          { credential_name: "selected-credential", credential_values: {}, credential_info: {} },
          { credential_name: "other-credential", credential_values: {}, credential_info: {} },
        ],
      } as never);
      const user = userEvent.setup();
      await enterEditMode(user);

      await user.click(await screen.findByText("selected-credential"));
      await user.click(await screen.findByText("other-credential"));

      const payload = await save(user);

      expect(payload.litellm_params.litellm_credential_name).toBe("other-credential");
    });

    it("sends the vector stores picked in the knowledge base selector", async () => {
      const user = userEvent.setup();
      await enterEditMode(user);

      await user.click(screen.getByPlaceholderText("Select knowledge bases (optional)"));
      await user.click(await screen.findByText("Beta (vs-beta)"));
      await user.keyboard("{Escape}");

      const payload = await save(user);

      expect(payload.litellm_params.vector_store_ids).toEqual(["vs-beta"]);
    });

    it("sends the health check model picked for a wildcard deployment", async () => {
      const wildcard = {
        ...defaultModelData,
        litellm_params: { ...defaultModelData.litellm_params, model: "openai/gpt-4*" },
      };
      mockUseModelsInfo.mockReturnValue({ data: { data: [wildcard] }, isLoading: false, error: null });
      mockModelInfoV1Call.mockResolvedValue({ data: [wildcard] });
      mockUseModelHub.mockReturnValue({
        data: { data: [{ model_group: "openai/gpt-4o", providers: ["openai"] }] },
        isLoading: false,
        error: null,
      });
      const user = userEvent.setup();
      await enterEditMode(user);

      await user.click(screen.getByText("Select existing health check model"));
      await user.click(await screen.findByText("openai/gpt-4o"));

      const payload = await save(user);

      expect(payload.model_info.health_check_model).toBe("openai/gpt-4o");
    });

    it("keeps a pricing field in the payload after the operator types a value and restores the original", async () => {
      // antd marks a field touched on change and never clears it, so retyping the seeded value
      // still ships the key. RHF's dirtyFields resets on a value returning to its default, which
      // would silently drop input_cost_per_token here.
      const user = userEvent.setup();
      await enterEditMode(user);

      const inputCost = screen.getByPlaceholderText("Enter input cost") as HTMLInputElement;
      const seeded = inputCost.value;
      expect(seeded).toBe("30");

      await user.clear(inputCost);
      await user.type(inputCost, "7");
      await user.clear(inputCost);
      await user.type(inputCost, seeded);

      const payload = await save(user);

      expect(payload.litellm_params.input_cost_per_token).toBe(0.00003);
      expect(payload.litellm_params.cache_read_input_token_cost).toBe(0.00003);
    });

    it("clears a pricing override with an explicit null once the field is emptied", async () => {
      const user = userEvent.setup();
      await enterEditMode(user);

      await user.clear(screen.getByPlaceholderText("Enter input cost"));
      const payload = await save(user);

      expect(payload.litellm_params.input_cost_per_token).toBeNull();
      expect(payload.litellm_params).not.toHaveProperty("cache_read_input_token_cost");
    });

    describe("cache control injection points", () => {
      const withCachePoints = (points: unknown) => {
        const data = {
          ...defaultModelData,
          litellm_params: { ...defaultModelData.litellm_params, cache_control_injection_points: points },
        };
        mockUseModelsInfo.mockReturnValue({ data: { data: [data] }, isLoading: false, error: null });
        mockModelInfoV1Call.mockResolvedValue({ data: [data] });
      };

      it("omits the key when the deployment has none and the operator leaves the toggle alone", async () => {
        const user = userEvent.setup();
        await enterEditMode(user);
        const payload = await save(user);

        expect(payload.litellm_params).not.toHaveProperty("cache_control_injection_points");
      });

      it("hides the injection point rows until the toggle is on", async () => {
        const user = userEvent.setup();
        await enterEditMode(user);

        expect(screen.queryByRole("button", { name: /add injection point/i })).not.toBeInTheDocument();

        await user.click(screen.getByRole("switch"));

        expect(await screen.findByRole("button", { name: /add injection point/i })).toBeInTheDocument();
      });

      it("round-trips the stored injection points on an untouched save", async () => {
        withCachePoints([{ location: "message", role: "user" }]);
        const user = userEvent.setup();
        await enterEditMode(user);
        const payload = await save(user);

        expect(payload.litellm_params.cache_control_injection_points).toEqual([{ location: "message", role: "user" }]);
      });

      it("drops the stored injection points when the operator turns the toggle off", async () => {
        withCachePoints([{ location: "message", role: "user" }]);
        const user = userEvent.setup();
        await enterEditMode(user);

        await user.click(screen.getByRole("switch"));
        const payload = await save(user);

        expect(payload.litellm_params).not.toHaveProperty("cache_control_injection_points");
      });

      it("adds a typed index as a string, matching what the deployment already stores", async () => {
        withCachePoints([{ location: "message" }]);
        const user = userEvent.setup();
        await enterEditMode(user);

        await user.type(screen.getByPlaceholderText("Optional"), "2");
        const payload = await save(user);

        expect(payload.litellm_params.cache_control_injection_points).toEqual([{ location: "message", index: "2" }]);
      });
    });
  });
});
