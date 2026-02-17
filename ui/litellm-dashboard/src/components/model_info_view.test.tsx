import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import React, { ReactNode } from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import ModelInfoView from "./model_info_view";
import NotificationsManager from "./molecules/notifications_manager";
import * as networking from "./networking";

vi.mock("../../utils/dataUtils", () => ({
  copyToClipboard: vi.fn().mockResolvedValue(true),
}));

vi.mock("./molecules/notifications_manager", () => ({
  default: {
    success: vi.fn(),
    error: vi.fn(),
    info: vi.fn(),
    fromBackend: vi.fn(),
  },
}));

vi.mock("./networking", () => ({
  modelInfoV1Call: vi.fn(),
  credentialGetCall: vi.fn(),
  getGuardrailsList: vi.fn(),
  tagListCall: vi.fn(),
  testConnectionRequest: vi.fn(),
  modelPatchUpdateCall: vi.fn(),
  modelDeleteCall: vi.fn(),
  credentialCreateCall: vi.fn(),
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

const mockNotificationsManager = vi.mocked(NotificationsManager);
const mockModelInfoV1Call = vi.mocked(networking.modelInfoV1Call);
const mockCredentialGetCall = vi.mocked(networking.credentialGetCall);
const mockGetGuardrailsList = vi.mocked(networking.getGuardrailsList);
const mockTagListCall = vi.mocked(networking.tagListCall);
const mockTestConnectionRequest = vi.mocked(networking.testConnectionRequest);
const mockModelPatchUpdateCall = vi.mocked(networking.modelPatchUpdateCall);
const mockModelDeleteCall = vi.mocked(networking.modelDeleteCall);
const mockCredentialCreateCall = vi.mocked(networking.credentialCreateCall);

describe("ModelInfoView", () => {
  let queryClient: QueryClient;

  const defaultModelData = {
    model_name: "GPT-4",
    litellm_params: {
      model: "gpt-4",
      api_base: "https://api.openai.com/v1",
      custom_llm_provider: "openai",
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
      expect(mockNotificationsManager.success).toHaveBeenCalledWith("Connection test successful!");
    });
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
      expect(mockNotificationsManager.error).toHaveBeenCalled();
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
    await user.type(modelNameInput, "Updated Model Name");

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
      expect(mockNotificationsManager.success).toHaveBeenCalledWith("Model settings updated successfully");
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
});
