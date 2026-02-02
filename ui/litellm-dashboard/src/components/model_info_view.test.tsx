import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, waitFor } from "@testing-library/react";
import React, { ReactNode } from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import ModelInfoView from "./model_info_view";

vi.mock("../../utils/dataUtils", () => ({
  copyToClipboard: vi.fn().mockResolvedValue(true),
}));

vi.mock("./networking", () => ({
  modelInfoV1Call: vi.fn().mockResolvedValue({
    data: [
      {
        model_name: "GPT-4",
        litellm_params: {
          model: "gpt-4",
          api_base: "https://api.openai.com/v1",
          custom_llm_provider: "openai",
        },
        model_info: {
          id: "123",
          created_by: "123",
          db_model: true,
          input_cost_per_token: 0.00003,
          output_cost_per_token: 0.00006,
        },
      },
    ],
  }),
  credentialGetCall: vi.fn().mockResolvedValue({
    credential_name: "test-credential",
    credential_values: {},
    credential_info: {},
  }),
  getGuardrailsList: vi.fn().mockResolvedValue({
    guardrails: [{ guardrail_name: "content_filter" }, { guardrail_name: "toxicity_filter" }],
  }),
  tagListCall: vi.fn().mockResolvedValue({
    test_tag: {
      name: "test_tag",
      description: "A test tag",
    },
    production_tag: {
      name: "production_tag",
      description: "Production ready models",
    },
  }),
  testConnectionRequest: vi.fn().mockResolvedValue({
    status: "success",
  }),
  modelPatchUpdateCall: vi.fn().mockResolvedValue({}),
  modelDeleteCall: vi.fn().mockResolvedValue({}),
}));

// Mock the useModelsInfo hook since it uses React Query
const mockUseModelsInfo = vi.fn();
const mockUseModelHub = vi.fn();

vi.mock("@/app/(dashboard)/hooks/models/useModels", () => ({
  useModelsInfo: (...args: any[]) => mockUseModelsInfo(...args),
  useModelHub: (...args: any[]) => mockUseModelHub(...args),
}));

// Mock the useModelCostMap hook
const mockUseModelCostMap = vi.fn();
vi.mock("@/app/(dashboard)/hooks/models/useModelCostMap", () => ({
  useModelCostMap: (...args: any[]) => mockUseModelCostMap(...args),
}));

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

  beforeEach(() => {
    queryClient = new QueryClient({
      defaultOptions: {
        queries: {
          retry: false,
        },
      },
    });
    vi.clearAllMocks();

    // Set up default mocks
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
  });

  const wrapper = ({ children }: { children: ReactNode }) =>
    React.createElement(QueryClientProvider, { client: queryClient }, children);

  const DEFAULT_ADMIN_PROPS = {
    modelId: "123",
    onClose: () => {},
    accessToken: "123",
    userID: "123",
    userRole: "Admin",
    onModelUpdate: () => {},
    modelAccessGroups: [],
  };

  describe("Edit Model", () => {
    it("should render the model info view", async () => {
      const { getByText } = render(<ModelInfoView {...DEFAULT_ADMIN_PROPS} />, { wrapper });
      await waitFor(() => {
        expect(getByText("Model Settings")).toBeInTheDocument();
      });
    });

    it("should not render an edit settings button if the model is not a DB model", async () => {
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

      const { queryByText } = render(<ModelInfoView {...DEFAULT_ADMIN_PROPS} />, { wrapper });
      await waitFor(() => {
        expect(queryByText("Edit Settings")).not.toBeInTheDocument();
      });
    });

    it("should render tags in the edit model", async () => {
      const { getByText } = render(<ModelInfoView {...DEFAULT_ADMIN_PROPS} />, { wrapper });
      await waitFor(() => {
        expect(getByText("Tags")).toBeInTheDocument();
      });
    });

    it("should render the litellm params in the edit model", async () => {
      const { getByText } = render(<ModelInfoView {...DEFAULT_ADMIN_PROPS} />, { wrapper });
      await waitFor(() => {
        expect(getByText("LiteLLM Params")).toBeInTheDocument();
      });
    });
  });

  it("should render a test connection button", async () => {
    const { getByTestId } = render(<ModelInfoView {...DEFAULT_ADMIN_PROPS} />, { wrapper });
    await waitFor(() => {
      expect(getByTestId("test-connection-button")).toBeInTheDocument();
    });
  });

  it("should render a reuse credentials button", async () => {
    const { getByTestId } = render(<ModelInfoView {...DEFAULT_ADMIN_PROPS} />, { wrapper });
    await waitFor(() => {
      expect(getByTestId("reuse-credentials-button")).toBeInTheDocument();
    });
  });

  it("should render a delete model button", async () => {
    const { getByTestId } = render(<ModelInfoView {...DEFAULT_ADMIN_PROPS} />, { wrapper });
    await waitFor(() => {
      expect(getByTestId("delete-model-button")).toBeInTheDocument();
    });
  });

  it("should render a disabled delete model button if the model is not a DB model", async () => {
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

    const { getByTestId } = render(<ModelInfoView {...DEFAULT_ADMIN_PROPS} />, { wrapper });
    await waitFor(() => {
      expect(getByTestId("delete-model-button")).toBeDisabled();
    });
  });

  it("should render a disabled delete model button if the user is not an admin and model is not created by the user", async () => {
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

    const NON_CREATED_BY_USER_ADMIN_PROPS = {
      ...DEFAULT_ADMIN_PROPS,
      userRole: "User",
    };

    const { getByTestId } = render(<ModelInfoView {...NON_CREATED_BY_USER_ADMIN_PROPS} />, { wrapper });
    await waitFor(() => {
      expect(getByTestId("delete-model-button")).toBeDisabled();
    });
  });

  it("should render health check model field for wildcard routes", async () => {
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

    const { getByText } = render(<ModelInfoView {...DEFAULT_ADMIN_PROPS} />, { wrapper });
    await waitFor(() => {
      expect(getByText("Model Settings")).toBeInTheDocument();
    });
    await waitFor(() => {
      expect(getByText("Health Check Model")).toBeInTheDocument();
    });
  });

  it("should not render health check model field for non-wildcard routes", async () => {
    const { queryByText } = render(<ModelInfoView {...DEFAULT_ADMIN_PROPS} />, { wrapper });
    await waitFor(() => {
      expect(queryByText("Model Settings")).toBeInTheDocument();
    });
    await waitFor(() => {
      expect(queryByText("Health Check Model")).not.toBeInTheDocument();
    });
  });

  describe("View Model", () => {
    it("should render the model info view", async () => {
      const { getByText } = render(<ModelInfoView {...DEFAULT_ADMIN_PROPS} />, { wrapper });
      await waitFor(() => {
        expect(getByText("Model Settings")).toBeInTheDocument();
      });
    });

    it("should render tags in the view model", async () => {
      const { getByText } = render(<ModelInfoView {...DEFAULT_ADMIN_PROPS} />, { wrapper });
      await waitFor(() => {
        expect(getByText("Tags")).toBeInTheDocument();
      });
    });
  });
});
