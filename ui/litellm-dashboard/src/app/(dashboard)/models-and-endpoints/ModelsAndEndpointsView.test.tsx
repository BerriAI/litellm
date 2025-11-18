import { render, waitFor, screen } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach, beforeAll } from "vitest";
import ModelsAndEndpointsView from "./ModelsAndEndpointsView";
import * as useAuthorizedModule from "@/app/(dashboard)/hooks/useAuthorized";
import * as useTeamsModule from "@/app/(dashboard)/hooks/useTeams";

global.ResizeObserver = vi.fn().mockImplementation(() => ({
  observe: vi.fn(),
  unobserve: vi.fn(),
  disconnect: vi.fn(),
}));

const mockUseAuthorized = {
  token: "mock-token",
  accessToken: "mock-access-token",
  userId: "user-123",
  userEmail: "test@example.com",
  userRole: "Admin",
  premiumUser: true,
  disabledPersonalKeyCreation: false,
  showSSOBanner: false,
};

beforeAll(() => {
  vi.spyOn(useAuthorizedModule, "default").mockReturnValue(mockUseAuthorized);
  vi.spyOn(useTeamsModule, "default").mockReturnValue({
    teams: [],
    setTeams: vi.fn(),
  });
});

vi.mock("@/components/networking", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/components/networking")>();
  return {
    ...actual,
    modelInfoCall: vi.fn().mockResolvedValue({
      data: [
        {
          model_name: "gpt-4",
          litellm_params: {
            model: "gpt-4",
            custom_llm_provider: "openai",
          },
          model_info: {
            id: "model-1",
            access_groups: [],
          },
        },
      ],
    }),
    modelProviderMap: vi.fn().mockResolvedValue({
      "gpt-4": {
        litellm_provider: "openai",
      },
    }),
    modelSettingsCall: vi.fn().mockResolvedValue([]),
    credentialListCall: vi.fn().mockResolvedValue({
      credentials: [],
    }),
    modelMetricsCall: vi.fn().mockResolvedValue({
      data: [],
      all_api_bases: [],
    }),
    streamingModelMetricsCall: vi.fn().mockResolvedValue({
      data: [],
      all_api_bases: [],
    }),
    modelExceptionsCall: vi.fn().mockResolvedValue({
      data: [],
      exception_types: [],
    }),
    modelMetricsSlowResponsesCall: vi.fn().mockResolvedValue([]),
    getCallbacksCall: vi.fn().mockResolvedValue({
      router_settings: {
        model_group_retry_policy: {},
        retry_policy: {},
        num_retries: 0,
        model_group_alias: {},
      },
    }),
    setCallbacksCall: vi.fn().mockResolvedValue({}),
    adminGlobalActivityExceptions: vi.fn().mockResolvedValue({
      sum_num_rate_limit_exceptions: 0,
      daily_data: [],
    }),
    adminGlobalActivityExceptionsPerDeployment: vi.fn().mockResolvedValue([]),
    allEndUsersCall: vi.fn().mockResolvedValue([]),
    modelAvailableCall: vi.fn().mockResolvedValue({
      data: [],
    }),
    getPassThroughEndpointsCall: vi.fn().mockResolvedValue({
      data: [],
    }),
  };
});

describe("ModelsAndEndpointsView", () => {
  const defaultProps = {
    accessToken: "test-access-token",
    token: "test-token",
    userRole: "Admin",
    userID: "test-user-id",
    modelData: {
      data: [
        {
          model_name: "gpt-4",
          litellm_params: {
            model: "gpt-4",
            custom_llm_provider: "openai",
          },
          model_info: {
            id: "model-1",
            access_groups: [],
          },
        },
      ],
    },
    keys: [],
    setModelData: vi.fn(),
    premiumUser: true,
    teams: [],
  };

  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("should render the component successfully", async () => {
    const { container } = render(<ModelsAndEndpointsView {...defaultProps} />);

    await waitFor(() => {
      expect(container).toBeTruthy();
    });

    expect(screen.getByText("Model Management")).toBeInTheDocument();
  });

  it("should render tabs", async () => {
    render(<ModelsAndEndpointsView {...defaultProps} />);
    await waitFor(() => {
      expect(screen.getByText("Model Management")).toBeInTheDocument();
    });

    const allModelsTabs = screen.getAllByRole("tab", { name: /All Models/i });
    expect(allModelsTabs.length).toBeGreaterThan(0);

    const addModelTabs = screen.getAllByRole("tab", { name: /Add Model/i });
    expect(addModelTabs.length).toBeGreaterThan(0);

    expect(screen.getByRole("tab", { name: /LLM Credentials/i })).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: /Pass-Through Endpoints/i })).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: /Health Status/i })).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: /Model Analytics/i })).toBeInTheDocument();
  });
});
