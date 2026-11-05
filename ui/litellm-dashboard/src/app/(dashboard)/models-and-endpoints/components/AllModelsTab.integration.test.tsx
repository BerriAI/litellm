import * as useAuthorizedModule from "@/app/(dashboard)/hooks/useAuthorized";
import { useQuery } from "@tanstack/react-query";
import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { renderWithProviders, testQueryClient } from "../../../../../tests/test-utils";
import AllModelsTab from "./AllModelsTab";
import ModelDeprecationBanner from "./ModelDeprecationBanner";

const mockModelDeleteCall = vi.fn().mockResolvedValue({});
const mockModelPatchUpdateCall = vi.fn().mockResolvedValue({});
vi.mock("@/components/networking", () => ({
  serverRootPath: "/",
  modelDeleteCall: (...args: unknown[]) => mockModelDeleteCall(...args),
  modelPatchUpdateCall: (...args: unknown[]) => mockModelPatchUpdateCall(...args),
}));

vi.mock("@/components/model_dashboard/ModelSettingsModal/ModelSettingsModal", () => ({
  default: function ModelSettingsModalMock({ isVisible }: { isVisible: boolean }) {
    return isVisible ? <div data-testid="model-settings-modal" /> : null;
  },
}));

let modelsInfoResult: Record<string, unknown> = {};
const mockRefetch = vi.fn();
vi.mock("../../hooks/models/useModels", () => ({
  useModelsInfo: () => ({ ...modelsInfoResult, refetch: mockRefetch }),
}));

vi.mock("../../hooks/models/useModelCostMap", () => ({
  useModelCostMap: () => ({ data: { "gpt-4": { litellm_provider: "openai" } }, isLoading: false, error: null }),
}));

const mockTeams = [{ team_id: "team-1", team_alias: "Engineering" }];
const mockCredentials = [
  { credential_name: "openai-prod", display_name: "Prod OpenAI", credential_values: {}, credential_info: {} },
];
const mockUseCredentials = vi.hoisted(() => vi.fn());
vi.mock("../../hooks/credentials/useCredentials", () => ({
  useCredentials: mockUseCredentials,
}));

vi.mock("../../hooks/teams/useTeams", () => ({
  useTeams: () => ({ data: mockTeams, isLoading: false, error: null, refetch: vi.fn() }),
}));

const BASE_MODEL_INFO = {
  id: "model-1",
  db_model: true,
  created_by: "user-123",
  created_at: "2024-01-01T00:00:00Z",
  updated_at: "2024-01-02T00:00:00Z",
  team_id: "team-1",
  access_groups: [],
};

const makeRow = () => ({
  model_name: "gpt-4",
  litellm_params: { model: "openai/gpt-4", custom_llm_provider: "openai" },
  model_info: BASE_MODEL_INFO,
});

const setModelsInfo = (rows: Record<string, unknown>[]) => {
  modelsInfoResult = {
    data: { data: rows, total_count: rows.length, current_page: 1, total_pages: 1, size: 50 },
    isLoading: false,
    isFetching: false,
    error: null,
  };
};

const MOCK_AUTHORIZED = {
  isLoading: false,
  isAuthorized: true,
  token: "mock-token",
  accessToken: "mock-access-token",
  userId: "user-123",
  userEmail: "test@example.com",
  userRole: "Admin",
  userRoleLabel: "Admin",
  isViewOnly: false,
  premiumUser: true,
  disabledPersonalKeyCreation: false,
  showSSOBanner: false,
};

const mockSetSelectedModelGroup = vi.fn();
const mockSetSelectedModelId = vi.fn();
const mockSetSelectedTeamId = vi.fn();

const defaultProps = {
  selectedModelGroup: "all",
  setSelectedModelGroup: mockSetSelectedModelGroup,
  availableModelGroups: ["gpt-4", "gpt-3.5-turbo"],
  availableModelAccessGroups: ["sales-team"],
  setSelectedModelId: mockSetSelectedModelId,
  setSelectedTeamId: mockSetSelectedTeamId,
};

const TEST_TODAY = new Date(2026, 10, 4);

const deprecatedResponse = {
  checked_at: "2026-06-01T00:00:00Z",
  warn_within_days: 30,
  deprecated: [
    {
      model_name: "old-model",
      litellm_provider: null,
      deprecation_date: "2026-06-01",
      days_until_deprecation: -1,
      status: "deprecated",
      successor_model: null,
    },
  ],
  imminent: [],
  upcoming: [],
};

const currentResponse = {
  checked_at: "2026-06-02T00:00:00Z",
  warn_within_days: 30,
  deprecated: [],
  imminent: [
    {
      model_name: "current-model",
      litellm_provider: null,
      deprecation_date: "2026-11-30",
      days_until_deprecation: 26,
      status: "imminent",
      successor_model: null,
    },
  ],
  upcoming: [],
};

const stubDeprecationResponses = () => {
  const fetchMock = vi
    .fn<typeof fetch>()
    .mockImplementationOnce(async (input) => {
      const requestUrl = input instanceof Request ? input.url : String(input);
      expect(new URL(requestUrl, "http://localhost").pathname).toBe("/model/deprecations");
      return Response.json(deprecatedResponse);
    })
    .mockImplementationOnce(async (input) => {
      const requestUrl = input instanceof Request ? input.url : String(input);
      expect(new URL(requestUrl, "http://localhost").pathname).toBe("/model/deprecations");
      return Response.json(currentResponse);
    });
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
};

const renderModelsWithDeprecationBanner = () =>
  renderWithProviders(
    <>
      <AllModelsTab {...defaultProps} />
      <ModelDeprecationBanner today={TEST_TODAY} />
    </>,
  );

const ModelListQueryProbe = ({ queryFn }: { queryFn: () => Promise<string> }) => {
  useQuery({ queryKey: ["models", "list"], queryFn, staleTime: Infinity });
  return null;
};

describe("AllModelsTab", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    testQueryClient.clear();
    setModelsInfo([makeRow()]);
    vi.spyOn(useAuthorizedModule, "default").mockReturnValue(MOCK_AUTHORIZED);
    mockUseCredentials.mockImplementation(({ enabled = true }: { enabled?: boolean } = {}) => ({
      data: enabled ? { credentials: mockCredentials } : undefined,
      isLoading: false,
    }));
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("refreshes deprecation banner data after deleting a model", async () => {
    const fetchMock = stubDeprecationResponses();
    const user = userEvent.setup();
    renderModelsWithDeprecationBanner();

    expect(await screen.findByText("old-model was deprecated on 2026-06-01")).toBeInTheDocument();
    await user.click(await screen.findByTestId("model-delete-model-1"));
    await user.click(await screen.findByRole("button", { name: /^delete$/i }));
    expect(await screen.findByText("current-model deprecates on 2026-11-30, in 26 days")).toBeInTheDocument();
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(2));
    expect(screen.queryByText("old-model was deprecated on 2026-06-01")).not.toBeInTheDocument();
  });

  it("refreshes deprecation banner data after pausing a model", async () => {
    const fetchMock = stubDeprecationResponses();
    const user = userEvent.setup();
    renderModelsWithDeprecationBanner();

    expect(await screen.findByText("old-model was deprecated on 2026-06-01")).toBeInTheDocument();
    await user.click(await screen.findByTestId("model-pause-toggle-model-1"));
    expect(await screen.findByText("current-model deprecates on 2026-11-30, in 26 days")).toBeInTheDocument();
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(2));
    expect(screen.queryByText("old-model was deprecated on 2026-06-01")).not.toBeInTheDocument();
  });

  it("refreshes the table and deprecations once without refetching another models list observer", async () => {
    const fetchMock = stubDeprecationResponses();
    const observerQueryFn = vi.fn<() => Promise<string>>().mockResolvedValue("models");
    const user = userEvent.setup();
    renderWithProviders(
      <>
        <AllModelsTab {...defaultProps} />
        <ModelDeprecationBanner today={TEST_TODAY} />
        <ModelListQueryProbe queryFn={observerQueryFn} />
      </>,
    );

    expect(await screen.findByText("old-model was deprecated on 2026-06-01")).toBeInTheDocument();
    await waitFor(() => expect(observerQueryFn).toHaveBeenCalledTimes(1));
    mockRefetch.mockClear();
    fetchMock.mockClear();

    await user.click(screen.getByTestId("datatable-refresh"));

    expect(await screen.findByText("current-model deprecates on 2026-11-30, in 26 days")).toBeInTheDocument();
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1));
    expect(observerQueryFn).toHaveBeenCalledTimes(1);
    expect(mockRefetch).toHaveBeenCalledTimes(1);
  });
});
