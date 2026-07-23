/* @vitest-environment jsdom */
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, render } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import ModelsAndEndpointsLayout from "./layout";

const { mockPush, mockReplace, navState } = vi.hoisted(() => ({
  mockPush: vi.fn(),
  mockReplace: vi.fn(),
  navState: { pathname: "/models-and-endpoints", search: "" },
}));
vi.mock("next/navigation", () => ({
  usePathname: () => navState.pathname,
  useRouter: () => ({ push: mockPush, replace: mockReplace }),
  useSearchParams: () => new URLSearchParams(navState.search),
}));

vi.mock("@/components/networking", () => ({ serverRootPath: "" }));

vi.mock("@/components/molecules/cost_optimization_feedback_banner", () => ({ default: () => null }));
vi.mock("@/components/model_info_view", () => ({
  default: ({ modelId }: { modelId: string }) => <div data-testid="model-info">model:{modelId}</div>,
}));
vi.mock("@/components/team/TeamInfo", () => ({
  default: ({ teamId }: { teamId: string }) => <div data-testid="team-info">team:{teamId}</div>,
}));

const mockUseAuthorized = vi.fn();
vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({ default: () => mockUseAuthorized() }));
vi.mock("@/app/(dashboard)/hooks/teams/useTeams", () => ({ useTeams: () => ({ data: [] }) }));
vi.mock("@/app/(dashboard)/hooks/uiSettings/useUISettings", () => ({
  useUISettings: () => ({ data: { values: {} } }),
}));
vi.mock("@/app/(dashboard)/models-and-endpoints/useModelDashboardData", () => ({
  useModelDashboardData: () => ({
    availableModelGroups: [],
    availableModelAccessGroups: [],
    allModelsOnProxy: [],
    isLoading: false,
  }),
}));

const renderLayout = () => {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: 0 } } });
  return render(
    <QueryClientProvider client={queryClient}>
      <ModelsAndEndpointsLayout>
        <div data-testid="tab-content">CHILD</div>
      </ModelsAndEndpointsLayout>
    </QueryClientProvider>,
  );
};

describe("ModelsAndEndpointsLayout", () => {
  beforeEach(() => {
    navState.pathname = "/models-and-endpoints";
    navState.search = "";
    mockPush.mockClear();
    mockReplace.mockClear();
    mockUseAuthorized.mockReturnValue({
      accessToken: "123",
      token: "123",
      userRole: "Admin",
      userId: "123",
      premiumUser: false,
    });
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (global as any).ResizeObserver = class {
      observe() {}
      unobserve() {}
      disconnect() {}
    };
  });

  it("renders the admin tab bar and the active tab's page content", () => {
    const { getByRole, getByTestId } = renderLayout();
    expect(getByRole("tab", { name: "LLM Credentials" })).toBeInTheDocument();
    expect(getByRole("tab", { name: "Health Status" })).toBeInTheDocument();
    expect(getByTestId("tab-content")).toHaveTextContent("CHILD");
  });

  it("navigates to a tab's path when its tab is clicked", async () => {
    const { getByRole } = renderLayout();
    await act(async () => {
      getByRole("tab", { name: "Health Status" }).click();
    });
    expect(mockPush).toHaveBeenCalledWith(expect.stringMatching(/\/models-and-endpoints\/health\/$/));
  });

  it("redirects to the base models path when the tab path is not permitted for the role", async () => {
    const replaceMock = vi.fn();
    const originalLocation = window.location;
    Object.defineProperty(window, "location", {
      configurable: true,
      value: { replace: replaceMock, assign: vi.fn(), href: "http://localhost/", pathname: "/", search: "" },
    });
    mockUseAuthorized.mockReturnValue({
      accessToken: "123",
      token: "123",
      userRole: "Internal User",
      userId: "123",
      premiumUser: false,
    });
    navState.pathname = "/models-and-endpoints/llm-credentials";
    await act(async () => {
      renderLayout();
    });
    expect(replaceMock).toHaveBeenCalledWith(expect.stringMatching(/\/models-and-endpoints\/$/));
    Object.defineProperty(window, "location", { configurable: true, value: originalLocation });
  });

  it("renders the model detail overlay from ?model and hides the tabs and page content", () => {
    navState.search = "model=abc-123";
    const { getByTestId, queryByTestId, queryByRole } = renderLayout();
    expect(getByTestId("model-info")).toHaveTextContent("model:abc-123");
    expect(queryByTestId("tab-content")).toBeNull();
    expect(queryByRole("tab", { name: "Health Status" })).toBeNull();
  });

  it("renders the team detail overlay from ?team", () => {
    navState.search = "team=team-9";
    const { getByTestId, queryByTestId } = renderLayout();
    expect(getByTestId("team-info")).toHaveTextContent("team:team-9");
    expect(queryByTestId("tab-content")).toBeNull();
  });
});
