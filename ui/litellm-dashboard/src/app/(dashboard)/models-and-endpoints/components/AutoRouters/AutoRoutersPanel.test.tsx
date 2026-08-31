import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { renderWithProviders, screen, testQueryClient, waitFor } from "@/../tests/test-utils";

import { AutoRoutersPanel } from "./AutoRoutersPanel";

const { modelInfoCall, modelDeleteCall } = vi.hoisted(() => ({
  modelInfoCall: vi.fn(),
  modelDeleteCall: vi.fn().mockResolvedValue({}),
}));

vi.mock("@/components/networking", () => ({
  modelInfoCall,
  modelDeleteCall,
  modelHubCall: vi.fn(),
  modelAvailableCall: vi.fn().mockResolvedValue({ data: [] }),
}));

vi.mock("@/components/llm_calls/fetch_models", () => ({
  fetchAvailableModels: vi.fn().mockResolvedValue([]),
}));

const { openModel } = vi.hoisted(() => ({ openModel: vi.fn() }));

vi.mock("@/app/(dashboard)/models-and-endpoints/detailNavigation", () => ({
  useModelDetailRouting: () => ({ openModel, modelId: null, teamId: null, openTeam: vi.fn(), close: vi.fn() }),
}));

vi.mock("@/components/edit_auto_router/edit_auto_router_modal", () => ({
  __esModule: true,
  default: ({ modelData }: { modelData: { model_name?: string; model_info?: { id?: string } } }) => (
    <div data-testid="edit-auto-router-modal">
      edit:{modelData.model_name}:{modelData.model_info?.id}
    </div>
  ),
}));

vi.mock("@/components/add_model/add_auto_router_tab", () => ({
  __esModule: true,
  default: ({ handleOk }: { handleOk: () => void }) => (
    <button type="button" onClick={handleOk}>
      Submit auto router
    </button>
  ),
}));

// A realistic /v2/model/info page: two auto-routers among ordinary deployments. The panel must
// render exactly the auto_router/* rows; a view that renders page.data unfiltered passes a
// "renders a table" assertion but fails this one.
const DEPLOYMENTS = [
  {
    // DB-created adaptive router: no editor for its shape, but it must stay deletable, since
    // auto-routers are excluded from Models + Endpoints and this tab is the only delete path.
    model_name: "adaptive-router",
    litellm_params: { model: "auto_router/adaptive_router" },
    model_info: { id: "auto-3", db_model: true },
  },
  {
    // config.yaml row: the API refuses both update and delete, so neither control may appear.
    model_name: "config-router",
    litellm_params: {
      model: "auto_router/complexity_router",
      complexity_router_config: { tiers: {}, classifier_type: "llm" },
    },
    model_info: { id: "auto-4", db_model: false },
  },
  {
    model_name: "gpt-4o-mini",
    litellm_params: { model: "openai/gpt-4o-mini" },
    model_info: { id: "plain-1" },
  },
  {
    model_name: "tri-tier-router",
    litellm_params: {
      model: "auto_router/complexity_router",
      complexity_router_config: { tiers: { SIMPLE: ["gpt-4o-mini"] }, classifier_type: "heuristic" },
      complexity_router_default_model: "gpt-4o-mini",
    },
    model_info: { id: "auto-1", db_model: true, created_at: "2026-07-28T21:40:09.900000+00:00" },
  },
  {
    model_name: "anthropic-opus-4-6",
    litellm_params: { model: "anthropic/claude-opus-4-6" },
    model_info: { id: "plain-2" },
  },
  {
    model_name: "support-router",
    litellm_params: {
      model: "auto_router/support-router",
      auto_router_config: JSON.stringify({ routes: [{ name: "gpt-4o-mini" }] }),
      auto_router_default_model: "gpt-4o-mini",
    },
    model_info: { id: "auto-2", db_model: true, created_at: "2026-07-27T10:00:00.000000+00:00" },
  },
];

const pageOf = (data: typeof DEPLOYMENTS) => ({
  data,
  total_count: data.length,
  current_page: 1,
  total_pages: 1,
  size: 1000,
});

const mockDeploymentsPage = () => {
  modelInfoCall.mockResolvedValue(pageOf(DEPLOYMENTS));
};

// Oldest-first, as the proxy returns them, and two more than the ten-row first page holds.
const BULK_ROUTER_NAMES = [
  "router-01-oldest",
  ...Array.from({ length: 10 }, (_, i) => `router-${i + 2}`),
  "router-12-newest",
];

const A_FULL_PAGE_AND_TWO_MORE = Array.from({ length: 12 }, (_, index) => ({
  model_name: BULK_ROUTER_NAMES[index],
  litellm_params: {
    model: "auto_router/complexity_router",
    complexity_router_config: { tiers: {}, classifier_type: "heuristic" },
  },
  model_info: {
    id: `bulk-${index + 1}`,
    db_model: true,
    created_at: `2026-08-${String(index + 1).padStart(2, "0")}T00:00:00.000000+00:00`,
  },
}));

/** Row order as rendered, header row dropped. */
const routerNamesInOrder = () =>
  screen
    .getAllByRole("row")
    .slice(1)
    .map((row) => row.querySelector("span.text-sm.font-medium")?.textContent ?? "");

const renderPanel = (canModify = true) =>
  renderWithProviders(
    <AutoRoutersPanel
      accessToken="token"
      userRole="Admin"
      userID="u-admin"
      teams={null}
      createScope={canModify ? "unscoped-ok" : "forbidden"}
    />,
  );

describe("AutoRoutersPanel", () => {
  beforeEach(() => {
    // The shared test client caches with staleTime: Infinity and refetchOnMount: false, so
    // without this every test after the first reads the previous test's deployment page.
    testQueryClient.clear();
    modelInfoCall.mockReset();
    modelDeleteCall.mockClear();
    openModel.mockClear();
    mockDeploymentsPage();
  });

  it("lists only auto_router deployments, not every model on the proxy", async () => {
    renderPanel();

    expect(await screen.findByText("tri-tier-router")).toBeInTheDocument();
    expect(await screen.findByText("support-router")).toBeInTheDocument();
    expect(screen.queryByText("gpt-4o-mini", { selector: "span.text-sm.font-medium" })).not.toBeInTheDocument();
    expect(screen.queryByText("anthropic-opus-4-6", { selector: "span.text-sm.font-medium" })).not.toBeInTheDocument();
  });

  it("labels Type by classifier rather than by router family", async () => {
    renderPanel();

    expect(await screen.findByText("Heuristic")).toBeInTheDocument();
    expect(await screen.findByText("Semantic")).toBeInTheDocument();
  });

  // Reuses the models-page drill-in, so an auto router opens the full ModelInfoView with
  // Model Settings and Edit Settings, not a parallel detail view that reimplements part of it.
  it("opens the shared model detail view on row click", async () => {
    const user = userEvent.setup();
    renderPanel();

    await user.click(await screen.findByRole("button", { name: "support-router" }));

    expect(openModel).toHaveBeenCalledWith("auto-2");
  });

  it("opens the create form in a dialog and refetches the list after a create", async () => {
    const user = userEvent.setup();
    renderPanel();

    await screen.findByText("tri-tier-router");
    const callsBeforeCreate = modelInfoCall.mock.calls.length;

    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Add Auto Router" }));

    // A dialog, not a full-panel swap: the list stays mounted behind it.
    const dialog = await screen.findByRole("dialog");
    expect(dialog).toHaveTextContent("Add Auto Router");
    expect(screen.getByText("tri-tier-router")).toBeInTheDocument();

    await user.click(await screen.findByRole("button", { name: "Submit auto router" }));

    // Back on the list, and the deployment query was invalidated so a new router shows up
    // without a manual page reload.
    expect(await screen.findByText("tri-tier-router")).toBeInTheDocument();
    await waitFor(() => expect(modelInfoCall.mock.calls.length).toBeGreaterThan(callsBeforeCreate));
  });

  // The page decides who may write (proxy admin or team admin); the panel just has to make
  // every write affordance absent when told no, rather than let a submit 403 later. Reading
  // stays open: a read-only caller can still drill into the detail view.
  it("shows the list but no write affordances when canModify is false", async () => {
    renderPanel(false);

    expect(await screen.findByText("tri-tier-router")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Add Auto Router" })).not.toBeInTheDocument();
    expect(screen.queryByTestId("auto-router-actions-auto-1")).not.toBeInTheDocument();
    // Still navigable, because opening the detail view is a read.
    expect(screen.getByRole("button", { name: "tri-tier-router" })).toBeInTheDocument();
  });

  // Auto-routers are hidden from Models + Endpoints, which used to be the only route to the
  // delete action, so this tab is now the only place an auto router can be removed.
  it("deletes the chosen router by its model id and refetches", async () => {
    const user = userEvent.setup();
    renderPanel();

    await screen.findByText("support-router");
    const callsBeforeDelete = modelInfoCall.mock.calls.length;

    await user.click(screen.getByTestId("auto-router-actions-auto-2"));
    await user.click(await screen.findByTestId("auto-router-action-delete"));
    await user.click(await screen.findByRole("button", { name: /^delete$/i }));

    await waitFor(() => expect(modelDeleteCall).toHaveBeenCalledWith("token", "auto-2"));
    await waitFor(() => expect(modelInfoCall.mock.calls.length).toBeGreaterThan(callsBeforeDelete));
  });

  it("does not delete when the confirmation is dismissed", async () => {
    const user = userEvent.setup();
    renderPanel();

    await screen.findByText("support-router");

    await user.click(screen.getByTestId("auto-router-actions-auto-2"));
    await user.click(await screen.findByTestId("auto-router-action-delete"));
    await user.click(await screen.findByRole("button", { name: /cancel/i }));

    expect(modelDeleteCall).not.toHaveBeenCalled();
  });

  it("gives a read-only caller no delete affordance", async () => {
    renderPanel(false);

    await screen.findByText("support-router");
    expect(screen.queryByTestId("auto-router-actions-auto-2")).not.toBeInTheDocument();
  });

  it("renders an empty state when the proxy has models but no auto routers", async () => {
    modelInfoCall.mockResolvedValue(
      pageOf(DEPLOYMENTS.filter((d) => !d.litellm_params.model.startsWith("auto_router/"))),
    );

    renderPanel();

    expect(await screen.findByText("No auto routers yet")).toBeInTheDocument();
  });

  it("keeps delete available on a DB-created adaptive router that has no editor", async () => {
    const user = userEvent.setup();
    renderPanel();

    await screen.findByText("adaptive-router");
    await user.click(screen.getByTestId("auto-router-actions-auto-3"));
    await user.click(await screen.findByTestId("auto-router-action-delete"));
    await user.click(await screen.findByRole("button", { name: /^delete$/i }));

    await waitFor(() => expect(modelDeleteCall).toHaveBeenCalledWith("token", "auto-3"));
  });

  it("offers no delete on a config-defined router, which the API would refuse", async () => {
    renderPanel();

    await screen.findByText("config-router");
    expect(screen.queryByTestId("auto-router-actions-auto-4")).not.toBeInTheDocument();
  });

  // /v2/model/info returns an unordered model_list, and created_at is absent on config routers
  // and on non-enterprise proxies, so both halves of the order have to be pinned here.
  it("orders newest first, then the undated routers by name", async () => {
    renderPanel();

    await screen.findByText("tri-tier-router");

    expect(routerNamesInOrder()).toEqual([
      "tri-tier-router", // 2026-07-28
      "support-router", // 2026-07-27
      "adaptive-router", // undated, sorts after every dated row, then by name
      "config-router",
    ]);
  });

  // The reported bug: the newest router was rendered last, so it landed on page 2 and read
  // as never created.
  it("puts a just-created router on the first page of a list longer than one page", async () => {
    modelInfoCall.mockResolvedValue(pageOf(A_FULL_PAGE_AND_TWO_MORE));

    renderPanel();

    expect(await screen.findByRole("button", { name: "router-12-newest" })).toBeInTheDocument();
    // Page one holds the ten newest, so the two oldest are the ones pushed off it.
    expect(screen.queryByRole("button", { name: "router-01-oldest" })).not.toBeInTheDocument();
    expect(routerNamesInOrder()[0]).toBe("router-12-newest");
  });
});
