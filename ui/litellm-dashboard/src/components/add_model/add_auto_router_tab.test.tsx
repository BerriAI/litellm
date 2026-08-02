import { renderWithProviders, screen, waitFor } from "../../../tests/test-utils";
import userEvent from "@testing-library/user-event";
import { vi } from "vitest";
import AddAutoRouterTab from "./add_auto_router_tab";
import NotificationManager from "../molecules/notifications_manager";
import { handleAddAutoRouterSubmit } from "./handle_add_auto_router_submit";
import { getMissingTiersError } from "./build_complexity_router_config";

vi.mock("../networking", () => ({
  modelAvailableCall: vi.fn().mockResolvedValue({ data: [] }),
}));

vi.mock("@/components/llm_calls/fetch_models", () => ({
  fetchAvailableModels: vi.fn().mockResolvedValue([]),
}));

vi.mock("./handle_add_auto_router_submit", () => ({
  handleAddAutoRouterSubmit: vi.fn(),
}));

vi.mock("../molecules/notifications_manager", () => ({
  default: { fromBackend: vi.fn() },
}));

// Kept real by default so the "mandatory field" test still sees genuine tier validation; one
// test overrides it to reach the submit path without driving four tier selects.
vi.mock("./build_complexity_router_config", async (importOriginal) => {
  const actual = await importOriginal<typeof import("./build_complexity_router_config")>();
  return { ...actual, getMissingTiersError: vi.fn(actual.getMissingTiersError) };
});

// A real TeamDropdown fetches teams and renders an antd Select; the wiring under test is
// whether team_id is registered, validated and forwarded, so a plain control stands in.
vi.mock("../common_components/team_dropdown", () => ({
  default: ({ value, onChange }: { value?: string; onChange?: (next: string) => void }) => (
    <select
      data-testid="team-dropdown"
      value={value ?? ""}
      onChange={(event) => onChange?.(event.target.value)}
      aria-label="Select Team"
    >
      <option value="">none</option>
      <option value="team-1">team-1</option>
    </select>
  ),
}));

const Harness = () => <AddAutoRouterTab handleOk={vi.fn()} accessToken="token" userRole="Admin" />;

describe("AddAutoRouterTab", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("flags every mandatory field when Add Auto Router is clicked with nothing filled", async () => {
    const user = userEvent.setup();
    renderWithProviders(<Harness />);

    await user.click(screen.getByRole("button", { name: /add auto router/i }));

    expect(await screen.findByText("Auto router name is required")).toBeInTheDocument();
    expect(screen.getAllByText("This tier is required")).toHaveLength(4);
    expect(NotificationManager.fromBackend).toHaveBeenCalledWith("Please enter an Auto Router Name");
  });

  it("offers no team selector to a proxy admin, who may create an unscoped router", () => {
    renderWithProviders(<Harness />);

    expect(screen.queryByTestId("team-dropdown")).not.toBeInTheDocument();
  });

  it("requires a team admin to pick a team", async () => {
    renderWithProviders(
      <AddAutoRouterTab handleOk={vi.fn()} accessToken="token" userRole="Internal User" createScope="team-required" />,
    );

    expect(screen.getByTestId("team-dropdown")).toBeInTheDocument();
    expect(screen.getByText("Select Team")).toBeInTheDocument();
  });

  // POST /model/new 403s an unscoped create from a non-proxy-admin, so a selected team that
  // never reaches the payload is indistinguishable from having no selector at all. The value
  // has to survive form.validateFields, which only returns the fields it is asked for.
  it("carries the selected team through to the create payload", async () => {
    const user = userEvent.setup();
    vi.mocked(getMissingTiersError).mockReturnValue(null);

    renderWithProviders(
      <AddAutoRouterTab handleOk={vi.fn()} accessToken="token" userRole="Internal User" createScope="team-required" />,
    );

    await user.type(screen.getByPlaceholderText(/smart_router/i), "team-scoped-router");
    await user.selectOptions(screen.getByTestId("team-dropdown"), "team-1");
    await user.click(screen.getByRole("button", { name: /add auto router/i }));

    await waitFor(() => expect(handleAddAutoRouterSubmit).toHaveBeenCalled());
    expect(vi.mocked(handleAddAutoRouterSubmit).mock.calls.at(-1)?.[0]).toMatchObject({ team_id: "team-1" });
  });

  it("blocks the submit when a team admin has not picked a team", async () => {
    const user = userEvent.setup();
    vi.mocked(getMissingTiersError).mockReturnValue(null);

    renderWithProviders(
      <AddAutoRouterTab handleOk={vi.fn()} accessToken="token" userRole="Internal User" createScope="team-required" />,
    );

    await user.type(screen.getByPlaceholderText(/smart_router/i), "team-scoped-router");
    await user.click(screen.getByRole("button", { name: /add auto router/i }));

    expect(await screen.findByText("Please select a team to continue")).toBeInTheDocument();
    expect(handleAddAutoRouterSubmit).not.toHaveBeenCalled();
  });
});
