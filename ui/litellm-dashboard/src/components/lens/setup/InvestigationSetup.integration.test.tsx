import { act, fireEvent, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { mockAllIsIntersecting, setupIntersectionMocking } from "react-intersection-observer/test-utils";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { testQueryClient } from "@/../tests/test-utils";
import { renderWithLens as renderWithProviders, stubGateway } from "@/../tests/lens-test-utils";
import { MonitoringDialog } from "./MonitoringDialog";
import { InvestigationSetup } from "./InvestigationSetup";
import { initialWatches, watchChecks } from "../model/watches";
import { type AnalysisModelInfo, type Settings } from "../model/types";

let proxy = stubGateway();

const settings: Settings = {
  lookback_hours: 24,
  name: "Research quality",
  model: "analysis",
  context: "",
  enabled: false,
  interval_minutes: 15,
  monthly_budget: 20,
  sample_size: 100,
  sample_percent: 100,
  concurrency: 8,
  execution_ids: [],
  q: "",
  checks: [
    { id: "first", instruction: "Find repeated searches", enabled: false },
    { id: "second", instruction: "Find incomplete reports", enabled: true },
  ],
};

afterEach(() => testQueryClient.clear());

const traceRef = (n: number) => n.toString(16).toUpperCase().padStart(64, "0");

function run(name: string, n = 1, startTime = "2026-10-01T12:00:00Z") {
  const trace_ref = traceRef(n);
  const trace_id = `trace-${n}`;
  return {
    id: `${trace_ref}:${trace_id}`,
    trace_id,
    trace_ref,
    summary: {
      trace_id,
      trace_ref,
      name,
      service: "svc",
      input_preview: "",
      start_time: startTime,
      duration_ms: 1,
      status: "ok" as const,
      span_count: 2,
      agent_count: 0,
      agent_invocations: 0,
      llm_calls: 0,
      tool_calls: 0,
      error_count: 0,
      input_tokens: 0,
      output_tokens: 0,
      models: [],
      spend: null,
    },
  };
}

const analysisWorker = {
  id: "worker",
  name: "Worker",
  revoked: false,
  analysis_key_id: "a".repeat(64),
  scope: { all_teams: true, api_key_hash: "", team_id: "" },
  last_seen: "2026-10-01T12:00:00Z",
};

interface Gateway {
  readonly models?: readonly string[];
  readonly modelDetails?: readonly AnalysisModelInfo[];
  /** Models the lone worker's analysis key may use; one model becomes the setup's default. */
  readonly keyModels?: readonly string[];
}

function gatewayResponse(path: string, gateway: Gateway): unknown {
  const { models = ["analysis"], modelDetails = [], keyModels = [] } = gateway;
  if (path === "/models") return { data: models.map((id) => ({ id })) };
  if (path === "/model_group/info") return { data: modelDetails };
  if (path === "/lens") return { lenses: [], workers: keyModels.length ? [analysisWorker] : [], tracing_enabled: true };
  if (path === "/key/info") return { info: { models: keyModels, max_budget: null } };
  return [];
}

function mockGateway(gateway: Gateway = {}) {
  proxy.get.mockImplementation(async (path) => gatewayResponse(path, gateway));
}

beforeEach(() => {
  testQueryClient.clear();
  setupIntersectionMocking(vi.fn);
  proxy = stubGateway();
  mockGateway();
  proxy.post.mockResolvedValue({ eligible: 1, selected: 1, executions: [] });
});
describe("Investigation setup", () => {
  it("preserves saved manual run selections when editing and lets the preview footer clear them", async () => {
    const user = userEvent.setup();
    proxy.post.mockResolvedValue({
      eligible: 2,
      selected: 2,
      executions: [run("Saved run", 1), run("Other run", 2)],
    });
    renderWithProviders(
      <InvestigationSetup
        mode="edit"
        initial={{ ...settings, execution_ids: [run("Saved run", 1).id] }}
        onClose={vi.fn()}
        onSave={vi.fn()}
      />,
    );
    await user.click(screen.getByRole("button", { name: "Continue" }));
    await user.click(screen.getByRole("button", { name: "Continue" }));
    const preview = within(screen.getByRole("region", { name: "Matching activity" }));
    expect(await preview.findByRole("checkbox", { name: "Select Saved run" })).toBeChecked();
    expect(preview.getByText("1 selected for analysis")).toBeVisible();
    await waitFor(() => expect(screen.getByRole("button", { name: "Save changes" })).toBeEnabled());
    await user.click(preview.getByRole("button", { name: "Clear 1 selected runs" }));
    expect(preview.getByRole("checkbox", { name: "Select Saved run" })).not.toBeChecked();
    expect(preview.queryByRole("button", { name: /Clear/ })).not.toBeInTheDocument();
    expect(screen.getByRole("alert")).toHaveTextContent("Choose at least one run or turn off individual selection");
    expect(screen.getByRole("button", { name: "Save changes" })).toBeDisabled();
  });

  it.each([
    ["search", () => userEvent.type(screen.getByRole("combobox", { name: "Search runs" }), "x")],
    [
      "window",
      async () =>
        fireEvent.change(screen.getByRole("spinbutton", { name: "Review the last" }), { target: { value: "3" } }),
    ],
  ])("drops saved hand-picked runs once the %s changes, and only then", async (_, change) => {
    proxy.post.mockResolvedValue({ eligible: 2, selected: 2, executions: [run("Saved run", 1), run("Other run", 2)] });
    renderWithProviders(
      <InvestigationSetup
        mode="edit"
        initial={{ ...settings, execution_ids: [run("Saved run", 1).id] }}
        onClose={vi.fn()}
        onSave={vi.fn()}
      />,
    );
    const preview = within(screen.getByRole("region", { name: "Matching activity" }));
    expect(await preview.findByRole("button", { name: "Clear 1 selected runs" })).toBeVisible();
    await change();
    await waitFor(() => expect(preview.queryByRole("button", { name: /Clear/ })).not.toBeInTheDocument());
  });

  it("preserves check identity and disabled state when a check is edited", async () => {
    const save = vi.fn().mockResolvedValue(undefined);
    const user = userEvent.setup();
    renderWithProviders(<InvestigationSetup mode="edit" initial={settings} onClose={vi.fn()} onSave={save} />);
    await user.click(screen.getByRole("button", { name: "Continue" }));
    fireEvent.change(screen.getByRole("textbox", { name: "Check 1" }), {
      target: { value: "Find repetitive searches\nInclude retries that add no information" },
    });
    await user.click(screen.getByRole("button", { name: "Continue" }));
    await waitFor(() => expect(screen.getByRole("button", { name: "Save changes" })).toBeEnabled());
    await user.click(screen.getByRole("button", { name: "Save changes" }));
    expect(save).toHaveBeenCalledWith(
      expect.objectContaining({
        checks: [
          { ...settings.checks[0], instruction: "Find repetitive searches\nInclude retries that add no information" },
          settings.checks[1],
        ],
      }),
    );
  });

  it("previews identifiable matching runs and saves the same search", async () => {
    const save = vi.fn().mockResolvedValue(undefined);
    const user = userEvent.setup();
    proxy.post.mockImplementation(async (_path, options) => {
      const body = options?.body as { selection: Settings };
      return body.selection.q === "attr.swarm:research"
        ? { eligible: 1, selected: 1, executions: [run("Research report", 1, "2026-09-30T18:00:00Z")] }
        : { eligible: 0, executions: [] };
    });
    mockGateway({ keyModels: ["analysis"] });
    renderWithProviders(<InvestigationSetup mode="new" onClose={vi.fn()} onSave={save} />);
    fireEvent.change(screen.getByRole("textbox", { name: "Investigation name" }), {
      target: { value: "Research follow-up" },
    });
    await user.type(screen.getByRole("combobox", { name: "Search runs" }), "attr.swarm:research");
    await user.keyboard("{Escape}");
    await user.click(screen.getByRole("button", { name: "Continue" }));
    await user.click(screen.getByRole("button", { name: /Add your own/ }));
    await user.type(screen.getByRole("textbox", { name: "Check 1" }), "Find incomplete reports");
    await user.click(screen.getByRole("button", { name: "Continue" }));
    await waitFor(() => expect(screen.getByRole("button", { name: "Run and monitor" })).toBeEnabled());
    expect(screen.getByText("Research report")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Run and monitor" }));
    expect(save).toHaveBeenCalledWith(
      expect.objectContaining({
        name: "Research follow-up",
        q: "attr.swarm:research",
        enabled: true,
        checks: [
          ...watchChecks(initialWatches(undefined)),
          expect.objectContaining({ instruction: "Find incomplete reports" }),
        ],
      }),
    );
  });
});

it("walks the three steps in order, reopens a finished step from its summary, and keeps the preview beside them", async () => {
  const user = userEvent.setup();
  mockGateway({ keyModels: ["analysis"] });
  renderWithProviders(<InvestigationSetup mode="new" onClose={vi.fn()} onSave={vi.fn()} />);
  const steps = within(screen.getByRole("list", { name: "Investigation setup" }));
  expect(steps.getByRole("button", { name: /^Activity/ })).toHaveAttribute("aria-current", "step");
  expect(steps.getByRole("button", { name: /^Schedule/ })).toBeDisabled();
  expect(screen.getByRole("region", { name: "Matching activity" })).toBeVisible();
  expect(screen.queryByRole("textbox", { name: "What should the agent be doing?" })).not.toBeInTheDocument();
  await user.type(screen.getByRole("combobox", { name: "Search runs" }), "agent:support_agent");
  await user.keyboard("{Escape}");
  await user.click(screen.getByRole("button", { name: "Continue" }));
  expect(steps.getByRole("button", { name: /^Activity.*agent:support_agent/ })).toBeEnabled();
  expect(screen.queryByRole("combobox", { name: "Search runs" })).not.toBeInTheDocument();
  expect(screen.getByRole("textbox", { name: "What should the agent be doing?" })).toBeVisible();
  await user.click(screen.getByRole("button", { name: "Continue" }));
  expect(screen.getByRole("switch", { name: "Keep watching for new traces" })).toBeVisible();
  expect(screen.getByRole("region", { name: "Matching activity" })).toBeVisible();
  await user.click(steps.getByRole("button", { name: /^Activity/ }));
  expect(screen.getByRole("combobox", { name: "Search runs" })).toBeVisible();
  expect(screen.queryByRole("switch", { name: "Keep watching for new traces" })).not.toBeInTheDocument();
});

it("searches providers and saves custom history while preserving existing schedule values", async () => {
  const user = userEvent.setup();
  const save = vi.fn().mockResolvedValue(undefined);
  mockGateway({
    models: ["review", "other"],
    modelDetails: [
      { model_group: "review", providers: ["OpenAI"], mode: "chat", supported_openai_params: ["response_format"] },
      { model_group: "other", providers: ["Anthropic"], mode: "chat" },
    ],
  });
  renderWithProviders(<InvestigationSetup mode="edit" initial={settings} onClose={vi.fn()} onSave={save} />);
  await user.selectOptions(screen.getByRole("combobox", { name: "Review the last unit" }), "1");
  fireEvent.change(screen.getByRole("spinbutton", { name: "Review the last" }), { target: { value: "3" } });
  await user.click(screen.getByRole("button", { name: "Continue" }));
  await user.click(screen.getByRole("button", { name: "Continue" }));
  expect(screen.getByText(/is no longer available/)).toBeVisible();
  await user.clear(screen.getByRole("combobox", { name: "Analysis model" }));
  await user.type(screen.getByRole("combobox", { name: "Analysis model" }), "OpenAI");
  expect(screen.queryByRole("option", { name: /Anthropic/ })).not.toBeInTheDocument();
  await user.click(await screen.findByRole("option", { name: /review.*JSON output supported/ }));
  await waitFor(() => expect(screen.getByRole("button", { name: "Save changes" })).toBeEnabled());
  await user.click(screen.getByRole("button", { name: "Save changes" }));
  const expected = { model: "review", lookback_hours: 3, interval_minutes: 15, enabled: false };
  expect(save).toHaveBeenCalledWith(expect.objectContaining(expected));
});

it("configures monitoring separately and rejects a zero interval", async () => {
  const user = userEvent.setup();
  const save = vi.fn().mockResolvedValue(undefined);
  renderWithProviders(<MonitoringDialog settings={settings} ready onSave={save} onClose={vi.fn()} />);
  fireEvent.change(screen.getByRole("spinbutton", { name: "Check every" }), { target: { value: "2" } });
  await user.click(screen.getByRole("button", { name: "Enable monitoring" }));
  expect(save).toHaveBeenCalledWith(expect.objectContaining({ interval_minutes: 2, enabled: true }));
  fireEvent.change(screen.getByRole("spinbutton", { name: "Check every" }), { target: { value: "0" } });
  await waitFor(() => expect(screen.getByRole("button", { name: "Enable monitoring" })).toBeDisabled());
});

it("allows retrying an investigation save after a server error", async () => {
  const user = userEvent.setup();
  const save = vi.fn().mockRejectedValueOnce(new Error("boom")).mockResolvedValue(undefined);
  renderWithProviders(<InvestigationSetup mode="edit" initial={settings} onClose={vi.fn()} onSave={save} />);
  await user.click(screen.getByRole("button", { name: "Continue" }));
  await user.click(screen.getByRole("button", { name: "Continue" }));
  const saveButton = screen.getByRole("button", { name: "Save changes" });
  await waitFor(() => expect(saveButton).toBeEnabled());

  await user.click(saveButton);
  expect(await screen.findByRole("alert")).toHaveTextContent("boom");
  await waitFor(() => expect(saveButton).toBeEnabled());

  await user.click(saveButton);
  await waitFor(() => expect(save).toHaveBeenCalledTimes(2));
});

it("allows retrying monitoring after a server error", async () => {
  const user = userEvent.setup();
  const save = vi.fn().mockRejectedValueOnce(new Error("boom")).mockResolvedValue(undefined);
  renderWithProviders(<MonitoringDialog settings={settings} ready onSave={save} onClose={vi.fn()} />);
  const saveButton = screen.getByRole("button", { name: "Enable monitoring" });

  await user.click(saveButton);
  expect(await screen.findByRole("alert")).toHaveTextContent("boom");
  await waitFor(() => expect(saveButton).toBeEnabled());

  await user.click(saveButton);
  await waitFor(() => expect(save).toHaveBeenCalledTimes(2));
});

it("keeps the draft when readiness changes and blocks a run until the worker recovers", async () => {
  const user = userEvent.setup();
  const save = vi.fn();
  const props = {
    initial: settings,
    mode: "duplicate" as const,
    accessToken: "test",
    onClose: vi.fn(),
    onSave: save,
  };
  const view = renderWithProviders(<InvestigationSetup {...props} ready />);
  await user.click(screen.getByRole("button", { name: "Continue" }));
  await user.type(screen.getByRole("textbox", { name: "What should the agent be doing?" }), "Finish the report");
  view.rerender(<InvestigationSetup {...props} ready={false} />);
  expect(screen.getByRole("textbox", { name: "What should the agent be doing?" })).toHaveValue("Finish the report");
  await user.click(screen.getByRole("button", { name: "Continue" }));
  expect(screen.getByRole("button", { name: "Run investigation" })).toBeDisabled();
  view.rerender(<InvestigationSetup {...props} ready />);
  await waitFor(() => expect(screen.getByRole("button", { name: "Run investigation" })).toBeEnabled());
});

it.each(["empty", "error"])("allows editing saved settings when the preview is %s", async (state) => {
  if (state === "error") proxy.post.mockRejectedValue(new Error("Storage unavailable"));
  else proxy.post.mockResolvedValue({ eligible: 0, selected: 0, executions: [] });
  const user = userEvent.setup();
  renderWithProviders(<InvestigationSetup mode="edit" initial={settings} onClose={vi.fn()} onSave={vi.fn()} />);
  await user.click(screen.getByRole("button", { name: "Continue" }));
  await user.click(screen.getByRole("button", { name: "Continue" }));
  expect(screen.getByRole("button", { name: "Save changes" })).toBeEnabled();
  if (state === "error") expect(await screen.findByRole("button", { name: "Retry preview" })).toBeVisible();
  else expect(await screen.findByText(/No matches/)).toBeVisible();
});

it.each(["loading", "error"])("saves edits with the existing model while models are %s", async (state) => {
  const user = userEvent.setup();
  const save = vi.fn().mockResolvedValue(undefined);
  proxy.get.mockImplementation((path) => {
    if (path !== "/models") return Promise.resolve(gatewayResponse(path, {}));
    return state === "loading" ? new Promise(() => {}) : Promise.reject(new Error("Temporarily unavailable"));
  });
  renderWithProviders(<InvestigationSetup mode="edit" initial={settings} onClose={vi.fn()} onSave={save} />);
  await user.click(screen.getByRole("button", { name: "Continue" }));
  fireEvent.change(screen.getByRole("textbox", { name: "What should the agent be doing?" }), {
    target: { value: "Include verified sources" },
  });
  await user.click(screen.getByRole("button", { name: "Continue" }));
  await user.click(screen.getByRole("button", { name: "Save changes" }));
  expect(save).toHaveBeenCalledWith(
    expect.objectContaining({ model: settings.model, context: "Include verified sources" }),
  );
});

it.each(["new", "duplicate"] as const)("blocks a %s investigation until its model is verified", async (mode) => {
  const user = userEvent.setup();
  const save = vi.fn();
  proxy.get.mockImplementation((path) =>
    path === "/models"
      ? Promise.reject(new Error("Temporarily unavailable"))
      : Promise.resolve(gatewayResponse(path, {})),
  );
  renderWithProviders(<InvestigationSetup initial={settings} mode={mode} onClose={vi.fn()} onSave={save} />);
  expect(await screen.findByText("100% of 1")).toBeInTheDocument();
  await user.click(screen.getByRole("button", { name: "Continue" }));
  await user.click(screen.getByRole("button", { name: "Continue" }));
  expect(screen.getByRole("button", { name: mode === "new" ? "Run and monitor" : "Run investigation" })).toBeDisabled();
  expect(save).not.toHaveBeenCalled();
});

it("keeps a duplicated investigation's schedule off and saves the interval once switched on", async () => {
  const user = userEvent.setup();
  const save = vi.fn().mockResolvedValue(undefined);
  renderWithProviders(<InvestigationSetup initial={settings} mode="duplicate" onClose={vi.fn()} onSave={save} />);
  await user.click(screen.getByRole("button", { name: "Continue" }));
  await user.click(screen.getByRole("button", { name: "Continue" }));
  await waitFor(() => expect(screen.getByRole("button", { name: "Run investigation" })).toBeEnabled());
  expect(screen.getByRole("switch", { name: "Keep watching for new traces" })).not.toBeChecked();
  fireEvent.change(screen.getByRole("spinbutton", { name: "Monthly limit" }), { target: { value: "8" } });
  await user.click(screen.getByRole("switch", { name: "Keep watching for new traces" }));
  fireEvent.change(screen.getByRole("spinbutton", { name: "Check every" }), { target: { value: "120" } });
  await user.click(screen.getByRole("button", { name: "Run and monitor" }));
  expect(save).toHaveBeenCalledWith(
    expect.objectContaining({ enabled: true, interval_minutes: 120, monthly_budget: 8 }),
  );
});

it("watches new investigations every 15 minutes by default", async () => {
  const user = userEvent.setup();
  const save = vi.fn().mockResolvedValue(undefined);
  mockGateway({ keyModels: ["analysis"] });
  renderWithProviders(<InvestigationSetup mode="new" onClose={vi.fn()} onSave={save} />);
  await user.click(screen.getByRole("button", { name: "Continue" }));
  await user.click(screen.getByRole("button", { name: "Continue" }));
  const watching = await screen.findByRole("switch", { name: "Keep watching for new traces" });
  expect(watching).toBeChecked();
  expect(watching).toBeVisible();
  await waitFor(() => expect(screen.getByRole("button", { name: "Run and monitor" })).toBeEnabled());
  await user.click(screen.getByRole("button", { name: "Run and monitor" }));
  expect(save).toHaveBeenCalledWith(expect.objectContaining({ enabled: true, interval_minutes: 15 }));
});

it("appends the next preview page as the list scrolls near its end, then stops at the last page", async () => {
  let finishSecondPage = (): void => {};
  proxy.post.mockImplementation((_path, options) => {
    const { cursor } = options?.body as { cursor: string };
    const firstPage = { eligible: 2, selected: 2, executions: [run("Run one", 1)], next_cursor: "next" };
    const secondPage = { eligible: 2, selected: 2, executions: [run("Run two", 2)], next_cursor: null };
    if (cursor === "") return Promise.resolve(firstPage);
    return new Promise((resolve) => {
      finishSecondPage = () => resolve(secondPage);
    });
  });
  renderWithProviders(<InvestigationSetup mode="edit" initial={settings} onClose={vi.fn()} onSave={vi.fn()} />);
  expect(await screen.findByText("Run one")).toBeVisible();
  expect(screen.getByText("100% of 2")).toBeVisible();
  const nextPageCalls = () =>
    proxy.post.mock.calls.filter(([, options]) => (options?.body as { cursor: string }).cursor === "next");
  expect(nextPageCalls()).toHaveLength(0);
  act(() => mockAllIsIntersecting(true));
  await waitFor(() => expect(nextPageCalls()).toHaveLength(1));
  expect(screen.getByText("Run one")).toBeVisible();
  expect(screen.getByText("100% of 2")).toBeVisible();
  act(() => finishSecondPage());
  expect(await screen.findByText("Run two")).toBeVisible();
  expect(screen.getByText("Run one")).toBeVisible();
  expect(screen.queryByTestId("preview-placeholder")).not.toBeInTheDocument();
  act(() => mockAllIsIntersecting(true));
  expect(nextPageCalls()).toHaveLength(1);
});

it("fetches one preview for two keystrokes inside the debounce window", async () => {
  vi.useFakeTimers({ shouldAdvanceTime: true });
  try {
    const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });
    renderWithProviders(<InvestigationSetup mode="new" onClose={vi.fn()} onSave={vi.fn()} />);
    const previewsFor = (q: string) =>
      proxy.post.mock.calls.filter(([, options]) => (options?.body as { selection: Settings }).selection.q === q);
    await user.type(screen.getByRole("combobox", { name: "Search runs" }), "ab");
    expect(previewsFor("a")).toHaveLength(0);
    expect(previewsFor("ab")).toHaveLength(0);
    await vi.advanceTimersByTimeAsync(1000);
    await waitFor(() => expect(previewsFor("ab")).toHaveLength(1));
    expect(previewsFor("a")).toHaveLength(0);
    expect(await screen.findByText(/% of 1$/)).toBeVisible();
  } finally {
    vi.useRealTimers();
  }
});

it("shows skeleton rows on first load and keeps the previous rows while a new search loads", async () => {
  const pending = new Map<string, (page: unknown) => void>();
  proxy.post.mockImplementation(
    (_path, options) =>
      new Promise((resolve) => {
        pending.set((options?.body as { selection: Settings }).selection.q, resolve);
      }),
  );
  const user = userEvent.setup();
  renderWithProviders(<InvestigationSetup mode="new" onClose={vi.fn()} onSave={vi.fn()} />);
  const preview = within(screen.getByRole("region", { name: "Matching activity" }));
  expect((await preview.findAllByTestId("runs-placeholder")).length).toBeGreaterThan(0);
  act(() => pending.get("")?.({ eligible: 1, selected: 1, executions: [run("Old run", 1)] }));
  expect(await preview.findByText("Old run")).toBeVisible();
  expect(preview.queryByTestId("runs-placeholder")).not.toBeInTheDocument();
  await user.type(screen.getByRole("combobox", { name: "Search runs" }), "new");
  await waitFor(() => expect(pending.has("new")).toBe(true));
  expect(preview.getByText("Old run")).toBeVisible();
  expect(preview.queryByTestId("runs-placeholder")).not.toBeInTheDocument();
  act(() => pending.get("new")?.({ eligible: 1, selected: 1, executions: [run("New run", 2)] }));
  expect(await preview.findByText("New run")).toBeVisible();
  expect(preview.queryByText("Old run")).not.toBeInTheDocument();
});

it.each(["empty", "error"])("blocks a new investigation when its preview is %s", async (state) => {
  if (state === "error") proxy.post.mockRejectedValue(new Error("Storage unavailable"));
  else proxy.post.mockResolvedValue({ eligible: 0, selected: 0, executions: [] });
  const user = userEvent.setup();
  renderWithProviders(<InvestigationSetup mode="duplicate" initial={settings} onClose={vi.fn()} onSave={vi.fn()} />);
  await user.click(screen.getByRole("button", { name: "Continue" }));
  await user.click(screen.getByRole("button", { name: "Continue" }));
  expect(screen.getByRole("button", { name: "Run investigation" })).toBeDisabled();
});

describe("Sample", () => {
  it("slides the sample rate to scale the run count, caps it by typing, and saves both", async () => {
    const save = vi.fn().mockResolvedValue(undefined);
    const user = userEvent.setup();
    proxy.post.mockResolvedValue({ eligible: 2000, selected: 2000, executions: [run("Run one", 1)] });
    mockGateway({ keyModels: ["analysis"] });
    renderWithProviders(<InvestigationSetup mode="new" onClose={vi.fn()} onSave={save} />);
    const count = screen.getByRole("textbox", { name: "Runs to analyze" });
    await waitFor(() => expect(count).toHaveValue("2,000"));
    const slider = screen.getByLabelText("Sample", { selector: "input[type=range]" });
    slider.focus();
    await user.keyboard("{Home}{ArrowRight}{ArrowRight}");
    expect(screen.getByText("3% of 2,000")).toBeInTheDocument();
    expect(count).toHaveValue("60");
    await user.clear(count);
    await user.type(count, "25");
    expect(count).toHaveValue("25");
    await user.click(screen.getByRole("button", { name: "Continue" }));
    await user.click(screen.getByRole("button", { name: "Continue" }));
    await waitFor(() => expect(screen.getByRole("button", { name: "Run and monitor" })).toBeEnabled());
    await user.click(screen.getByRole("button", { name: "Run and monitor" }));
    expect(save).toHaveBeenCalledWith(expect.objectContaining({ sample_percent: 3, sample_size: 25 }));
  });

  it("drops a cap that no longer limits anything once the field is left", async () => {
    const user = userEvent.setup();
    proxy.post.mockResolvedValue({ eligible: 40, selected: 40, executions: [run("Run one", 1)] });
    renderWithProviders(<InvestigationSetup mode="new" onClose={vi.fn()} onSave={vi.fn()} />);
    const count = screen.getByRole("textbox", { name: "Runs to analyze" });
    await waitFor(() => expect(count).toHaveValue("40"));
    await user.clear(count);
    await user.type(count, "10");
    expect(screen.getByRole("button", { name: /Remove cap/ })).toBeInTheDocument();
    await user.clear(count);
    await user.type(count, "500");
    await user.tab();
    expect(count).toHaveValue("40");
    expect(screen.queryByRole("button", { name: /Remove cap/ })).not.toBeInTheDocument();
  });
});

describe("URL draft", () => {
  const lastUrl = (onUrlUpdate: ReturnType<typeof vi.fn>) =>
    new URLSearchParams((onUrlUpdate.mock.lastCall?.[0] as { queryString: string } | undefined)?.queryString ?? "");

  it("opens a new investigation filled from the URL, previews that search, and writes edits back", async () => {
    const user = userEvent.setup();
    const onUrlUpdate = vi.fn();
    renderWithProviders(<InvestigationSetup mode="new" onClose={vi.fn()} onSave={vi.fn()} />, {
      searchParams: "?q=agent:support_agent&lookback=72&sample=50&name=Refunds&step=criteria",
      onUrlUpdate,
    });
    expect(screen.getByRole("textbox", { name: "Investigation name" })).toHaveValue("Refunds");
    expect(screen.getByRole("button", { name: /^Criteria/ })).toHaveAttribute("aria-current", "step");
    expect(screen.getByRole("button", { name: /^Activity.*agent:support_agent.*3 days/ })).toBeEnabled();
    await waitFor(() =>
      expect(proxy.post).toHaveBeenCalledWith(
        expect.anything(),
        expect.objectContaining({
          body: expect.objectContaining({
            lookback_hours: 72,
            selection: expect.objectContaining({ q: "agent:support_agent", sample_percent: 50 }),
          }),
        }),
      ),
    );
    await user.type(screen.getByRole("textbox", { name: "What should the agent be doing?" }), "Refund with receipts");
    await waitFor(() => expect(lastUrl(onUrlUpdate).get("context")).toBe("Refund with receipts"));
    expect(lastUrl(onUrlUpdate).get("q")).toBe("agent:support_agent");
    await user.click(screen.getByRole("button", { name: "Continue" }));
    await waitFor(() => expect(lastUrl(onUrlUpdate).get("step")).toBe("run"));
  });

  it("edits an investigation from its saved settings, never from a draft left in the URL", () => {
    const onUrlUpdate = vi.fn();
    renderWithProviders(<InvestigationSetup mode="edit" initial={settings} onClose={vi.fn()} onSave={vi.fn()} />, {
      searchParams: "?name=Draft&lookback=72",
      onUrlUpdate,
    });
    expect(screen.getByRole("textbox", { name: "Investigation name" })).toHaveValue("Research quality");
    expect(screen.getByRole("spinbutton", { name: "Review the last" })).toHaveValue(1);
    fireEvent.change(screen.getByRole("textbox", { name: "Investigation name" }), { target: { value: "Renamed" } });
    expect(onUrlUpdate).not.toHaveBeenCalled();
  });
});

describe("Watch for", () => {
  const tile = (name: string) => screen.getByRole("button", { name: new RegExp(`^${name}`) });

  it("saves exactly the presets the user toggled, by click and by number key", async () => {
    const save = vi.fn().mockResolvedValue(undefined);
    const user = userEvent.setup();
    mockGateway({ keyModels: ["analysis"] });
    renderWithProviders(<InvestigationSetup mode="new" onClose={vi.fn()} onSave={save} />);
    await user.click(screen.getByRole("button", { name: "Continue" }));
    await user.click(tile("unhappy"));
    tile("unsolved").focus();
    await user.keyboard("6");
    await user.keyboard("{ArrowRight}{ArrowRight}");
    expect(tile("unsafe")).toHaveFocus();
    expect(tile("unhappy")).toHaveAttribute("aria-pressed", "false");
    expect(tile("looping")).toHaveAttribute("aria-pressed", "true");
    await user.click(screen.getByRole("button", { name: "Continue" }));
    await waitFor(() => expect(screen.getByRole("button", { name: "Run and monitor" })).toBeEnabled());
    await user.click(screen.getByRole("button", { name: "Run and monitor" }));
    const saved = (save.mock.calls[0][0] as Settings).checks.map((check) => check.id);
    expect(saved).toEqual(["watch_unsolved", "watch_blocked", "watch_looping"]);
  });

  it("keeps an edited investigation's preset choices and custom checks apart", async () => {
    const save = vi.fn().mockResolvedValue(undefined);
    const user = userEvent.setup();
    const initial: Settings = {
      ...settings,
      checks: [{ id: "watch_invented", instruction: "old wording", enabled: true }, settings.checks[1]],
    };
    renderWithProviders(<InvestigationSetup mode="edit" initial={initial} onClose={vi.fn()} onSave={save} />);
    await user.click(screen.getByRole("button", { name: "Continue" }));
    expect(tile("invented")).toHaveAttribute("aria-pressed", "true");
    expect(tile("unsolved")).toHaveAttribute("aria-pressed", "false");
    expect(screen.getByRole("textbox", { name: "Check 1" })).toHaveValue("Find incomplete reports");
    await user.click(screen.getByRole("button", { name: "Continue" }));
    await waitFor(() => expect(screen.getByRole("button", { name: "Save changes" })).toBeEnabled());
    await user.click(screen.getByRole("button", { name: "Save changes" }));
    expect((save.mock.calls[0][0] as Settings).checks).toEqual([
      ...watchChecks(new Set(["watch_invented"])),
      settings.checks[1],
    ]);
  });

  it("lets a run start from presets alone and blocks it once nothing is selected", async () => {
    const user = userEvent.setup();
    mockGateway({ keyModels: ["analysis"] });
    renderWithProviders(<InvestigationSetup mode="new" onClose={vi.fn()} onSave={vi.fn()} />);
    await user.click(screen.getByRole("button", { name: "Continue" }));
    await user.click(screen.getByRole("button", { name: "Continue" }));
    expect(await screen.findByRole("button", { name: "Run and monitor" })).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: /^Criteria/ }));
    for (const name of ["unsolved", "blocked", "unhappy"]) await user.click(tile(name));
    await user.click(screen.getByRole("button", { name: "Continue" }));
    expect(screen.getByRole("alert")).toHaveTextContent(
      "Describe the expected behavior or pick something to watch for",
    );
  });
});
