import { fireEvent, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { renderWithProviders } from "@/../tests/test-utils";
import { MonitoringDialog } from "./MonitoringDialog";
import { InvestigationSetupDialog } from "./InvestigationSetupDialog";
import { apiClient } from "@/components/networking";
import { initialWatches, watchChecks } from "./watches";
import { type Settings } from "../model/types";

vi.mock("@/components/networking", () => ({ apiClient: { post: vi.fn(), get: vi.fn() } }));

const settings: Settings = {
  lookback_hours: 24,
  name: "Research quality",
  model: "analysis",
  source: "traces",
  context: "",
  enabled: false,
  filters: [],
  interval_minutes: 15,
  monthly_budget: 20,
  sample_size: 100,
  sample_percent: 100,
  concurrency: 8,
  team_id: "",
  execution_ids: [],
  service: "",
  agent_name: "",
  checks: [
    { id: "first", instruction: "Find repeated searches", enabled: false },
    { id: "second", instruction: "Find incomplete reports", enabled: true },
  ],
};

beforeEach(() => {
  vi.mocked(apiClient.get).mockReset();
  vi.mocked(apiClient.get).mockResolvedValue([]);
  vi.mocked(apiClient.post).mockReset();
  vi.mocked(apiClient.post).mockResolvedValue({ eligible: 1, selected: 1, executions: [] });
});
describe("Lens setup", () => {
  it("preserves saved manual run selections when editing an investigation", async () => {
    const user = userEvent.setup();
    renderWithProviders(
      <InvestigationSetupDialog
        initial={{ ...settings, execution_ids: ["saved-run"] }}
        models={["analysis"]}
        accessToken="test"
        onClose={vi.fn()}
        onSave={vi.fn()}
      />,
    );
    await user.click(screen.getByRole("button", { name: "Continue" }));
    await user.click(screen.getByRole("button", { name: "Continue" }));
    expect(await screen.findByRole("button", { name: "Clear 1 selected runs" })).toBeInTheDocument();
  });

  it("preserves check identity and disabled state when a check is edited", async () => {
    const save = vi.fn().mockResolvedValue(undefined);
    const user = userEvent.setup();
    renderWithProviders(
      <InvestigationSetupDialog
        initial={settings}
        models={["analysis"]}
        accessToken="test"
        onClose={vi.fn()}
        onSave={save}
      />,
    );
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

  it("previews identifiable matching runs and saves the same filter selection", async () => {
    const save = vi.fn().mockResolvedValue(undefined);
    const user = userEvent.setup();
    vi.mocked(apiClient.post).mockImplementation(async (_path, options) => {
      const body = options?.body as { settings: Settings };
      return body.settings.filters?.some((f) => f.key === "swarm" && f.value === "research")
        ? {
            eligible: 1,
            selected: 1,
            executions: [
              {
                id: "run",
                source: "requests",
                trace_id: "request-42",
                name: "Research report",
                start_time: "2026-09-30 18:00:00.000",
                span_count: 1,
              },
            ],
          }
        : { eligible: 0, executions: [] };
    });
    renderWithProviders(
      <InvestigationSetupDialog
        models={["analysis"]}
        defaultModel="analysis"
        accessToken="test"
        onClose={vi.fn()}
        onSave={save}
      />,
    );
    fireEvent.change(screen.getByRole("textbox", { name: "Investigation name" }), {
      target: { value: "Research follow-up" },
    });
    await user.click(screen.getByText("Advanced filters"));
    await user.click(screen.getByRole("button", { name: "Add condition" }));
    fireEvent.change(screen.getByRole("combobox", { name: "Metadata key 1" }), { target: { value: "swarm" } });
    fireEvent.change(screen.getByRole("combobox", { name: "Metadata value 1" }), { target: { value: "research" } });
    await user.click(screen.getByRole("button", { name: "Continue" }));
    await user.click(screen.getByRole("button", { name: /Add your own/ }));
    await user.type(screen.getByRole("textbox", { name: "Check 1" }), "Find incomplete reports");
    await user.click(screen.getByRole("button", { name: /Add your own/ }));
    fireEvent.change(screen.getByRole("textbox", { name: "Check 2" }), {
      target: { value: "Find repeated searches\nInclude retries that add no information" },
    });
    await user.click(screen.getByRole("button", { name: /Add your own/ }));
    await user.click(screen.getByRole("button", { name: "Remove check 3" }));
    await user.click(screen.getByRole("button", { name: "Continue" }));
    await waitFor(() => expect(screen.getByRole("button", { name: "Run investigation" })).toBeEnabled());
    expect(screen.getByText("1 matching run")).toBeInTheDocument();
    expect(screen.getByText("Research report")).toBeInTheDocument();
    expect(screen.getByText(/2026-09-30/)).toBeInTheDocument();
    expect(screen.getByRole("spinbutton", { name: "Monthly limit (USD)" })).not.toBeVisible();
    await user.click(screen.getByRole("button", { name: "Run investigation" }));
    const expected = {
      name: "Research follow-up",
      service: "",
      agent_name: "",
      filters: [{ key: "swarm", value: "research" }],
      enabled: false,
      sample_size: null,
      model: "analysis",
      monthly_budget: 100,
      checks: [
        ...watchChecks(initialWatches(undefined)),
        expect.objectContaining({ instruction: "Find incomplete reports" }),
        expect.objectContaining({ instruction: "Find repeated searches\nInclude retries that add no information" }),
      ],
    };
    expect(save).toHaveBeenCalledWith(expect.objectContaining(expected));
  });
});

it("searches providers and saves custom history while preserving existing schedule values", async () => {
  const user = userEvent.setup();
  const save = vi.fn().mockResolvedValue(undefined);
  renderWithProviders(
    <InvestigationSetupDialog
      initial={settings}
      models={["review", "other"]}
      modelDetails={[
        { model_group: "review", providers: ["OpenAI"], mode: "chat", supported_openai_params: ["response_format"] },
        { model_group: "other", providers: ["Anthropic"], mode: "chat" },
      ]}
      accessToken="test"
      onClose={vi.fn()}
      onSave={save}
    />,
  );
  await user.click(screen.getByRole("button", { name: "Continue" }));
  await user.click(screen.getByRole("button", { name: "Continue" }));
  await user.selectOptions(screen.getByRole("combobox", { name: "Review the last unit" }), "1");
  fireEvent.change(screen.getByRole("spinbutton", { name: "Review the last" }), { target: { value: "3" } });
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
  renderWithProviders(
    <InvestigationSetupDialog
      initial={settings}
      models={["analysis"]}
      accessToken="test"
      onClose={vi.fn()}
      onSave={save}
    />,
  );
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
    models: ["analysis"],
    accessToken: "test",
    onClose: vi.fn(),
    onSave: save,
  };
  const view = renderWithProviders(<InvestigationSetupDialog {...props} ready />);
  await user.click(screen.getByRole("button", { name: "Continue" }));
  await user.type(screen.getByRole("textbox", { name: "What should the agent be doing?" }), "Finish the report");
  view.rerender(<InvestigationSetupDialog {...props} ready={false} />);
  expect(screen.getByRole("textbox", { name: "What should the agent be doing?" })).toHaveValue("Finish the report");
  await user.click(screen.getByRole("button", { name: "Continue" }));
  expect(screen.getByRole("button", { name: "Run investigation" })).toBeDisabled();
  view.rerender(<InvestigationSetupDialog {...props} ready />);
  await waitFor(() => expect(screen.getByRole("button", { name: "Run investigation" })).toBeEnabled());
});

it.each(["empty", "error"])("allows editing saved settings when the preview is %s", async (state) => {
  if (state === "error") vi.mocked(apiClient.post).mockRejectedValue(new Error("Storage unavailable"));
  else vi.mocked(apiClient.post).mockResolvedValue({ eligible: 0, selected: 0, executions: [] });
  const user = userEvent.setup();
  renderWithProviders(
    <InvestigationSetupDialog
      initial={settings}
      models={["analysis"]}
      accessToken="test"
      onClose={vi.fn()}
      onSave={vi.fn()}
    />,
  );
  await user.click(screen.getByRole("button", { name: "Continue" }));
  await user.click(screen.getByRole("button", { name: "Continue" }));
  expect(screen.getByRole("button", { name: "Save changes" })).toBeEnabled();
  if (state === "error") expect(await screen.findByRole("button", { name: "Retry preview" })).toBeVisible();
  else expect(await screen.findByText(/No matches/)).toBeVisible();
});

it.each(["loading", "error"])("saves edits with the existing model while models are %s", async (state) => {
  const user = userEvent.setup();
  const save = vi.fn().mockResolvedValue(undefined);
  renderWithProviders(
    <InvestigationSetupDialog
      initial={settings}
      models={[]}
      modelsLoading={state === "loading"}
      modelsError={state === "error" ? "Temporarily unavailable" : undefined}
      accessToken="test"
      onClose={vi.fn()}
      onSave={save}
    />,
  );
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
  renderWithProviders(
    <InvestigationSetupDialog
      initial={settings}
      mode={mode}
      models={[]}
      modelsError="Temporarily unavailable"
      accessToken="test"
      onClose={vi.fn()}
      onSave={save}
    />,
  );
  await user.click(screen.getByRole("button", { name: "Continue" }));
  await user.click(screen.getByRole("button", { name: "Continue" }));
  expect(await screen.findByText("1 matching run")).toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Run investigation" })).toBeDisabled();
  expect(save).not.toHaveBeenCalled();
});

it("shows optional budget and repeat controls only under advanced options and saves their values", async () => {
  const user = userEvent.setup();
  const save = vi.fn().mockResolvedValue(undefined);
  renderWithProviders(
    <InvestigationSetupDialog
      initial={settings}
      mode="duplicate"
      models={["analysis"]}
      accessToken="test"
      onClose={vi.fn()}
      onSave={save}
    />,
  );
  await user.click(screen.getByRole("button", { name: "Continue" }));
  await user.click(screen.getByRole("button", { name: "Continue" }));
  await waitFor(() => expect(screen.getByRole("button", { name: "Run investigation" })).toBeEnabled());
  await user.click(screen.getByText("Advanced options"));
  expect(screen.getByRole("checkbox", { name: "Repeat this investigation" })).not.toBeChecked();
  fireEvent.change(screen.getByRole("spinbutton", { name: "Monthly limit (USD)" }), { target: { value: "8" } });
  await user.click(screen.getByRole("checkbox", { name: "Repeat this investigation" }));
  fireEvent.change(screen.getByRole("spinbutton", { name: "Repeat every" }), { target: { value: "120" } });
  await user.click(screen.getByRole("button", { name: "Run and monitor" }));
  expect(save).toHaveBeenCalledWith(
    expect.objectContaining({ enabled: true, interval_minutes: 120, monthly_budget: 8 }),
  );
});

it("does not silently analyze everything after individual selection is enabled", async () => {
  const user = userEvent.setup();
  vi.mocked(apiClient.post).mockResolvedValue({
    eligible: 1,
    selected: 1,
    executions: [
      {
        id: "selected-run",
        name: "Example run",
        trace_id: "trace",
        source: "traces",
        start_time: "2026-10-01T12:00:00Z",
        span_count: 2,
      },
    ],
  });
  renderWithProviders(
    <InvestigationSetupDialog
      initial={settings}
      models={["analysis"]}
      accessToken="test"
      onClose={vi.fn()}
      onSave={vi.fn()}
    />,
  );
  await user.click(screen.getByRole("button", { name: "Continue" }));
  await user.click(screen.getByRole("button", { name: "Continue" }));
  await user.click(screen.getByText("Advanced options"));
  await user.click(screen.getByRole("checkbox", { name: "Choose individual runs" }));
  expect(screen.getByRole("button", { name: "Save changes" })).toBeDisabled();
  await user.click(await screen.findByRole("checkbox", { name: "Select Example run" }));
  await waitFor(() => expect(screen.getByRole("button", { name: "Save changes" })).toBeEnabled());
});

it("refreshes agent suggestions when the first activity arrives", async () => {
  vi.useFakeTimers({ shouldAdvanceTime: true });
  try {
    const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });
    vi.mocked(apiClient.post).mockResolvedValue({ eligible: 0, selected: 0, executions: [] });
    renderWithProviders(
      <InvestigationSetupDialog models={["analysis"]} accessToken="test" onClose={vi.fn()} onSave={vi.fn()} />,
    );
    await vi.advanceTimersByTimeAsync(400);
    await user.click(screen.getByRole("combobox", { name: "Agent (optional)" }));
    expect(await screen.findByText(/No matches. You can enter/)).toBeVisible();
    await user.keyboard("{Escape}");
    vi.mocked(apiClient.post).mockResolvedValue({
      eligible: 1,
      selected: 1,
      executions: [
        {
          id: "new-run",
          name: "First support run",
          service: "support-agent",
          trace_id: "new-trace",
          source: "traces",
          start_time: "2026-10-01T12:00:00Z",
          span_count: 2,
        },
      ],
    });
    vi.mocked(apiClient.get).mockResolvedValue(["support-agent"]);
    await vi.advanceTimersByTimeAsync(15000);
    await user.click(screen.getByRole("combobox", { name: "Agent (optional)" }));
    expect(await screen.findByRole("option", { name: "support-agent" })).toBeVisible();
  } finally {
    vi.useRealTimers();
  }
});

it.each(["empty", "error"])("blocks a new investigation when its preview is %s", async (state) => {
  if (state === "error") vi.mocked(apiClient.post).mockRejectedValue(new Error("Storage unavailable"));
  else vi.mocked(apiClient.post).mockResolvedValue({ eligible: 0, selected: 0, executions: [] });
  const user = userEvent.setup();
  renderWithProviders(
    <InvestigationSetupDialog
      mode="duplicate"
      initial={settings}
      models={["analysis"]}
      accessToken="test"
      onClose={vi.fn()}
      onSave={vi.fn()}
    />,
  );
  await user.click(screen.getByRole("button", { name: "Continue" }));
  await user.click(screen.getByRole("button", { name: "Continue" }));
  expect(screen.getByRole("button", { name: "Run investigation" })).toBeDisabled();
});

it("exposes an unsupported inherited model before allowing a run", async () => {
  const user = userEvent.setup();
  renderWithProviders(
    <InvestigationSetupDialog
      mode="duplicate"
      initial={settings}
      models={["analysis"]}
      modelDetails={[{ model_group: "analysis", mode: "embedding", providers: [] }]}
      accessToken="test"
      onClose={vi.fn()}
      onSave={vi.fn()}
    />,
  );
  await user.click(screen.getByRole("button", { name: "Continue" }));
  await user.click(screen.getByRole("button", { name: "Continue" }));
  expect(screen.getByText("Choose a chat model that supports JSON output.")).toBeVisible();
  expect(screen.getByRole("combobox", { name: "Analysis model" })).toBeVisible();
  expect(screen.getByRole("button", { name: "Run investigation" })).toBeDisabled();
});

it("saves a discovered agent independently of the application name", async () => {
  vi.mocked(apiClient.get).mockResolvedValue(["research_agent", "support_agent"]);
  const user = userEvent.setup();
  const save = vi.fn();
  renderWithProviders(
    <InvestigationSetupDialog
      initial={{ ...settings, service: "shared-service" }}
      models={["analysis"]}
      accessToken="test"
      onClose={vi.fn()}
      onSave={save}
    />,
  );
  await user.click(screen.getByRole("combobox", { name: "Agent (optional)" }));
  await user.click(await screen.findByRole("option", { name: "research_agent" }));
  await user.click(screen.getByRole("button", { name: "Continue" }));
  await user.click(screen.getByRole("button", { name: "Continue" }));
  await user.click(screen.getByRole("button", { name: "Save changes" }));
  expect(save).toHaveBeenCalledWith(
    expect.objectContaining({ agent_name: "research_agent", service: "shared-service" }),
  );
});

describe("Watch for", () => {
  const tile = (name: string) => screen.getByRole("button", { name: new RegExp(`^${name}`) });

  it("saves exactly the presets the user toggled, by click and by number key", async () => {
    const save = vi.fn().mockResolvedValue(undefined);
    const user = userEvent.setup();
    renderWithProviders(
      <InvestigationSetupDialog
        models={["analysis"]}
        defaultModel="analysis"
        accessToken="test"
        onClose={vi.fn()}
        onSave={save}
      />,
    );
    await user.click(screen.getByRole("button", { name: "Continue" }));
    await user.click(tile("unhappy"));
    tile("unsolved").focus();
    await user.keyboard("6");
    await user.keyboard("{ArrowRight}{ArrowRight}");
    expect(tile("unsafe")).toHaveFocus();
    expect(tile("unhappy")).toHaveAttribute("aria-pressed", "false");
    expect(tile("looping")).toHaveAttribute("aria-pressed", "true");
    await user.click(screen.getByRole("button", { name: "Continue" }));
    await waitFor(() => expect(screen.getByRole("button", { name: "Run investigation" })).toBeEnabled());
    await user.click(screen.getByRole("button", { name: "Run investigation" }));
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
    renderWithProviders(
      <InvestigationSetupDialog
        initial={initial}
        models={["analysis"]}
        accessToken="test"
        onClose={vi.fn()}
        onSave={save}
      />,
    );
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
    renderWithProviders(
      <InvestigationSetupDialog
        models={["analysis"]}
        defaultModel="analysis"
        accessToken="test"
        onClose={vi.fn()}
        onSave={vi.fn()}
      />,
    );
    await user.click(screen.getByRole("button", { name: "Continue" }));
    await user.click(screen.getByRole("button", { name: "Continue" }));
    expect(await screen.findByRole("button", { name: "Run investigation" })).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Back" }));
    for (const name of ["unsolved", "blocked", "unhappy"]) await user.click(tile(name));
    await user.click(screen.getByRole("button", { name: "Continue" }));
    expect(screen.getByRole("alert")).toHaveTextContent(
      "Describe the expected behavior or pick something to watch for",
    );
  });
});
