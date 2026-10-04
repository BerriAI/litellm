import { fireEvent, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { renderWithProviders } from "../../../../tests/test-utils";
import { clampDrawerWidth, PANEL_TRIGGER } from "@/components/shared/SidePanel";
import { RunDrawer } from "./RunDrawer";
import { type RunSelection, traceRefOf, useOpenTraceRouting } from "./traceRouting";
import type { TraceSummary } from "./traceTypes";

vi.mock("./TraceDrawer", () => ({
  RunView: ({ traceId }: { traceId: string }) => <div data-testid="run-view">run {traceId}</div>,
}));

const run = (trace_id: string): TraceSummary => ({
  trace_id,
  name: trace_id,
  service: "svc",
  input_preview: "",
  start_time: "2026-09-30T00:00:00Z",
  duration_ms: 1,
  status: "ok",
  span_count: 1,
  agent_count: 1,
  agent_invocations: 1,
  llm_calls: 0,
  tool_calls: 0,
  error_count: 0,
  input_tokens: 0,
  output_tokens: 0,
  models: [],
  spend: null,
});

const selection: RunSelection = {
  spanId: null,
  view: "steps",
  spanTab: "content",
  stepQuery: "",
  errorsOnly: false,
  selectSpan: vi.fn(),
  setView: vi.fn(),
  setSpanTab: vi.fn(),
  setStepQuery: vi.fn(),
  setErrorsOnly: vi.fn(),
};

function RoutedDrawer({ runs }: { runs: TraceSummary[] }) {
  const { trace, openTrace, selection, fullScreen, setFullScreen } = useOpenTraceRouting();
  return (
    <RunDrawer
      trace={trace}
      runs={runs}
      accessToken="sk"
      selection={selection}
      onSelect={openTrace}
      fullScreen={fullScreen}
      onFullScreenChange={setFullScreen}
    />
  );
}

const lastUrl = (onUrlUpdate: ReturnType<typeof vi.fn>) =>
  new URLSearchParams(String(onUrlUpdate.mock.lastCall?.[0].queryString ?? ""));

const mockReducedMotion = (reduce: boolean) =>
  vi.spyOn(window, "matchMedia").mockImplementation(
    (query: string) =>
      ({
        matches: reduce && query.includes("reduce"),
        media: query,
        addEventListener: vi.fn(),
        removeEventListener: vi.fn(),
      }) as unknown as MediaQueryList,
  );

describe("clampDrawerWidth", () => {
  it("keeps at least 700px and a 100px strip of list on wide screens", () => {
    expect(clampDrawerWidth(200, 1600)).toBe(700);
    expect(clampDrawerWidth(5000, 1600)).toBe(1500);
    expect(clampDrawerWidth(900, 1600)).toBe(900);
  });

  it("never exceeds the screen on viewports narrower than the minimum width", () => {
    expect(clampDrawerWidth(900, 600)).toBe(600);
    expect(clampDrawerWidth(100, 600)).toBe(600);
    expect(clampDrawerWidth(750, 760)).toBe(700);
  });
});

describe("RunDrawer", () => {
  afterEach(() => {
    vi.restoreAllMocks();
    window.localStorage.clear();
  });

  it("fills the page across trace navigation and restores the resized drawer width", async () => {
    const runs = [run("a"), run("b")];
    const onUrlUpdate = vi.fn();
    renderWithProviders(<RoutedDrawer runs={runs} />, { searchParams: "?trace=a", onUrlUpdate });
    const drawer = screen.getByRole("complementary", { name: "Trace details" });
    fireEvent.keyDown(screen.getByRole("separator", { name: "Resize trace panel" }), { key: "ArrowLeft" });
    const resizedWidth = drawer.style.width;
    const storedWidth = window.localStorage.getItem("litellm.agentTraces.drawerWidth");

    fireEvent.click(screen.getByRole("button", { name: "Enter full screen" }));
    expect(drawer).toHaveStyle({ width: "100%" });
    expect(screen.queryByRole("separator", { name: "Resize trace panel" })).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Next trace (J)" }));
    expect(drawer).toHaveStyle({ width: "100%" });
    expect(screen.getByTestId("run-view")).toHaveTextContent("run b");
    await waitFor(() => expect(lastUrl(onUrlUpdate).get("fullscreen")).toBe("true"));

    fireEvent.click(screen.getByRole("button", { name: "Exit full screen" }));
    expect(drawer).toHaveStyle({ width: resizedWidth });
    expect(screen.getByRole("separator", { name: "Resize trace panel" })).toBeInTheDocument();
    expect(window.localStorage.getItem("litellm.agentTraces.drawerWidth")).toBe(storedWidth);
  });

  it("opens full screen from a shared link and drops it from the URL on close", async () => {
    const onUrlUpdate = vi.fn();
    renderWithProviders(<RoutedDrawer runs={[run("a")]} />, { searchParams: "?trace=a&fullscreen=true", onUrlUpdate });
    expect(screen.getByRole("complementary", { name: "Trace details" })).toHaveStyle({ width: "100%" });
    fireEvent.click(screen.getByRole("button", { name: "Close trace (Esc)" }));
    await waitFor(() => expect(lastUrl(onUrlUpdate).has("trace")).toBe(false));
    expect(lastUrl(onUrlUpdate).has("fullscreen")).toBe(false);
  });

  it.each(["Close (Esc)", "Close trace (Esc)"])("closes a full-screen trace using %s", (name) => {
    const runs = [run("a")];
    const onSelect = vi.fn();
    renderWithProviders(
      <RunDrawer
        trace={traceRefOf(runs[0])}
        runs={runs}
        accessToken="sk"
        selection={selection}
        onSelect={onSelect}
        fullScreen
        onFullScreenChange={vi.fn()}
      />,
    );
    fireEvent.click(screen.getByRole("button", { name }));
    expect(onSelect).toHaveBeenCalledExactlyOnceWith(null);
  });

  it("keeps Escape available to close a full-screen trace", () => {
    const runs = [run("a")];
    const onSelect = vi.fn();
    renderWithProviders(
      <RunDrawer
        trace={traceRefOf(runs[0])}
        runs={runs}
        accessToken="sk"
        selection={selection}
        onSelect={onSelect}
        fullScreen
        onFullScreenChange={vi.fn()}
      />,
    );
    fireEvent.keyDown(window, { key: "Escape" });
    expect(onSelect).toHaveBeenCalledExactlyOnceWith(null);
  });

  it("leaves J, K and Escape to an open menu or listbox instead of stepping or closing the trace", () => {
    const runs = [run("a"), run("b"), run("c")];
    const onSelect = vi.fn();
    renderWithProviders(
      <>
        <div role="menu">
          <button type="button">menu item</button>
        </div>
        <div role="listbox">
          <button type="button">option</button>
        </div>
        <RunDrawer
          trace={traceRefOf(runs[1])}
          runs={runs}
          accessToken="sk"
          selection={selection}
          onSelect={onSelect}
          fullScreen={false}
          onFullScreenChange={vi.fn()}
        />
      </>,
    );
    for (const target of [screen.getByText("menu item"), screen.getByText("option")]) {
      for (const key of ["j", "k", "Escape"]) fireEvent.keyDown(target, { key });
    }
    expect(onSelect).not.toHaveBeenCalled();
    fireEvent.keyDown(window, { key: "j" });
    expect(onSelect).toHaveBeenCalledExactlyOnceWith(traceRefOf(runs[2]));
  });

  it("closes on a press outside the panel but not inside it, on a trigger row, or in a dialog", () => {
    const runs = [run("a")];
    const onSelect = vi.fn();
    renderWithProviders(
      <>
        <button type="button">outside</button>
        <table>
          <tbody>
            <tr {...PANEL_TRIGGER}>
              <td>row</td>
            </tr>
          </tbody>
        </table>
        <div role="dialog">dialog</div>
        <RunDrawer
          trace={traceRefOf(runs[0])}
          runs={runs}
          accessToken="sk"
          selection={selection}
          onSelect={onSelect}
          fullScreen={false}
          onFullScreenChange={vi.fn()}
        />
      </>,
    );
    fireEvent.mouseDown(screen.getByTestId("run-view"));
    fireEvent.mouseDown(screen.getByText("row"));
    fireEvent.mouseDown(screen.getByText("dialog"));
    fireEvent.mouseDown(screen.getByText("outside"), { button: 2 });
    expect(onSelect).not.toHaveBeenCalled();
    fireEvent.mouseDown(screen.getByText("outside"));
    expect(onSelect).toHaveBeenCalledExactlyOnceWith(null);
  });

  it("unmounts right away on close when the user prefers reduced motion", () => {
    mockReducedMotion(true);
    const runs = [run("a")];
    const { rerender } = renderWithProviders(
      <RunDrawer
        trace={traceRefOf(runs[0])}
        runs={runs}
        accessToken="sk"
        selection={selection}
        onSelect={vi.fn()}
        fullScreen={false}
        onFullScreenChange={vi.fn()}
      />,
    );
    expect(screen.getByRole("complementary", { name: "Trace details" })).toBeInTheDocument();
    rerender(
      <RunDrawer
        trace={null}
        runs={runs}
        accessToken="sk"
        selection={selection}
        onSelect={vi.fn()}
        fullScreen={false}
        onFullScreenChange={vi.fn()}
      />,
    );
    expect(screen.queryByRole("complementary", { name: "Trace details" })).not.toBeInTheDocument();
  });

  it("plays the exit animation before unmounting when motion is allowed", () => {
    mockReducedMotion(false);
    const runs = [run("a")];
    const { rerender } = renderWithProviders(
      <RunDrawer
        trace={traceRefOf(runs[0])}
        runs={runs}
        accessToken="sk"
        selection={selection}
        onSelect={vi.fn()}
        fullScreen={false}
        onFullScreenChange={vi.fn()}
      />,
    );
    rerender(
      <RunDrawer
        trace={null}
        runs={runs}
        accessToken="sk"
        selection={selection}
        onSelect={vi.fn()}
        fullScreen={false}
        onFullScreenChange={vi.fn()}
      />,
    );
    expect(screen.getByRole("complementary", { name: "Trace details" })).toHaveClass("animate-trace-drawer-out");
  });
});
