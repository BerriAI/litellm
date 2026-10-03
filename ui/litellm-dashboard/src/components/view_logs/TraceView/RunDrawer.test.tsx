import { fireEvent, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { renderWithProviders } from "../../../../tests/test-utils";
import { clampDrawerWidth, RunDrawer } from "./RunDrawer";
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

const mockReducedMotion = (reduce: boolean) =>
  vi
    .spyOn(window, "matchMedia")
    .mockImplementation(
      (query: string) => ({ matches: reduce && query.includes("reduce"), media: query }) as MediaQueryList,
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

  it("fills the page across trace navigation and restores the resized drawer width", () => {
    const runs = [run("a"), run("b")];
    const onSelect = vi.fn();
    const { rerender } = renderWithProviders(
      <RunDrawer trace={runs[0]} runs={runs} accessToken="sk" onSelect={onSelect} />,
    );
    const drawer = screen.getByRole("complementary", { name: "Trace details" });
    fireEvent.keyDown(screen.getByRole("separator", { name: "Resize trace panel" }), { key: "ArrowLeft" });
    const resizedWidth = drawer.style.width;
    const storedWidth = window.localStorage.getItem("litellm.agentTraces.drawerWidth");

    fireEvent.click(screen.getByRole("button", { name: "Enter full screen" }));
    expect(drawer).toHaveStyle({ width: "100%" });
    expect(screen.queryByRole("separator", { name: "Resize trace panel" })).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Next trace (J)" }));
    expect(onSelect).toHaveBeenCalledWith(runs[1]);
    rerender(<RunDrawer trace={runs[1]} runs={runs} accessToken="sk" onSelect={onSelect} />);
    expect(drawer).toHaveStyle({ width: "100%" });
    expect(screen.getByTestId("run-view")).toHaveTextContent("run b");

    fireEvent.click(screen.getByRole("button", { name: "Exit full screen" }));
    expect(drawer).toHaveStyle({ width: resizedWidth });
    expect(screen.getByRole("separator", { name: "Resize trace panel" })).toBeInTheDocument();
    expect(window.localStorage.getItem("litellm.agentTraces.drawerWidth")).toBe(storedWidth);
  });

  it.each([false, true])("reopens at the saved width after closing full screen (reduced motion: %s)", (reduce) => {
    mockReducedMotion(reduce);
    const runs = [run("a"), run("b")];
    const onSelect = vi.fn();
    const { rerender } = renderWithProviders(
      <RunDrawer trace={runs[0]} runs={runs} accessToken="sk" onSelect={onSelect} />,
    );
    fireEvent.keyDown(screen.getByRole("separator", { name: "Resize trace panel" }), { key: "ArrowLeft" });
    const savedWidth = screen.getByRole("complementary", { name: "Trace details" }).style.width;
    fireEvent.click(screen.getByRole("button", { name: "Enter full screen" }));
    rerender(<RunDrawer trace={null} runs={runs} accessToken="sk" onSelect={onSelect} />);
    rerender(<RunDrawer trace={runs[1]} runs={runs} accessToken="sk" onSelect={onSelect} />);
    expect(screen.getByRole("complementary", { name: "Trace details" })).toHaveStyle({ width: savedWidth });
    expect(screen.getByRole("button", { name: "Enter full screen" })).toBeVisible();
    expect(screen.getByRole("separator", { name: "Resize trace panel" })).toBeVisible();
  });

  it.each(["Close (Esc)", "Close trace (Esc)"])("closes a full-screen trace using %s", (name) => {
    const runs = [run("a")];
    const onSelect = vi.fn();
    renderWithProviders(<RunDrawer trace={runs[0]} runs={runs} accessToken="sk" onSelect={onSelect} />);
    fireEvent.click(screen.getByRole("button", { name: "Enter full screen" }));
    fireEvent.click(screen.getByRole("button", { name }));
    expect(onSelect).toHaveBeenCalledExactlyOnceWith(null);
  });

  it("keeps Escape available to close a full-screen trace", () => {
    const runs = [run("a")];
    const onSelect = vi.fn();
    renderWithProviders(<RunDrawer trace={runs[0]} runs={runs} accessToken="sk" onSelect={onSelect} />);
    fireEvent.click(screen.getByRole("button", { name: "Enter full screen" }));
    fireEvent.keyDown(window, { key: "Escape" });
    expect(onSelect).toHaveBeenCalledExactlyOnceWith(null);
  });

  it("unmounts right away on close when the user prefers reduced motion", () => {
    mockReducedMotion(true);
    const runs = [run("a")];
    const { rerender } = renderWithProviders(
      <RunDrawer trace={runs[0]} runs={runs} accessToken="sk" onSelect={vi.fn()} />,
    );
    expect(screen.getByRole("complementary", { name: "Trace details" })).toBeInTheDocument();
    rerender(<RunDrawer trace={null} runs={runs} accessToken="sk" onSelect={vi.fn()} />);
    expect(screen.queryByRole("complementary", { name: "Trace details" })).not.toBeInTheDocument();
  });

  it("plays the exit animation before unmounting when motion is allowed", () => {
    mockReducedMotion(false);
    const runs = [run("a")];
    const { rerender } = renderWithProviders(
      <RunDrawer trace={runs[0]} runs={runs} accessToken="sk" onSelect={vi.fn()} />,
    );
    rerender(<RunDrawer trace={null} runs={runs} accessToken="sk" onSelect={vi.fn()} />);
    expect(screen.getByRole("complementary", { name: "Trace details" })).toHaveClass("animate-trace-drawer-out");
  });
});
