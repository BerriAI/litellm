import { fireEvent, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { chooseSelectOption, renderWithProviders, testQueryClient } from "@/../tests/test-utils";
import { LensWorkspace } from "./LensWorkspace";
import { createLensDemoData } from "./demo/createLensDemo";

const network = vi.fn<typeof fetch>();
beforeEach(() => {
  testQueryClient.clear();
  vi.stubGlobal("fetch", network);
  network.mockReset();
  network.mockImplementation(async (input) => {
    const path = new URL(String(input), "http://localhost").pathname;
    if (path === "/v1/traces") return Response.json({ detail: "Tracing is not enabled" }, { status: 501 });
    if (path === "/lens") return Response.json({ lenses: [], workers: [], tracing_enabled: false });
    return Response.json({ data: [], traces: false, requests: false });
  });
});

describe("Lens interactive demo", () => {
  it("opens without tracing, filters sample runs, and restores the live view without mixing data", async () => {
    const user = userEvent.setup();
    const onUrlUpdate = vi.fn();
    renderWithProviders(<LensWorkspace accessToken="live-token" userRole="Internal User" readOnly={false} />, {
      onUrlUpdate,
    });
    expect(await screen.findByText("Enable tracing")).toBeVisible();
    await user.click(screen.getByRole("button", { name: "Preview sample" }));
    expect(await screen.findByText("Where is order #1042?")).toBeVisible();
    expect(screen.getByText("You’re viewing demo data")).toBeVisible();
    expect(screen.queryByRole("button", { name: "Set up tracing" })).not.toBeInTheDocument();
    network.mockClear();
    await user.click(screen.getByRole("button", { name: "Refresh" }));
    expect(screen.getByText("Where is order #1042?")).toBeVisible();
    fireEvent.change(screen.getByPlaceholderText("Search input or trace ID"), { target: { value: "headphones" } });
    expect(screen.getByText("Can I return my headphones?")).toBeVisible();
    expect(screen.queryByText("Where is order #1042?")).not.toBeInTheDocument();
    fireEvent.change(screen.getByRole("textbox", { name: "Search runs" }), { target: { value: "" } });
    await chooseSelectOption(user, screen.getByRole("combobox", { name: "Filter by agent" }), "support_agent");
    await chooseSelectOption(user, screen.getByRole("combobox", { name: "Filter by status" }), "Failed");
    expect(within(screen.getByRole("table", { name: "Agent runs" })).getAllByRole("row")).toHaveLength(4);
    await user.click(screen.getByRole("tab", { name: "Investigations" }));
    expect(await screen.findByRole("row", { name: /Support quality/ })).toBeVisible();
    expect(screen.queryByRole("button", { name: "New investigation" })).not.toBeInTheDocument();
    expect(network).not.toHaveBeenCalled();
    expect(onUrlUpdate).not.toHaveBeenCalled();
    await user.click(screen.getByRole("button", { name: "Exit demo" }));
    expect(await screen.findByText("Enable tracing")).toBeVisible();
    expect(screen.queryByText("You’re viewing demo data")).not.toBeInTheDocument();
    expect(screen.queryByText("Can I return my headphones?")).not.toBeInTheDocument();
  });

  it("connects findings and history to their original trace without live requests or URL changes", async () => {
    const user = userEvent.setup();
    const onUrlUpdate = vi.fn();
    renderWithProviders(<LensWorkspace accessToken="live-token" userRole="Admin" readOnly={false} />, {
      searchParams: "?tab=investigations",
      onUrlUpdate,
    });
    await screen.findByRole("button", { name: "Preview sample" });
    await user.click(screen.getByRole("button", { name: "Preview sample" }));
    await user.click(await screen.findByRole("tab", { name: "Findings" }));
    network.mockClear();
    await user.click(await screen.findByRole("row", { name: /Repeated lookups leave customers without an answer/ }));
    const finding = screen.getByRole("dialog");
    expect(within(finding).getByText(/The support agent retries/)).toBeVisible();
    const summaries = within(finding).getAllByText("support_agent", { exact: true });
    await user.click(summaries[0]);
    await user.click(within(finding).getAllByRole("button", { name: /Open original step/ })[0]);
    expect(await screen.findByRole("complementary", { name: "Span details" })).toHaveTextContent(
      "I will check that for you.",
    );
    await user.click(screen.getByRole("button", { name: "Copy for agent" }));
    expect(await navigator.clipboard.readText()).toContain("I will check that for you.");
    expect(await navigator.clipboard.readText()).not.toContain("Authorization");
    await user.click(screen.getByRole("tab", { name: "Attributes" }));
    expect(await screen.findByText("gen_ai.agent.name")).toBeVisible();
    await user.click(
      within(screen.getByRole("dialog", { name: "Original run" })).getByRole("button", { name: "Close" }),
    );
    await user.click(within(screen.getByRole("dialog")).getByRole("button", { name: "Close" }));
    expect(await screen.findByRole("table", { name: "Findings" })).toBeVisible();
    expect(network).not.toHaveBeenCalled();
    expect(onUrlUpdate).not.toHaveBeenCalled();
    await user.click(screen.getByRole("button", { name: "Exit demo" }));
    expect(await screen.findByRole("heading", { name: "Find what needs attention" })).toBeVisible();
  });

  it("has no demo entry for existing investigations, populated traces, or connecting another agent", async () => {
    const user = userEvent.setup();
    const data = createLensDemoData();
    const saved = data.lenses[0];
    network.mockImplementation(async (input) => {
      const path = new URL(String(input), "http://localhost").pathname;
      if (path === "/lens") return Response.json({ lenses: [saved], workers: [], tracing_enabled: true });
      if (path.endsWith("/runs")) return Response.json(saved.jobs);
      if (path === "/v1/traces") return Response.json({ data: data.runs.map((run) => run.trace.summary) });
      return Response.json({ data: [], traces: true, requests: false });
    });
    renderWithProviders(<LensWorkspace accessToken="live-token" userRole="Admin" readOnly={false} />, {
      searchParams: `?tab=investigations&lens=${saved.id}`,
    });
    expect(await screen.findByRole("heading", { name: saved.settings.name })).toBeVisible();
    expect(screen.queryByRole("button", { name: "Preview sample" })).not.toBeInTheDocument();
    await user.click(within(screen.getByRole("tablist", { name: "Lens" })).getByRole("tab", { name: "Traces" }));
    expect(await screen.findByText("Where is order #1042?")).toBeVisible();
    expect(screen.queryByRole("button", { name: "Preview sample" })).not.toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Set up tracing" }));
    expect(await screen.findByRole("heading", { name: "Connect another agent" })).toBeVisible();
    expect(screen.queryByRole("button", { name: "Preview sample" })).not.toBeInTheDocument();
  });

  it("shows the header preview only for the active tab that still needs setup", async () => {
    const user = userEvent.setup();
    const saved = createLensDemoData().lenses[0];
    network.mockImplementation(async (input) => {
      const path = new URL(String(input), "http://localhost").pathname;
      if (path === "/lens") return Response.json({ lenses: [saved], workers: [], tracing_enabled: false });
      if (path.endsWith("/runs")) return Response.json(saved.jobs);
      if (path === "/v1/traces") return Response.json({ detail: "Tracing is not enabled" }, { status: 501 });
      return Response.json({ data: [], traces: false, requests: false });
    });
    renderWithProviders(<LensWorkspace accessToken="live-token" userRole="Admin" readOnly={false} />);
    expect(await screen.findByRole("button", { name: "Preview sample" })).toBeVisible();
    const tabs = within(screen.getByRole("tablist", { name: "Lens" }));
    await user.click(tabs.getByRole("tab", { name: "Investigations" }));
    expect(await screen.findByRole("row", { name: new RegExp(saved.settings.name) })).toBeVisible();
    expect(screen.queryByRole("button", { name: "Preview sample" })).not.toBeInTheDocument();
    await user.click(tabs.getByRole("tab", { name: "Traces" }));
    await user.click(await screen.findByRole("button", { name: "Preview sample" }));
    expect(await screen.findByRole("table", { name: "Agent runs" })).toBeVisible();
    expect(screen.queryByRole("button", { name: "Preview sample" })).not.toBeInTheDocument();
  });
});
