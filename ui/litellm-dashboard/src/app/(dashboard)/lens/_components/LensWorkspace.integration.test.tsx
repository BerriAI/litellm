import { fireEvent, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { chooseSelectOption, renderWithProviders } from "@/../tests/test-utils";
import { LensWorkspace } from "./LensWorkspace";
import { createLensDemoData } from "./lensDemoData";

const network = vi.fn<typeof fetch>();
beforeEach(() => {
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
    await user.click(screen.getByRole("button", { name: "Try demo" }));
    expect(await screen.findByText("Where is order #1042?")).toBeVisible();
    expect(screen.getByText("Demo data")).toBeVisible();
    expect(screen.queryByRole("button", { name: "Set up tracing" })).not.toBeInTheDocument();
    network.mockClear();
    fireEvent.change(screen.getByPlaceholderText("Search input or trace ID"), { target: { value: "headphones" } });
    expect(screen.getByText("Can I return my headphones?")).toBeVisible();
    expect(screen.queryByText("Where is order #1042?")).not.toBeInTheDocument();
    fireEvent.change(screen.getByRole("textbox", { name: "Search runs" }), { target: { value: "" } });
    await chooseSelectOption(user, screen.getByRole("combobox", { name: "Filter by agent" }), "support_agent");
    await chooseSelectOption(user, screen.getByRole("combobox", { name: "Filter by status" }), "Failed");
    expect(within(screen.getByRole("table", { name: "Agent runs" })).getAllByRole("row")).toHaveLength(4);
    await user.click(screen.getByRole("tab", { name: "Investigations", exact: true }));
    expect(await screen.findByRole("button", { name: /Support quality/ })).toBeVisible();
    expect(screen.queryByRole("button", { name: "New investigation" })).not.toBeInTheDocument();
    expect(network).not.toHaveBeenCalled();
    expect(onUrlUpdate).not.toHaveBeenCalled();
    await user.click(screen.getByRole("button", { name: "Exit demo" }));
    expect(await screen.findByText("Enable tracing")).toBeVisible();
    expect(screen.queryByText("Demo data")).not.toBeInTheDocument();
    expect(screen.queryByText("Can I return my headphones?")).not.toBeInTheDocument();
  });

  it("connects findings and history to their original trace without live requests or URL changes", async () => {
    const user = userEvent.setup();
    const onUrlUpdate = vi.fn();
    const source = createLensDemoData().lenses[0];
    const saved = { ...source, id: "real-id", settings: { ...source.settings, name: "Existing investigation" } };
    network.mockImplementation(async (input) => {
      const path = new URL(String(input), "http://localhost").pathname;
      if (path === "/lens") return Response.json({ lenses: [saved], workers: [], tracing_enabled: true });
      if (path.endsWith("/runs")) return Response.json(saved.jobs);
      return Response.json({ data: [], traces: true, requests: false });
    });
    renderWithProviders(<LensWorkspace accessToken="live-token" userRole="Admin" readOnly={false} />, {
      searchParams: "?tab=investigations&lens=real-id",
      onUrlUpdate,
    });
    expect(await screen.findByRole("heading", { name: "Existing investigation" })).toBeVisible();
    await user.click(screen.getByRole("button", { name: "Try demo" }));
    await user.click(await screen.findByRole("button", { name: /Support quality/ }));
    network.mockClear();
    await user.click(screen.getByRole("button", { name: /Repeated lookups leave customers without an answer/ }));
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
      within(screen.getByRole("dialog", { name: "Original run" })).getByRole("button", { name: "Close", exact: true }),
    );
    await user.click(within(screen.getByRole("dialog")).getByRole("button", { name: "Close", exact: true }));
    await user.click(screen.getByRole("tab", { name: "History" }));
    await user.click(screen.getAllByRole("button", { name: /runs reviewed/ })[1]);
    expect(await screen.findByText(/1 linked run · high priority/)).toBeVisible();
    expect(network).not.toHaveBeenCalled();
    expect(onUrlUpdate).not.toHaveBeenCalled();
    await user.click(screen.getByRole("button", { name: "Exit demo" }));
    expect(await screen.findByRole("heading", { name: "Existing investigation" })).toBeVisible();
  });
});
