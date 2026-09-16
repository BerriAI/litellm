import React from "react";
import userEvent from "@testing-library/user-event";
import { describe, it, expect, vi } from "vitest";
import { NuqsTestingAdapter, type OnUrlUpdateFunction } from "nuqs/adapters/testing";
import MCPConnect from "./mcp_connect";
import { render, renderWithProviders, screen, waitFor, within } from "@/../tests/test-utils";

vi.mock("@/components/networking", () => ({
  getProxyBaseUrl: vi.fn().mockReturnValue("http://localhost:4000"),
}));

const activePanel = () => screen.getByRole("tabpanel");

describe("MCPConnect (tab mount contract)", () => {
  it("keeps the x-mcp-servers header toggle on after switching tabs away and back", async () => {
    renderWithProviders(<MCPConnect />);

    await userEvent.click(screen.getByRole("tab", { name: "LiteLLM Proxy" }));
    expect(within(activePanel()).queryByText(/"x-mcp-servers":/)).not.toBeInTheDocument();

    await userEvent.click(screen.getByRole("switch"));
    expect(within(activePanel()).getByText(/"x-mcp-servers": "Zapier_MCP,dev-group"/)).toBeInTheDocument();

    await userEvent.click(screen.getByRole("tab", { name: "Cursor" }));
    await userEvent.click(screen.getByRole("tab", { name: "LiteLLM Proxy" }));

    expect(screen.getByRole("switch")).toBeChecked();
    expect(within(activePanel()).getByText(/"x-mcp-servers": "Zapier_MCP,dev-group"/)).toBeInTheDocument();
  });
});

describe("MCPConnect client tab URL state", () => {
  it("opens the client tab named in the URL", () => {
    renderWithProviders(<MCPConnect />, { searchParams: "?connect_client=cursor" });

    expect(screen.getByRole("tab", { name: "Cursor" })).toHaveAttribute("aria-selected", "true");
  });

  it("writes the client tab the user picks and drops the default", async () => {
    const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
    renderWithProviders(<MCPConnect />, { searchParams: "?tab=connect", onUrlUpdate });

    await userEvent.click(screen.getByRole("tab", { name: "LiteLLM Proxy" }));

    expect(onUrlUpdate.mock.calls.at(-1)?.[0].searchParams.get("connect_client")).toBe("litellm");
    expect(onUrlUpdate.mock.calls.at(-1)?.[0].searchParams.get("tab")).toBe("connect");
    expect(screen.getByRole("tab", { name: "LiteLLM Proxy" })).toHaveAttribute("aria-selected", "true");

    await userEvent.click(screen.getByRole("tab", { name: "OpenAI API" }));

    expect(onUrlUpdate.mock.calls.at(-1)?.[0].searchParams.has("connect_client")).toBe(false);
  });

  it("falls back to the first client and drops an unknown one from the URL", async () => {
    const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
    render(
      <NuqsTestingAdapter
        searchParams="?connect_client=vim"
        onUrlUpdate={onUrlUpdate}
        hasMemory
        resetUrlUpdateQueueOnMount={false}
      >
        <MCPConnect />
      </NuqsTestingAdapter>,
    );

    expect(screen.getByRole("tab", { name: "OpenAI API" })).toHaveAttribute("aria-selected", "true");
    await waitFor(() => expect(onUrlUpdate).toHaveBeenCalled());
    expect(onUrlUpdate.mock.calls.at(-1)?.[0].searchParams.has("connect_client")).toBe(false);
  });
});
