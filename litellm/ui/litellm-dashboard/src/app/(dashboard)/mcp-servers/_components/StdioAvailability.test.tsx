import React from "react";
import { render, renderHook, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, describe, expect, it, vi } from "vitest";
import { Select, SelectContent, SelectTrigger, SelectValue } from "@/components/ui/select";
import { TRANSPORT_ITEMS } from "@/components/mcp_tools/types";
import { switchToWorkerUrl } from "@/components/networking";
import { STDIO_DISABLED_MESSAGE, TransportSelectItems, useMcpStdioEnabled } from "./StdioAvailability";

function openTransportSelect(stdioEnabled: boolean) {
  render(
    <Select items={TRANSPORT_ITEMS} value={null} defaultOpen>
      <SelectTrigger data-testid="trigger">
        <SelectValue placeholder="Select transport" />
      </SelectTrigger>
      <SelectContent>
        <TransportSelectItems stdioEnabled={stdioEnabled} />
      </SelectContent>
    </Select>,
  );
}

const option = (name: RegExp) => screen.getByRole("option", { name });

describe("TransportSelectItems", () => {
  it("greys out only the stdio option and explains how to enable it when stdio is off", async () => {
    const user = userEvent.setup();
    openTransportSelect(false);

    expect(option(/Standard Input\/Output \(stdio\)/)).toHaveAttribute("data-disabled");
    expect(option(/Streamable HTTP/)).not.toHaveAttribute("data-disabled");
    expect(option(/Server-Sent Events/)).not.toHaveAttribute("data-disabled");
    expect(option(/OpenAPI Spec/)).not.toHaveAttribute("data-disabled");

    await user.hover(screen.getByLabelText("question-circle"));

    expect(await screen.findByText(STDIO_DISABLED_MESSAGE)).toBeInTheDocument();
  });

  it("offers stdio like any other transport when stdio is on", () => {
    openTransportSelect(true);

    expect(option(/Standard Input\/Output \(stdio\)/)).not.toHaveAttribute("data-disabled");
    expect(screen.queryByLabelText("question-circle")).not.toBeInTheDocument();
  });
});

describe("useMcpStdioEnabled", () => {
  afterEach(() => {
    switchToWorkerUrl(null);
    vi.restoreAllMocks();
  });

  const renderWithConfig = (config: object) => {
    const fetchSpy = vi
      .spyOn(globalThis, "fetch")
      .mockImplementation(async () => new Response(JSON.stringify(config), { status: 200 }));
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    const wrapper = ({ children }: { children: React.ReactNode }) => (
      <QueryClientProvider client={client}>{children}</QueryClientProvider>
    );
    const { result } = renderHook(() => useMcpStdioEnabled(), { wrapper });
    const settled = () =>
      waitFor(() =>
        expect(
          client
            .getQueryCache()
            .getAll()
            .map((query) => query.state.status),
        ).toEqual(["success"]),
      );
    return { result, fetchSpy, settled };
  };

  it("reads the flag from the worker the dashboard is managing", async () => {
    switchToWorkerUrl("http://worker-b.example:4000");
    const { result, fetchSpy, settled } = renderWithConfig({ mcp_stdio_enabled: false });

    await settled();
    expect(result.current).toBe(false);
    expect(fetchSpy.mock.calls.map(([request]) => (request as Request).url)).toEqual([
      "http://worker-b.example:4000/.well-known/litellm-ui-config",
    ]);
  });

  it.each([
    [{ mcp_stdio_enabled: false }, false],
    [{ mcp_stdio_enabled: true }, true],
    [{}, true],
  ])("treats %o as stdio enabled=%s", async (config, enabled) => {
    const { result, settled } = renderWithConfig(config);

    await settled();
    expect(result.current).toBe(enabled);
  });

  it("keeps stdio available while the proxy has not answered yet", async () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch").mockImplementation(() => new Promise<Response>(() => {}));
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    const { result } = renderHook(() => useMcpStdioEnabled(), {
      wrapper: ({ children }: { children: React.ReactNode }) => (
        <QueryClientProvider client={client}>{children}</QueryClientProvider>
      ),
    });

    await waitFor(() => expect(fetchSpy).toHaveBeenCalled());
    expect(result.current).toBe(true);
  });
});
