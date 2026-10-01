import React from "react";
import { render, renderHook, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { describe, expect, it, vi } from "vitest";
import { Select, SelectContent, SelectTrigger, SelectValue } from "@/components/ui/select";
import { TRANSPORT_ITEMS } from "@/components/mcp_tools/types";
import { useUIConfig } from "@/app/(dashboard)/hooks/uiConfig/useUIConfig";
import { STDIO_DISABLED_MESSAGE, TransportSelectItems, useMcpStdioEnabled } from "./StdioAvailability";

const getUiConfig = vi.hoisted(() => vi.fn());
vi.mock("@/components/networking", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/components/networking")>()),
  getUiConfig,
}));

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
  const renderWithConfig = (config: object) => {
    getUiConfig.mockResolvedValue(config);
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    const wrapper = ({ children }: { children: React.ReactNode }) => (
      <QueryClientProvider client={client}>{children}</QueryClientProvider>
    );
    return renderHook(() => ({ stdioEnabled: useMcpStdioEnabled(), config: useUIConfig() }), { wrapper });
  };

  it("reports stdio as enabled only when the proxy says so", async () => {
    const { result } = renderWithConfig({ mcp_stdio_enabled: true });

    await waitFor(() => expect(result.current.config.isSuccess).toBe(true));
    expect(result.current.stdioEnabled).toBe(true);
  });

  it.each([{ mcp_stdio_enabled: false }, {}])("treats %o as stdio disabled", async (config) => {
    const { result } = renderWithConfig(config);

    await waitFor(() => expect(result.current.config.isSuccess).toBe(true));
    expect(result.current.stdioEnabled).toBe(false);
  });
});
