import { webcrypto } from "node:crypto";
import { afterEach, expect, it, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import ConnectFlowBanner from "./ConnectFlowBanner";

const locationDescriptor = Object.getOwnPropertyDescriptor(window, "location")!;

afterEach(() => {
  vi.unstubAllGlobals();
  window.sessionStorage.clear();
  Object.defineProperty(window, "location", locationDescriptor);
});

it("carries a challenged scope through the connect button into vendor authorization", async () => {
  vi.stubGlobal("crypto", webcrypto);
  vi.stubGlobal(
    "fetch",
    vi.fn(async () => Response.json({ client_id: "registered-client" })),
  );
  Object.defineProperty(window, "location", {
    configurable: true,
    value: {
      href: "https://gateway.example/ui/connect?connect_flow=flow-handle",
      origin: "https://gateway.example",
      pathname: "/ui/connect",
    },
  });
  render(
    <ConnectFlowBanner
      flowHandle="flow-handle"
      flow={{
        state: "interactive",
        client_origin: "https://client.example",
        server_id: "server-1",
        server_name: "Scoped tools",
        connected: false,
        requested_scopes: ["tools.write"],
      }}
      accessToken="caller"
      onConnected={vi.fn()}
      failed={false}
    />,
  );
  await waitFor(() => expect(window.location.href).toContain("scope=tools.write"));
  expect(new URL(window.location.href, window.location.origin).pathname).toBe(
    "/v1/mcp/server/oauth/server-1/authorize",
  );
  expect(screen.queryByRole("button", { name: /finish connecting/i })).not.toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Cancel" })).toBeInTheDocument();
});

it("starts consent for the next upstream in the same connect flow", async () => {
  vi.stubGlobal("crypto", webcrypto);
  vi.stubGlobal(
    "fetch",
    vi.fn(async () => Response.json({ client_id: "registered-client" })),
  );
  Object.defineProperty(window, "location", {
    configurable: true,
    value: {
      href: "https://gateway.example/ui/connect?connect_flow=flow-handle",
      origin: "https://gateway.example",
      pathname: "/ui/connect",
    },
  });
  const props = {
    flowHandle: "flow-handle",
    accessToken: "caller",
    onConnected: vi.fn(),
    failed: false,
  };
  const first = {
    state: "interactive" as const,
    client_origin: "https://client.example",
    server_id: "first",
    server_name: "First",
    connected: false,
    requested_scopes: ["first.write"],
  };
  const { rerender } = render(<ConnectFlowBanner {...props} flow={first} />);
  await waitFor(() => expect(window.location.href).toContain("scope=first.write"));
  rerender(
    <ConnectFlowBanner
      {...props}
      flow={{ ...first, server_id: "second", server_name: "Second", requested_scopes: ["second.write"] }}
    />,
  );
  await waitFor(() => expect(window.location.href).toContain("scope=second.write"));
  expect(new URL(window.location.href, window.location.origin).pathname).toBe("/v1/mcp/server/oauth/second/authorize");
  expect(screen.queryByRole("button", { name: /finish connecting/i })).not.toBeInTheDocument();
});
