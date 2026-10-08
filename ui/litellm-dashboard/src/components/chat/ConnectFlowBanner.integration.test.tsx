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
