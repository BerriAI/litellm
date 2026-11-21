import { fireEvent, screen } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { renderWithProviders } from "../../../tests/test-utils";
import { registerAuthTokenGetter } from "@/lib/http/runtime";
import { setGlobalLitellmHeaderName } from "@/components/networking";
import { EmbeddedLens } from "./EmbeddedLens";

const network = vi.fn<typeof fetch>();

beforeEach(() => {
  network.mockReset();
  registerAuthTokenGetter(() => "gateway-session");
  setGlobalLitellmHeaderName("X-Gateway-Key");
  vi.stubGlobal("fetch", network);
  window.history.replaceState(null, "", "/ui/lens/?tab=traces");
});

afterEach(() => {
  setGlobalLitellmHeaderName();
  vi.unstubAllGlobals();
});

const render = () => renderWithProviders(<EmbeddedLens accessToken="gateway-session" userRole="Admin" readOnly />);

const frameWindow = async (): Promise<Window> => {
  const frame = await screen.findByTitle<HTMLIFrameElement>("Lens");
  if (!frame.contentWindow) throw new Error("The Lens frame has no window");
  return frame.contentWindow;
};

const say = (source: Window | null, data: unknown, origin = window.location.origin) =>
  fireEvent(window, new MessageEvent("message", { data, origin, source }));

it("frames the workspace the connected Lens serves through the gateway, at the tab the dashboard URL names", async () => {
  network.mockResolvedValue(new Response("<html></html>"));
  render();
  const frame = await screen.findByTitle("Lens");
  expect(frame).toHaveAttribute("src", `${window.location.origin}/lens/ui/?tab=traces`);
  expect(new URL(String(network.mock.calls[0]?.[0])).pathname).toBe("/lens/ui/");
});

it("gives the gateway session only to the frame it loaded, once that frame asks", async () => {
  network.mockResolvedValue(new Response("<html></html>"));
  render();
  const target = await frameWindow();
  const post = vi.spyOn(target, "postMessage");
  say(window, { type: "lens:ready", protocol: 1 });
  say(target, { type: "lens:ready", protocol: 1 }, "https://elsewhere.test");
  say(target, { type: "lens:ready", protocol: 2 });
  expect(post).not.toHaveBeenCalled();
  say(target, { type: "lens:ready", protocol: 1 });
  const session = {
    type: "lens:host",
    protocol: 1,
    apiBaseUrl: window.location.origin,
    authToken: "gateway-session",
    authHeaderName: "X-Gateway-Key",
    serverRootPath: "/",
    userRole: "Admin",
    readOnly: true,
  };
  expect(post).toHaveBeenCalledExactlyOnceWith(session, window.location.origin);
});

it("keeps the dashboard URL on the view the framed workspace moved to", async () => {
  network.mockResolvedValue(new Response("<html></html>"));
  render();
  say(await frameWindow(), { type: "lens:location", protocol: 1, search: "tab=agents&agent=a1" });
  expect(window.location.pathname + window.location.search).toBe("/ui/lens/?tab=agents&agent=a1");
});

it("offers Lens at its own address when the connected Lens is too old to serve a workspace", async () => {
  network.mockResolvedValue(new Response("", { status: 404 }));
  render();
  expect(await screen.findByRole("heading", { name: "Update Lens to use it here" })).toBeVisible();
  expect(screen.getByRole("link", { name: "Open Lens" })).toHaveAttribute(
    "href",
    `${window.location.origin}/lens/login`,
  );
  expect(screen.queryByTitle("Lens")).not.toBeInTheDocument();
});

it.each([
  ["the gateway cannot reach Lens", () => Promise.resolve(new Response("", { status: 503 }))],
  ["the request fails", () => Promise.reject(new TypeError("network"))],
])("says Lens is not reachable when %s", async (_case, respond) => {
  network.mockImplementation(respond);
  render();
  expect(await screen.findByRole("heading", { name: "Lens is not reachable" })).toBeVisible();
  expect(screen.queryByRole("link", { name: "Open Lens" })).not.toBeInTheDocument();
});
