import { screen } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { renderWithProviders } from "../../../tests/test-utils";
import { EmbeddedLens } from "./EmbeddedLens";

const network = vi.fn<typeof fetch>();

beforeEach(() => {
  network.mockReset();
  vi.stubGlobal("fetch", network);
  window.history.replaceState(null, "", "/ui/lens/?tab=traces");
});

afterEach(() => {
  vi.unstubAllGlobals();
});

const render = () => renderWithProviders(<EmbeddedLens />);

it("frames the UI the connected Lens serves through the gateway, at the tab the dashboard URL names", async () => {
  network.mockResolvedValue(new Response("<html></html>"));
  render();
  const frame = await screen.findByTitle("Lens");
  expect(frame).toHaveAttribute("src", `${window.location.origin}/lens/ui/?tab=traces`);
  expect(new URL(String(network.mock.calls[0]?.[0])).pathname).toBe("/lens/ui/");
});

it("offers Lens at its own address when the connected Lens is too old to be shown here", async () => {
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
