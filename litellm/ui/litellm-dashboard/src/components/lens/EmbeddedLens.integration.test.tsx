import { fireEvent, screen } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { renderWithProviders } from "../../../tests/test-utils";
import { EmbeddedLens } from "./EmbeddedLens";

const network = vi.fn<typeof fetch>();
const connection = (configured = true, connected = true, browserUrl = "https://lens.example.test") =>
  Response.json({ configured, connected, browser_url: browserUrl });

beforeEach(() => {
  network.mockReset();
  vi.stubGlobal("fetch", network);
  window.history.replaceState(null, "", "/ui/lens/?tab=traces");
});

afterEach(() => {
  vi.unstubAllGlobals();
});

const render = () => renderWithProviders(<EmbeddedLens />);

it("frames the connected Lens at the tab in the dashboard URL", async () => {
  network.mockResolvedValueOnce(connection()).mockResolvedValueOnce(new Response("<html></html>"));
  render();
  expect(await screen.findByTitle("Lens")).toHaveAttribute("src", `${window.location.origin}/lens/ui/?tab=traces`);
});

it("shows the original introduction and linked sample evidence when Lens is not configured", async () => {
  network.mockResolvedValueOnce(connection(false, false));
  render();
  expect(await screen.findByRole("heading", { name: "The gateway that helps your agents improve" })).toBeVisible();
  expect(screen.getByRole("link", { name: "Set up Lens" })).toHaveAttribute(
    "href",
    "https://docs.litellm.ai/docs/proxy/lens",
  );
  expect(screen.getByRole("list", { name: "Sample trace timeline" })).toBeVisible();
  const finding = screen.getByRole("button", { name: "Claims numbers it never measured" });
  fireEvent.click(finding);
  expect(finding).toHaveAttribute("aria-pressed", "true");
  expect(screen.queryByRole("link", { name: "Open Lens" })).not.toBeInTheDocument();
  expect(screen.queryByRole("heading", { name: "Update Lens to use it here" })).not.toBeInTheDocument();
});

it("opens the workspace after setup succeeds and the user checks the connection again", async () => {
  network
    .mockResolvedValueOnce(connection(false, false))
    .mockResolvedValueOnce(connection())
    .mockResolvedValueOnce(new Response("<html></html>"));
  render();
  fireEvent.click(await screen.findByRole("button", { name: "Check connection" }));
  expect(await screen.findByTitle("Lens")).toBeVisible();
  expect(screen.queryByRole("heading", { name: "The gateway that helps your agents improve" })).not.toBeInTheDocument();
});

it("offers the standalone launch only for a connected older Lens with a browser address", async () => {
  network.mockResolvedValueOnce(connection()).mockResolvedValueOnce(new Response("", { status: 404 }));
  render();
  expect(await screen.findByRole("heading", { name: "Update Lens to use it here" })).toBeVisible();
  expect(screen.getByRole("link", { name: "Open Lens" })).toHaveAttribute(
    "href",
    `${window.location.origin}/lens/login`,
  );
  expect(screen.queryByTitle("Lens")).not.toBeInTheDocument();
});

it("does not offer the localhost launch when the gateway has no valid browser address", async () => {
  network.mockResolvedValueOnce(connection(true, true, "")).mockResolvedValueOnce(new Response("", { status: 404 }));
  render();
  expect(await screen.findByRole("heading", { name: "Update Lens to use it here" })).toBeVisible();
  expect(screen.queryByRole("link", { name: "Open Lens" })).not.toBeInTheDocument();
  expect(screen.getByText(/configure a Lens address that is accessible from your browser/)).toBeVisible();
});

it("shows a connection problem rather than installation or upgrade for a configured disconnected service", async () => {
  network.mockResolvedValueOnce(connection(true, false));
  render();
  expect(await screen.findByRole("heading", { name: "Could not connect to Lens" })).toBeVisible();
  expect(screen.queryByRole("link", { name: "Set up Lens" })).not.toBeInTheDocument();
  expect(screen.queryByRole("link", { name: "Open Lens" })).not.toBeInTheDocument();
});

it.each([401, 403, 404, 503])("does not mistake a failed connection check (%s) for missing setup", async (status) => {
  network.mockResolvedValueOnce(Response.json({ detail: "Unavailable" }, { status }));
  render();
  expect(await screen.findByRole("heading", { name: "Could not connect to Lens" })).toBeVisible();
  expect(screen.queryByRole("link", { name: "Set up Lens" })).not.toBeInTheDocument();
});

it("can retry a failed request without reloading the dashboard", async () => {
  network
    .mockRejectedValueOnce(new TypeError("network"))
    .mockResolvedValueOnce(connection())
    .mockResolvedValueOnce(new Response("<html></html>"));
  render();
  expect(await screen.findByRole("heading", { name: "Could not connect to Lens" })).toBeVisible();
  fireEvent.click(screen.getByRole("button", { name: "Check connection" }));
  expect(await screen.findByTitle("Lens")).toBeVisible();
});
