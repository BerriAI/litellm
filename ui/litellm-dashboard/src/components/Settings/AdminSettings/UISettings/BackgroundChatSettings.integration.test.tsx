import { fireEvent, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { renderWithProviders, testQueryClient } from "@/../tests/test-utils";
import BackgroundChatSettings from "./BackgroundChatSettings";

let settings: {
  enabled: boolean;
  redis_configured: boolean;
  managed_by_config: boolean;
  can_save: boolean;
};
let failLoad: boolean;
let failSave: boolean;
const updates: unknown[] = [];

beforeEach(() => {
  testQueryClient.clear();
  settings = {
    enabled: false,
    redis_configured: true,
    managed_by_config: false,
    can_save: true,
  };
  failLoad = false;
  failSave = false;
  updates.length = 0;
  vi.stubGlobal(
    "fetch",
    vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const request =
        input instanceof Request ? input : new Request(new URL(String(input), window.location.origin), init);
      if (request.method === "PATCH") {
        const body = await request.json();
        updates.push(body);
        if (failSave) return Response.json({ detail: "Save failed" }, { status: 500 });
        settings = { ...settings, enabled: body.enabled };
      } else if (failLoad) return Response.json({ detail: "Unavailable" }, { status: 503 });
      return Response.json(settings);
    }),
  );
});
afterEach(() => vi.unstubAllGlobals());

it("saves background execution next to its explanation and retains it after reopening", async () => {
  const first = renderWithProviders(<BackgroundChatSettings />);
  fireEvent.click(await screen.findByRole("switch", { name: "Continue chats in the background" }));
  await waitFor(() => expect(screen.getByRole("switch")).toBeChecked());
  expect(updates).toEqual([{ enabled: true }]);
  expect(screen.getByRole("switch")).toBeChecked();
  first.unmount();
  testQueryClient.clear();
  renderWithProviders(<BackgroundChatSettings />);
  expect(await screen.findByRole("switch")).toBeChecked();
  fireEvent.click(screen.getByRole("switch"));
  await waitFor(() => expect(screen.getByRole("switch")).not.toBeChecked());
  expect(updates).toEqual([{ enabled: true }, { enabled: false }]);
});

it("links to Redis setup and prevents enabling without Redis", async () => {
  settings.redis_configured = false;
  renderWithProviders(<BackgroundChatSettings />);
  expect(await screen.findByRole("switch")).toHaveAttribute("aria-disabled", "true");
  expect(screen.getByRole("link", { name: /Configure it in Caching/ })).toHaveAttribute(
    "href",
    expect.stringMatching(/\/caching$/),
  );
  expect(updates).toHaveLength(0);
});

it.each(["managed_by_config", "can_save"] as const)(
  "explains a setting that cannot be edited because of %s",
  async (field) => {
    settings[field] = field === "managed_by_config";
    renderWithProviders(<BackgroundChatSettings />);
    expect(await screen.findByRole("switch")).toHaveAttribute("aria-disabled", "true");
    const message =
      field === "managed_by_config" ? /Managed by your config file/ : /Enable database-backed configuration/;
    expect(screen.getByText(message)).toBeVisible();
  },
);

it("keeps a failed save off", async () => {
  failSave = true;
  renderWithProviders(<BackgroundChatSettings />);
  fireEvent.click(await screen.findByRole("switch"));
  await waitFor(() => expect(updates).toHaveLength(1));
  expect(screen.getByRole("switch")).not.toBeChecked();
});

it("offers retry for a settings fetch failure instead of claiming Redis is missing", async () => {
  failLoad = true;
  renderWithProviders(<BackgroundChatSettings />);
  await screen.findByRole("button", { name: "Retry" });
  expect(screen.queryByRole("link")).not.toBeInTheDocument();
  failLoad = false;
  fireEvent.click(screen.getByRole("button", { name: "Retry" }));
  expect(await screen.findByRole("switch")).not.toHaveAttribute("aria-disabled", "true");
});
