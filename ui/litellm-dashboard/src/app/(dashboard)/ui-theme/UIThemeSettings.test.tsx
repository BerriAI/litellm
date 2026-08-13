import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import NotificationsManager from "@/components/molecules/notifications_manager";

import UIThemeSettings from "./UIThemeSettings";

const setLogoUrl = vi.fn();
const setFaviconUrl = vi.fn();

vi.mock("@/contexts/ThemeContext", () => ({
  useTheme: () => ({ logoUrl: null, setLogoUrl, faviconUrl: null, setFaviconUrl }),
}));

vi.mock("@/components/networking", () => ({
  getProxyBaseUrl: () => "",
  getGlobalLitellmHeaderName: () => "Authorization",
}));

vi.mock("@/components/molecules/notifications_manager", () => ({
  __esModule: true,
  default: { success: vi.fn(), fromBackend: vi.fn() },
}));

const LOGO_PLACEHOLDER = "https://example.com/logo.png";
const FAVICON_PLACEHOLDER = "https://example.com/favicon.ico";

const okResponse = (values: Record<string, string | null> = {}) =>
  Promise.resolve({ ok: true, json: () => Promise.resolve({ values }) } as Response);

const fetchMock = vi.fn<typeof fetch>();

const patchCalls = () => fetchMock.mock.calls.filter(([, init]) => init?.method === "PATCH");

const bodyOf = (call: Parameters<typeof fetch>) => JSON.parse(String(call[1]?.body));

describe("UIThemeSettings", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    fetchMock.mockImplementation(() => okResponse());
    vi.stubGlobal("fetch", fetchMock);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("should render nothing without an access token", () => {
    const { container } = render(<UIThemeSettings userID="user-1" userRole="Admin" accessToken={null} />);

    expect(container).toBeEmptyDOMElement();
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("should load the saved logo and favicon urls into the inputs", async () => {
    fetchMock.mockImplementation(() =>
      okResponse({ logo_url: "https://cdn.example.com/logo.svg", favicon_url: "https://cdn.example.com/fav.ico" }),
    );

    render(<UIThemeSettings userID="user-1" userRole="Admin" accessToken="sk-test" />);

    await waitFor(() => {
      expect(screen.getByPlaceholderText(LOGO_PLACEHOLDER)).toHaveValue("https://cdn.example.com/logo.svg");
    });
    expect(screen.getByPlaceholderText(FAVICON_PLACEHOLDER)).toHaveValue("https://cdn.example.com/fav.ico");
    expect(setLogoUrl).toHaveBeenCalledWith("https://cdn.example.com/logo.svg");
    expect(setFaviconUrl).toHaveBeenCalledWith("https://cdn.example.com/fav.ico");
  });

  it("should save the entered urls and report success", async () => {
    const user = userEvent.setup();
    render(<UIThemeSettings userID="user-1" userRole="Admin" accessToken="sk-test" />);

    await waitFor(() => expect(fetchMock).toHaveBeenCalled());

    await user.type(screen.getByPlaceholderText(LOGO_PLACEHOLDER), "https://a.test/logo.png");
    await user.type(screen.getByPlaceholderText(FAVICON_PLACEHOLDER), "https://a.test/fav.ico");
    await user.click(screen.getByRole("button", { name: "Save Changes" }));

    await waitFor(() => expect(patchCalls()).toHaveLength(1));
    expect(bodyOf(patchCalls()[0])).toEqual({
      logo_url: "https://a.test/logo.png",
      favicon_url: "https://a.test/fav.ico",
    });
    await waitFor(() =>
      expect(NotificationsManager.success).toHaveBeenCalledWith("Theme settings updated successfully!"),
    );
  });

  it("should surface a backend failure when saving fails", async () => {
    const user = userEvent.setup();
    render(<UIThemeSettings userID="user-1" userRole="Admin" accessToken="sk-test" />);

    await waitFor(() => expect(fetchMock).toHaveBeenCalled());
    fetchMock.mockImplementation(() => Promise.resolve({ ok: false } as Response));

    await user.click(screen.getByRole("button", { name: "Save Changes" }));

    await waitFor(() =>
      expect(NotificationsManager.fromBackend).toHaveBeenCalledWith("Failed to update theme settings"),
    );
    expect(NotificationsManager.success).not.toHaveBeenCalled();
  });

  it("should clear both inputs and persist nulls when resetting to default", async () => {
    const user = userEvent.setup();
    fetchMock.mockImplementation(() =>
      okResponse({ logo_url: "https://cdn.example.com/logo.svg", favicon_url: "https://cdn.example.com/fav.ico" }),
    );

    render(<UIThemeSettings userID="user-1" userRole="Admin" accessToken="sk-test" />);

    await waitFor(() => {
      expect(screen.getByPlaceholderText(LOGO_PLACEHOLDER)).toHaveValue("https://cdn.example.com/logo.svg");
    });

    await user.click(screen.getByRole("button", { name: "Reset to Default" }));

    await waitFor(() => expect(patchCalls()).toHaveLength(1));
    expect(bodyOf(patchCalls()[0])).toEqual({ logo_url: null, favicon_url: null });
    expect(screen.getByPlaceholderText(LOGO_PLACEHOLDER)).toHaveValue("");
    expect(screen.getByPlaceholderText(FAVICON_PLACEHOLDER)).toHaveValue("");
    expect(setLogoUrl).toHaveBeenLastCalledWith(null);
    expect(setFaviconUrl).toHaveBeenLastCalledWith(null);
    await waitFor(() => expect(NotificationsManager.success).toHaveBeenCalledWith("Theme settings reset to default!"));
  });
});
