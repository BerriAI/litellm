import React from "react";
import { render, screen, waitFor } from "@testing-library/react";
import { vi, describe, it, beforeEach, afterEach, expect } from "vitest";

/** ----------------------------
 * Hoisted helpers for mocks (required by Vitest)
 * --------------------------- */
const { stub, jwtDecodeMock } = vi.hoisted(() => {
  const React = require("react");
  const stub = (name: string) => () => React.createElement("div", { "data-testid": name });
  return {
    stub,
    jwtDecodeMock: vi.fn(),
  };
});

/** ----------------------------
 * Mocks
 * --------------------------- */

vi.mock("@/hooks/useFeatureFlags", () => {
  const React = require("react");

  // minimal context so useFeatureFlags() returns something stable
  const FeatureFlagsCtx = React.createContext({ get: () => false, flags: {} });

  // Defensive provider: handle undefined props and allow optional value override
  const FeatureFlagsProvider = (props: any) => {
    const p = props || {};
    const value = p.value ?? { get: () => false, flags: {} };
    return React.createElement(FeatureFlagsCtx.Provider, { value }, p.children);
  };

  const useFeatureFlags = () => React.useContext(FeatureFlagsCtx);

  return {
    __esModule: true,
    default: FeatureFlagsProvider, // supports default import
    FeatureFlagsProvider, // supports named import
    useFeatureFlags, // supports named import
  };
});

// next/navigation mock: search params + router + pathname
vi.mock("next/navigation", () => {
  const router = {
    push: vi.fn(),
    replace: vi.fn(),
    prefetch: vi.fn(),
    back: vi.fn(),
    forward: vi.fn(),
    refresh: vi.fn(),
  };

  return {
    __esModule: true,
    // what your tests already relied on
    useSearchParams: () => new URLSearchParams(""),
    // added: satisfies useAuthorized / SidebarProvider
    useRouter: () => router,
    // optional helpers some components often read
    usePathname: () => "/",
    // optional: noop versions if code calls them
    redirect: vi.fn(), // App Router server action usually; safe noop here
    notFound: vi.fn(),
  };
});

// Networking layer
vi.mock("@/components/networking", () => {
  return {
    // Called on mount; we don't care about its contents, only that it resolves
    getUiConfig: vi.fn().mockResolvedValue({}),
    // Used to build the redirect URL
    proxyBaseUrl: "https://example.com",
    // Called when decoding a valid token
    setGlobalLitellmHeaderName: vi.fn(),
    Organization: {},
  };
});

// jwt-decode: we’ll swap implementation per test via mockImplementation
vi.mock("jwt-decode", () => ({
  jwtDecode: (token: string) => jwtDecodeMock(token),
}));

// Super-light stubs for all heavy components so rendering doesn't explode
vi.mock("@/components/navbar", () => ({ default: stub("navbar") }));
vi.mock("@/components/user_dashboard", () => ({ default: stub("user-dashboard") }));
vi.mock("@/components/templates/model_dashboard", () => ({ default: stub("model-dashboard") }));
vi.mock("@/components/view_users", () => ({ default: stub("view-users") }));
vi.mock("@/components/teams", () => ({ default: stub("teams") }));
vi.mock("@/components/organizations", () => ({
  default: stub("organizations"),
  fetchOrganizations: vi.fn(), // consumed in effects
}));
vi.mock("@/components/admins", () => ({ default: stub("admin-panel") }));
vi.mock("@/components/settings", () => ({ default: stub("settings") }));
vi.mock("@/components/general_settings", () => ({ default: stub("general-settings") }));
vi.mock("@/components/pass_through_settings", () => ({ default: stub("pass-through-settings") }));
vi.mock("@/components/budgets/budget_panel", () => ({ default: stub("budget-panel") }));
vi.mock("@/components/view_logs", () => ({ default: stub("spend-logs") }));
vi.mock("@/components/model_hub_table", () => ({ default: stub("model-hub-table") }));
vi.mock("@/components/new_usage", () => ({ default: stub("new-usage") }));
vi.mock("@/components/api_ref", () => ({ default: stub("api-ref") }));
vi.mock("@/components/chat_ui/ChatUI", () => ({ default: stub("chat-ui") }));
vi.mock("@/components/leftnav", () => ({ default: stub("sidebar") }));
vi.mock("@/components/usage", () => ({ default: stub("usage") }));
vi.mock("@/components/cache_dashboard", () => ({ default: stub("cache-dashboard") }));
vi.mock("@/components/guardrails", () => ({ default: stub("guardrails") }));
vi.mock("@/components/prompts", () => ({ default: stub("prompts") }));
vi.mock("@/components/transform_request", () => ({ default: stub("transform-request") }));
vi.mock("@/components/mcp_tools", () => ({ MCPServers: stub("mcp-servers") }));
vi.mock("@/components/tag_management", () => ({ default: stub("tag-management") }));
vi.mock("@/components/vector_store_management", () => ({ default: stub("vector-stores") }));
vi.mock("@/components/ui_theme_settings", () => ({ default: stub("ui-theme-settings") }));
vi.mock("@/components/organisms/create_key_button", () => ({ fetchUserModels: vi.fn() }));
vi.mock("@/components/common_components/fetch_teams", () => ({ fetchTeams: vi.fn() }));
vi.mock("@/components/ui/ui-loading-spinner", () => ({
  UiLoadingSpinner: stub("spinner"),
}));
vi.mock("@/contexts/ThemeContext", () => {
  const React = require("react");
  return {
    ThemeProvider: ({ children }: any) => React.createElement(React.Fragment, null, children),
  };
});
vi.mock("@/lib/cva.config", () => ({
  cx: (...args: string[]) => args.join(" "),
}));

import CreateKeyPage from "@/app/page";

/** ----------------------------
 * Helpers
 * --------------------------- */

function setCookie(raw: string) {
  // JSDOM allows simple string assignment to document.cookie
  document.cookie = raw;
}

function clearAllCookies() {
  // JSDOM doesn't give an API to clear; overwrite with empty string
  // plus ensure we wipe known names used by this app.
  document.cookie = "token=; Max-Age=0; Path=/";
}

const originalLocation = window.location;

beforeEach(() => {
  // Fresh module state & DOM
  vi.clearAllMocks();
  clearAllCookies();

  // Make location.replace spy-able to validate redirect
  delete (window as any).location;
  // minimal location object with replace and assign stubs
  (window as any).location = {
    ...originalLocation,
    href: "http://localhost/",
    assign: vi.fn(),
    replace: vi.fn(),
  };
});

afterEach(() => {
  // Restore location to avoid leaking across test envs
  delete (window as any).location;
  (window as any).location = originalLocation;
});

/** ----------------------------
 * Tests
 * --------------------------- */

describe("CreateKeyPage auth behavior", () => {
  it("redirects to SSO when cookie token is expired and clears it (no spasms)", async () => {
    // Arrange: expired token in cookie
    setCookie("token=expiredtoken");

    // jwtDecode returns past exp → expired
    jwtDecodeMock.mockImplementation((tok: string) => {
      expect(tok).toBe("expiredtoken");
      return { exp: Math.floor(Date.now() / 1000) - 60 }; // expired 60s ago
    });

    // Spy on cookie writes to ensure we clear with Max-Age=0
    const cookieSetSpy = vi.spyOn(document, "cookie", "set");

    // Act
    render(<CreateKeyPage />);

    // Assert: we eventually redirect to SSO login (single replace, not assign/href)
    await waitFor(() => {
      expect(window.location.replace).toHaveBeenCalledWith("https://example.com/sso/key/generate");
    });

    // And we attempted to clear the cookie (defensive deletion)
    const wroteDeletion = cookieSetSpy.mock.calls.some(
      (args) => typeof args[0] === "string" && args[0].includes("Max-Age=0") && args[0].startsWith("token="),
    );
    expect(wroteDeletion).toBe(true);
  });

  it("does NOT redirect when token is valid and renders the app chrome", async () => {
    // Arrange: valid token in cookie
    setCookie("token=validtoken");

    // jwtDecode returns future exp and expected shape
    jwtDecodeMock.mockImplementation((tok: string) => {
      expect(tok).toBe("validtoken");
      return {
        exp: Math.floor(Date.now() / 1000) + 60 * 60, // 1h in the future
        key: "accessKey-123",
        user_role: "app_user",
        user_email: "user@example.com",
        login_method: "username_password",
        premium_user: false,
        auth_header_name: "x-litellm-auth",
        user_id: "u_123",
      };
    });

    // Act
    render(<CreateKeyPage />);

    // Assert: no redirect
    await waitFor(() => {
      expect(window.location.replace).not.toHaveBeenCalled();
    });

    // And some top-level UI appears (Navbar stub)
    await waitFor(() => {
      expect(screen.getByTestId("navbar")).toBeInTheDocument();
    });
  });
});
