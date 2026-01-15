import "@testing-library/jest-dom";
import { cleanup } from "@testing-library/react";
import React from "react";
import { afterEach, vi } from "vitest";

// Global mock for NotificationManager to prevent React rendering issues in tests
// This avoids "window is not defined" errors when notifications try to render
// after test environment is torn down
vi.mock("@/components/molecules/notifications_manager", () => ({
  default: {
    success: vi.fn(),
    fromBackend: vi.fn(),
    error: vi.fn(),
    warning: vi.fn(),
    info: vi.fn(),
    clear: vi.fn(),
  },
}));

vi.mock("@tremor/react", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@tremor/react")>();
  return {
    ...actual,
    Button: React.forwardRef<HTMLButtonElement, any>(({ children, ...props }, ref) =>
      // Render as a native button to avoid Tremor-specific behaviors in tests
      React.createElement("button", { ...props, ref }, children),
    ),
    Tooltip: ({ children, ..._props }: { children?: React.ReactNode; [key: string]: unknown }) => {
      // Return children directly without tooltip functionality to prevent flaky tests
      // This avoids issues with hover states, positioning, and DOM queries in tests
      return React.createElement(React.Fragment, null, children);
    },
  };
});

// Global mock for useAuthorized hook to avoid repeating the same mock in every test file
vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({
  default: () => ({
    token: "123",
    accessToken: "123",
    userId: "user-1",
    userEmail: "user@example.com",
    userRole: "Admin",
    premiumUser: false,
    disabledPersonalKeyCreation: null,
    showSSOBanner: false,
  }),
}));

afterEach(() => {
  cleanup();
});

// Make toLocaleString deterministic in tests; individual tests can override
// This returns ISO-like strings to keep assertions stable.
vi.spyOn(Date.prototype, "toLocaleString").mockImplementation(function (this: Date, ..._args: unknown[]) {
  const d = this;
  const pad = (n: number) => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`;
});

// Fixed matchMedia not found error in tests: https://github.com/vitest-dev/vitest/issues/821
Object.defineProperty(window, "matchMedia", {
  writable: true,
  value: (query: string) => ({
    matches: false,
    media: query,
    onchange: null,
    addListener: vi.fn(),
    removeListener: vi.fn(),
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    dispatchEvent: vi.fn(),
  }),
});

// Silence jsdom "getComputedStyle with pseudo-elements" not implemented warnings
// by ignoring the second argument and delegating to the native implementation.
const realGetComputedStyle = window.getComputedStyle.bind(window);
window.getComputedStyle = ((elt: Element) => realGetComputedStyle(elt)) as any;

// Avoid "navigation to another Document" warnings when clicking <a> with blob: URLs
// used by download flows in tests.
Object.defineProperty(HTMLAnchorElement.prototype, "click", {
  configurable: true,
  writable: true,
  value: vi.fn(),
});

if (!document.getAnimations) {
  document.getAnimations = () => [];
}

// Mock ResizeObserver for components that use it (e.g., Tremor UI components)
// This prevents "ResizeObserver is not defined" errors in JSDOM
global.ResizeObserver = class ResizeObserver {
  observe() {}
  unobserve() {}
  disconnect() {}
};
