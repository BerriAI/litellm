import "@testing-library/jest-dom";
import { cleanup } from "@testing-library/react";
import React from "react";
import { afterEach, vi } from "vitest";

const ensureTestLocalStorage = () => {
  if (typeof window === "undefined" || typeof window.Storage === "undefined") {
    return;
  }

  if (typeof window.localStorage?.getItem === "function" && typeof window.localStorage?.clear === "function") {
    return;
  }

  const storageStores = new WeakMap<Storage, Map<string, string>>();
  const storagePrototype = window.Storage.prototype;
  const getStore = (storage: Storage) => {
    let store = storageStores.get(storage);
    if (store === undefined) {
      store = new Map<string, string>();
      storageStores.set(storage, store);
    }
    return store;
  };

  Object.defineProperties(storagePrototype, {
    getItem: {
      configurable: true,
      writable: true,
      value(this: Storage, key: string) {
        const store = getStore(this);
        const normalizedKey = String(key);
        return store.has(normalizedKey) ? store.get(normalizedKey)! : null;
      },
    },
    setItem: {
      configurable: true,
      writable: true,
      value(this: Storage, key: string, value: string) {
        const store = getStore(this);
        store.set(String(key), String(value));
      },
    },
    removeItem: {
      configurable: true,
      writable: true,
      value(this: Storage, key: string) {
        const store = getStore(this);
        store.delete(String(key));
      },
    },
    clear: {
      configurable: true,
      writable: true,
      value(this: Storage) {
        const store = getStore(this);
        store.clear();
      },
    },
    key: {
      configurable: true,
      writable: true,
      value(this: Storage, index: number) {
        const store = getStore(this);
        return Array.from(store.keys())[index] ?? null;
      },
    },
  });

  const localStorage = Object.create(storagePrototype);
  storageStores.set(localStorage, new Map<string, string>());
  Object.defineProperty(localStorage, "length", {
    configurable: true,
    get() {
      return getStore(localStorage).size;
    },
  });

  Object.defineProperty(window, "localStorage", {
    configurable: true,
    value: localStorage,
  });
  Object.defineProperty(globalThis, "localStorage", {
    configurable: true,
    value: localStorage,
  });
};

ensureTestLocalStorage();

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
    // Render as a plain checkbox so toggle interactions are testable without Tremor internals
    Switch: ({
      checked,
      onChange,
      className,
    }: {
      checked?: boolean;
      onChange?: (v: boolean) => void;
      className?: string;
    }) =>
      React.createElement("input", {
        type: "checkbox",
        role: "switch",
        checked,
        onChange: (e: React.ChangeEvent<HTMLInputElement>) => onChange?.(e.target.checked),
        className,
      }),
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

const pendingRefWarnings: string[] = [];
const consumePendingRefWarnings = (): string[] => pendingRefWarnings.splice(0, pendingRefWarnings.length);
(globalThis as { __consumePendingRefWarnings?: () => string[] }).__consumePendingRefWarnings =
  consumePendingRefWarnings;

const originalConsoleError = console.error.bind(console);
vi.spyOn(console, "error").mockImplementation((...args: unknown[]) => {
  originalConsoleError(...args);
  if (typeof args[0] === "string" && args[0].includes("Function components cannot be given refs")) {
    pendingRefWarnings.push(args.map(String).join(" "));
  }
});

afterEach(() => {
  cleanup();
  const refWarnings = consumePendingRefWarnings();
  if (refWarnings.length > 0) {
    throw new Error(
      "A ref was passed to a plain function component and silently dropped under React 18, which breaks " +
        "ref-based composition (Base UI render triggers, tooltips, focus). Wrap the component in React.forwardRef. " +
        "This tripwire lives in tests/setupTests.ts and can be removed after the React 19 upgrade.\n\n" +
        refWarnings.join("\n\n"),
    );
  }
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

// Stub URL.revokeObjectURL so vi.spyOn can intercept it in tests
if (!URL.revokeObjectURL) {
  URL.revokeObjectURL = () => {};
}

// Mock ResizeObserver for components that use it (recharts, Tremor UI components).
// JSDOM has no layout, so for observers inside a shadcn ChartContainer ([data-slot="chart"])
// the mock immediately reports a fixed 800x400 box; recharts renders nothing until it
// observes a size. Scoped to chart subtrees only: firing for every observer re-enters
// React mid-effect for tremor/headlessui consumers whose tests assume the old no-op
// (chart text would duplicate getByText targets, popover clicks go stale). Widen or
// drop the scoping once tremor is gone.
const MOCK_RESIZE_BOX = { inlineSize: 800, blockSize: 400 };
const MOCK_RESIZE_RECT: DOMRectReadOnly = {
  width: 800,
  height: 400,
  top: 0,
  left: 0,
  bottom: 400,
  right: 800,
  x: 0,
  y: 0,
  toJSON: () => ({}),
};
global.ResizeObserver = class ResizeObserver {
  private readonly callback: ResizeObserverCallback;
  constructor(callback: ResizeObserverCallback) {
    this.callback = callback;
  }
  observe(target: Element) {
    if (!target.closest('[data-slot="chart"]')) return;
    const entry: ResizeObserverEntry = {
      target,
      contentRect: MOCK_RESIZE_RECT,
      borderBoxSize: [MOCK_RESIZE_BOX],
      contentBoxSize: [MOCK_RESIZE_BOX],
      devicePixelContentBoxSize: [MOCK_RESIZE_BOX],
    };
    this.callback([entry], this);
  }
  unobserve() {}
  disconnect() {}
};
