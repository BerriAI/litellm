import "@testing-library/jest-dom";
import { cleanup } from "@testing-library/react";
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

// Global mock so every test can assert on toast calls; toast.test.ts opts back in with vi.unmock
vi.mock("@/lib/toast", () => ({
  toast: {
    success: vi.fn(),
    info: vi.fn(),
    warning: vi.fn(),
    error: vi.fn(),
    fromError: vi.fn(),
    dismiss: vi.fn(),
  },
}));

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

// Unmounting a Base UI dialog that is still open leaves its scroll lock behind: the <html>
// and <body> inline styles and the marker attribute survive cleanup() and make every later
// test in the file see a locked page, where popups compute pointer-events: none and clicks
// silently do nothing. Real users never unmount an open dialog, so undo it here.
const releaseBaseUiScrollLock = () => {
  const root = document.documentElement;
  if (!root.hasAttribute("data-base-ui-scroll-locked")) return;
  root.removeAttribute("data-base-ui-scroll-locked");
  for (const property of ["scrollbar-gutter", "overflow-y", "overflow-x", "scroll-behavior"]) {
    root.style.removeProperty(property);
  }
  for (const property of ["position", "height", "width", "box-sizing", "overflow", "scroll-behavior"]) {
    document.body.style.removeProperty(property);
  }
};

afterEach(() => {
  cleanup();
  releaseBaseUiScrollLock();
});

// Make toLocaleString deterministic in tests; individual tests can override
// This returns ISO-like strings to keep assertions stable.
vi.spyOn(Date.prototype, "toLocaleString").mockImplementation(function (this: Date, ..._args: unknown[]) {
  const d = this;
  const pad = (n: number) => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`;
});

if (typeof window !== "undefined") {
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

  // Base UI's ScrollAreaViewport calls viewport.getAnimations() from a timer, which jsdom
  // does not implement, so the TypeError surfaces as an unhandled error and fails the run.
  // BASE_UI_ANIMATIONS_DISABLED keeps useAnimationsFinished on the synchronous path it
  // already took while getAnimations was missing, so popup unmount timing is unchanged.
  (globalThis as { BASE_UI_ANIMATIONS_DISABLED?: boolean }).BASE_UI_ANIMATIONS_DISABLED = true;
  if (!Element.prototype.getAnimations) {
    Element.prototype.getAnimations = () => [];
  }

  // Stub URL.revokeObjectURL so vi.spyOn can intercept it in tests
  if (!URL.revokeObjectURL) {
    URL.revokeObjectURL = () => {};
  }

  // Mock ResizeObserver for components that use it (recharts, Tremor UI components).
  // JSDOM has no layout, so for observers inside a shadcn ChartContainer ([data-slot="chart"])
  // the mock immediately reports a fixed 800x400 box; recharts renders nothing until it
  // observes a size. Scoped to chart subtrees only: firing for every observer re-enters
  // React mid-effect for headlessui consumers whose tests assume the old no-op
  // (chart text would duplicate getByText targets, popover clicks go stale).
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
}
