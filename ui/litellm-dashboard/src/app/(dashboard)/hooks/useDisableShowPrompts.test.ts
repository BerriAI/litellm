import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import { renderHook, waitFor } from "@testing-library/react";
import { useDisableShowPrompts } from "./useDisableShowPrompts";
import { LOCAL_STORAGE_EVENT } from "@/utils/localStorageUtils";

let mockPremiumUser: boolean | null = false;

vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({
  default: () => ({ premiumUser: mockPremiumUser }),
}));

describe("useDisableShowPrompts", () => {
  const STORAGE_KEY = "disableShowPrompts";

  beforeEach(() => {
    localStorage.clear();
    vi.clearAllMocks();
    mockPremiumUser = false;
  });

  afterEach(() => {
    localStorage.clear();
  });

  describe("non-premium user", () => {
    beforeEach(() => {
      mockPremiumUser = false;
    });

    it("should return false when localStorage is empty", () => {
      const { result } = renderHook(() => useDisableShowPrompts());
      expect(result.current).toBe(false);
    });

    it("should return false even when localStorage is 'true'", () => {
      localStorage.setItem(STORAGE_KEY, "true");
      const { result } = renderHook(() => useDisableShowPrompts());
      expect(result.current).toBe(false);
    });

    it("should return false even when localStorage is 'false'", () => {
      localStorage.setItem(STORAGE_KEY, "false");
      const { result } = renderHook(() => useDisableShowPrompts());
      expect(result.current).toBe(false);
    });
  });

  describe("premium user", () => {
    beforeEach(() => {
      mockPremiumUser = true;
    });

    it("should return true when localStorage is empty (default on)", () => {
      const { result } = renderHook(() => useDisableShowPrompts());
      expect(result.current).toBe(true);
    });

    it("should return true when localStorage value is 'true'", () => {
      localStorage.setItem(STORAGE_KEY, "true");
      const { result } = renderHook(() => useDisableShowPrompts());
      expect(result.current).toBe(true);
    });

    it("should return false when localStorage value is 'false' (explicitly disabled)", () => {
      localStorage.setItem(STORAGE_KEY, "false");
      const { result } = renderHook(() => useDisableShowPrompts());
      expect(result.current).toBe(false);
    });

    it("should update when storage event fires for the correct key", async () => {
      const { result } = renderHook(() => useDisableShowPrompts());
      expect(result.current).toBe(true); // default on

      localStorage.setItem(STORAGE_KEY, "false");
      const storageEvent = new StorageEvent("storage", {
        key: STORAGE_KEY,
        newValue: "false",
      });
      window.dispatchEvent(storageEvent);

      await waitFor(() => {
        expect(result.current).toBe(false);
      });
    });

    it("should not update when storage event fires for a different key", () => {
      localStorage.setItem(STORAGE_KEY, "false");
      const { result } = renderHook(() => useDisableShowPrompts());
      expect(result.current).toBe(false);

      const storageEvent = new StorageEvent("storage", {
        key: "otherKey",
        newValue: "true",
      });
      window.dispatchEvent(storageEvent);

      expect(result.current).toBe(false);
    });

    it("should update when custom LOCAL_STORAGE_EVENT fires for the correct key", async () => {
      localStorage.setItem(STORAGE_KEY, "false");
      const { result } = renderHook(() => useDisableShowPrompts());
      expect(result.current).toBe(false);

      localStorage.setItem(STORAGE_KEY, "true");
      const customEvent = new CustomEvent(LOCAL_STORAGE_EVENT, {
        detail: { key: STORAGE_KEY },
      });
      window.dispatchEvent(customEvent);

      await waitFor(() => {
        expect(result.current).toBe(true);
      });
    });

    it("should not update when custom LOCAL_STORAGE_EVENT fires for a different key", () => {
      localStorage.setItem(STORAGE_KEY, "false");
      const { result } = renderHook(() => useDisableShowPrompts());
      expect(result.current).toBe(false);

      const customEvent = new CustomEvent(LOCAL_STORAGE_EVENT, {
        detail: { key: "otherKey" },
      });
      window.dispatchEvent(customEvent);

      expect(result.current).toBe(false);
    });
  });

  describe("loading state (premiumUser is null)", () => {
    it("should return false while premium status is loading", () => {
      mockPremiumUser = null;
      const { result } = renderHook(() => useDisableShowPrompts());
      expect(result.current).toBe(false);
    });
  });

  it("should cleanup event listeners on unmount", () => {
    mockPremiumUser = true;
    const addEventListenerSpy = vi.spyOn(window, "addEventListener");
    const removeEventListenerSpy = vi.spyOn(window, "removeEventListener");

    const { unmount } = renderHook(() => useDisableShowPrompts());

    expect(addEventListenerSpy).toHaveBeenCalledWith("storage", expect.any(Function));
    expect(addEventListenerSpy).toHaveBeenCalledWith(LOCAL_STORAGE_EVENT, expect.any(Function));

    unmount();

    expect(removeEventListenerSpy).toHaveBeenCalledWith("storage", expect.any(Function));
    expect(removeEventListenerSpy).toHaveBeenCalledWith(LOCAL_STORAGE_EVENT, expect.any(Function));
  });
});
