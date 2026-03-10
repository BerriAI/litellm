import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import { act, renderHook, waitFor } from "@testing-library/react";
import { useDisableUsageIndicator } from "./useDisableUsageIndicator";
import { LOCAL_STORAGE_EVENT } from "@/utils/localStorageUtils";

describe("useDisableUsageIndicator", () => {
  const STORAGE_KEY = "disableUsageIndicator";

  beforeEach(() => {
    localStorage.clear();
    vi.clearAllMocks();
  });

  afterEach(() => {
    localStorage.clear();
  });

  it("should return false when localStorage is empty", () => {
    const { result } = renderHook(() => useDisableUsageIndicator());

    expect(result.current).toBe(false);
  });

  it("should return false when localStorage value is not 'true'", () => {
    localStorage.setItem(STORAGE_KEY, "false");

    const { result } = renderHook(() => useDisableUsageIndicator());

    expect(result.current).toBe(false);
  });

  it("should return true when localStorage value is 'true'", () => {
    localStorage.setItem(STORAGE_KEY, "true");

    const { result } = renderHook(() => useDisableUsageIndicator());

    expect(result.current).toBe(true);
  });

  it("should return false when localStorage value is an empty string", () => {
    localStorage.setItem(STORAGE_KEY, "");

    const { result } = renderHook(() => useDisableUsageIndicator());

    expect(result.current).toBe(false);
  });

  it("should update when storage event fires for the correct key", async () => {
    const { result } = renderHook(() => useDisableUsageIndicator());

    expect(result.current).toBe(false);

    await act(async () => {
      localStorage.setItem(STORAGE_KEY, "true");
      const storageEvent = new StorageEvent("storage", {
        key: STORAGE_KEY,
        newValue: "true",
      });
      window.dispatchEvent(storageEvent);
    });

    await waitFor(() => {
      expect(result.current).toBe(true);
    });
  });

  it("should not update when storage event fires for a different key", () => {
    localStorage.setItem(STORAGE_KEY, "false");
    const { result } = renderHook(() => useDisableUsageIndicator());

    expect(result.current).toBe(false);

    const storageEvent = new StorageEvent("storage", {
      key: "otherKey",
      newValue: "true",
    });
    window.dispatchEvent(storageEvent);

    expect(result.current).toBe(false);
  });

  it("should update when custom LOCAL_STORAGE_EVENT fires for the correct key", async () => {
    const { result } = renderHook(() => useDisableUsageIndicator());

    expect(result.current).toBe(false);

    await act(async () => {
      localStorage.setItem(STORAGE_KEY, "true");
      const customEvent = new CustomEvent(LOCAL_STORAGE_EVENT, {
        detail: { key: STORAGE_KEY },
      });
      window.dispatchEvent(customEvent);
    });

    await waitFor(() => {
      expect(result.current).toBe(true);
    });
  });

  it("should not update when custom LOCAL_STORAGE_EVENT fires for a different key", () => {
    localStorage.setItem(STORAGE_KEY, "false");
    const { result } = renderHook(() => useDisableUsageIndicator());

    expect(result.current).toBe(false);

    const customEvent = new CustomEvent(LOCAL_STORAGE_EVENT, {
      detail: { key: "otherKey" },
    });
    window.dispatchEvent(customEvent);

    expect(result.current).toBe(false);
  });

  it("should update when localStorage changes from false to true via custom event", async () => {
    localStorage.setItem(STORAGE_KEY, "false");
    const { result } = renderHook(() => useDisableUsageIndicator());

    expect(result.current).toBe(false);

    await act(async () => {
      localStorage.setItem(STORAGE_KEY, "true");
      const customEvent = new CustomEvent(LOCAL_STORAGE_EVENT, {
        detail: { key: STORAGE_KEY },
      });
      window.dispatchEvent(customEvent);
    });

    await waitFor(() => {
      expect(result.current).toBe(true);
    });
  });

  it("should update when localStorage changes from true to false via storage event", async () => {
    localStorage.setItem(STORAGE_KEY, "true");
    const { result } = renderHook(() => useDisableUsageIndicator());

    expect(result.current).toBe(true);

    await act(async () => {
      localStorage.setItem(STORAGE_KEY, "false");
      const storageEvent = new StorageEvent("storage", {
        key: STORAGE_KEY,
        newValue: "false",
      });
      window.dispatchEvent(storageEvent);
    });

    await waitFor(() => {
      expect(result.current).toBe(false);
    });
  });

  it("should cleanup event listeners on unmount", () => {
    const addEventListenerSpy = vi.spyOn(window, "addEventListener");
    const removeEventListenerSpy = vi.spyOn(window, "removeEventListener");

    const { unmount } = renderHook(() => useDisableUsageIndicator());

    expect(addEventListenerSpy).toHaveBeenCalledTimes(2);
    expect(addEventListenerSpy).toHaveBeenCalledWith("storage", expect.any(Function));
    expect(addEventListenerSpy).toHaveBeenCalledWith(LOCAL_STORAGE_EVENT, expect.any(Function));

    unmount();

    expect(removeEventListenerSpy).toHaveBeenCalledTimes(2);
    expect(removeEventListenerSpy).toHaveBeenCalledWith("storage", expect.any(Function));
    expect(removeEventListenerSpy).toHaveBeenCalledWith(LOCAL_STORAGE_EVENT, expect.any(Function));
  });

  it("should handle multiple hooks independently", async () => {
    const { result: result1 } = renderHook(() => useDisableUsageIndicator());
    const { result: result2 } = renderHook(() => useDisableUsageIndicator());

    expect(result1.current).toBe(false);
    expect(result2.current).toBe(false);

    await act(async () => {
      localStorage.setItem(STORAGE_KEY, "true");
      const customEvent = new CustomEvent(LOCAL_STORAGE_EVENT, {
        detail: { key: STORAGE_KEY },
      });
      window.dispatchEvent(customEvent);
    });

    await waitFor(() => {
      expect(result1.current).toBe(true);
      expect(result2.current).toBe(true);
    });
  });
});
