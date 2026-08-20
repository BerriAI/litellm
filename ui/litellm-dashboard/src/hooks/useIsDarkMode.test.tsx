import { renderHook, waitFor } from "@testing-library/react";
import { afterAll, beforeEach, describe, expect, it, vi } from "vitest";
import { useIsDarkMode } from "./useIsDarkMode";

beforeEach(() => {
  document.documentElement.classList.remove("dark");
});

afterAll(() => {
  document.documentElement.classList.remove("dark");
});

describe("useIsDarkMode", () => {
  it("reports the dark class already on the root element at mount", () => {
    document.documentElement.classList.add("dark");

    const { result } = renderHook(() => useIsDarkMode());

    expect(result.current).toBe(true);
  });

  it("follows the root element's dark class as it is toggled", async () => {
    const { result } = renderHook(() => useIsDarkMode());
    expect(result.current).toBe(false);

    document.documentElement.classList.add("dark");
    await waitFor(() => expect(result.current).toBe(true));

    document.documentElement.classList.remove("dark");
    await waitFor(() => expect(result.current).toBe(false));
  });

  it("stops observing the root element once unmounted", () => {
    const disconnect = vi.spyOn(MutationObserver.prototype, "disconnect");

    const { unmount } = renderHook(() => useIsDarkMode());
    unmount();

    expect(disconnect).toHaveBeenCalled();
    disconnect.mockRestore();
  });
});
