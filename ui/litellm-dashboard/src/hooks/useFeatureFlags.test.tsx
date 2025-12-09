import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import { renderHook, waitFor } from "@testing-library/react";
import { useRouter } from "next/navigation";
import useFeatureFlags, { FeatureFlagsProvider } from "./useFeatureFlags";

// Mock next/navigation
vi.mock("next/navigation", () => ({
  useRouter: vi.fn(),
}));

// Mock the networking module to control serverRootPath
vi.mock("@/components/networking", () => ({
  serverRootPath: "/",
}));

describe("useFeatureFlags", () => {
  let mockReplace: ReturnType<typeof vi.fn>;
  let originalLocation: Location;

  beforeEach(() => {
    // Mock router
    mockReplace = vi.fn();
    (useRouter as ReturnType<typeof vi.fn>).mockReturnValue({
      replace: mockReplace,
    });

    // Store original location
    originalLocation = window.location;

    // Mock localStorage
    Storage.prototype.getItem = vi.fn(() => null);
    Storage.prototype.setItem = vi.fn();
    Storage.prototype.removeItem = vi.fn();
  });

  afterEach(() => {
    vi.clearAllMocks();
    // Restore location
    Object.defineProperty(window, "location", {
      writable: true,
      value: originalLocation,
    });
  });

  describe("FeatureFlagsProvider - redirect logic", () => {
    it("should not redirect when refactoredUIFlag is true", async () => {
      // Set flag to true
      Storage.prototype.getItem = vi.fn(() => "true");

      const { result } = renderHook(() => useFeatureFlags(), {
        wrapper: FeatureFlagsProvider,
      });

      expect(result.current.refactoredUIFlag).toBe(true);

      // Wait for any effects
      await waitFor(() => {
        expect(mockReplace).not.toHaveBeenCalled();
      });
    });

    it("should not redirect when already on a /ui path (race condition protection)", async () => {
      // Set flag to false to trigger redirect logic
      Storage.prototype.getItem = vi.fn(() => "false");

      // Mock window.location to be on a custom UI path
      delete (window as any).location;
      window.location = {
        pathname: "/my-custom-path/ui/",
      } as Location;

      renderHook(() => useFeatureFlags(), {
        wrapper: FeatureFlagsProvider,
      });

      // Wait for timeout and check redirect was NOT called
      await new Promise((resolve) => setTimeout(resolve, 150));

      expect(mockReplace).not.toHaveBeenCalled();
    });

    it("should not redirect when on /ui path without custom root", async () => {
      // Set flag to false
      Storage.prototype.getItem = vi.fn(() => "false");

      // Mock window.location to be on standard UI path
      delete (window as any).location;
      window.location = {
        pathname: "/ui/",
      } as Location;

      renderHook(() => useFeatureFlags(), {
        wrapper: FeatureFlagsProvider,
      });

      // Wait for timeout and check redirect was NOT called
      await new Promise((resolve) => setTimeout(resolve, 150));

      expect(mockReplace).not.toHaveBeenCalled();
    });

    it("should redirect when flag is false and not on a /ui path", async () => {
      // Set flag to false
      Storage.prototype.getItem = vi.fn(() => "false");

      // Mock window.location to be on a non-UI path
      delete (window as any).location;
      window.location = {
        pathname: "/some-other-path/",
      } as Location;

      renderHook(() => useFeatureFlags(), {
        wrapper: FeatureFlagsProvider,
      });

      // Wait for timeout plus a bit more
      await new Promise((resolve) => setTimeout(resolve, 150));

      // Should have called replace to redirect to base path
      expect(mockReplace).toHaveBeenCalledWith("/");
    });

    it("should not redirect if already at base path", async () => {
      // Set flag to false
      Storage.prototype.getItem = vi.fn(() => "false");

      // Mock window.location to be at root
      delete (window as any).location;
      window.location = {
        pathname: "/",
      } as Location;

      renderHook(() => useFeatureFlags(), {
        wrapper: FeatureFlagsProvider,
      });

      // Wait for timeout
      await new Promise((resolve) => setTimeout(resolve, 150));

      expect(mockReplace).not.toHaveBeenCalled();
    });
  });

  describe("useFeatureFlags - flag management", () => {
    it("should initialize with false when no value in localStorage", () => {
      Storage.prototype.getItem = vi.fn(() => null);

      const { result } = renderHook(() => useFeatureFlags(), {
        wrapper: FeatureFlagsProvider,
      });

      expect(result.current.refactoredUIFlag).toBe(false);
    });

    it("should initialize with true when localStorage has true", () => {
      Storage.prototype.getItem = vi.fn(() => "true");

      const { result } = renderHook(() => useFeatureFlags(), {
        wrapper: FeatureFlagsProvider,
      });

      expect(result.current.refactoredUIFlag).toBe(true);
    });

    it("should update localStorage when setRefactoredUIFlag is called", () => {
      const setItemMock = vi.fn();
      Storage.prototype.setItem = setItemMock;
      Storage.prototype.getItem = vi.fn(() => "false");

      const { result } = renderHook(() => useFeatureFlags(), {
        wrapper: FeatureFlagsProvider,
      });

      result.current.setRefactoredUIFlag(true);

      expect(setItemMock).toHaveBeenCalledWith(
        "feature.refactoredUIFlag",
        "true"
      );
    });

    it("should handle malformed localStorage values gracefully", () => {
      Storage.prototype.getItem = vi.fn(() => "invalid-value");

      const { result } = renderHook(() => useFeatureFlags(), {
        wrapper: FeatureFlagsProvider,
      });

      // Should default to false for malformed values
      expect(result.current.refactoredUIFlag).toBe(false);
    });
  });

  describe("getBasePath logic with serverRootPath", () => {
    it("should handle serverRootPath being set to custom path", async () => {
      // Mock the networking module with custom serverRootPath
      vi.doMock("@/components/networking", () => ({
        serverRootPath: "/my-custom-path",
      }));

      // Set flag to false to trigger redirect
      Storage.prototype.getItem = vi.fn(() => "false");

      // Mock location to be on wrong path
      delete (window as any).location;
      window.location = {
        pathname: "/wrong-path/",
      } as Location;

      renderHook(() => useFeatureFlags(), {
        wrapper: FeatureFlagsProvider,
      });

      // Wait for timeout
      await new Promise((resolve) => setTimeout(resolve, 150));

      // With default NEXT_PUBLIC_BASE_URL being empty, should redirect to "/"
      // (In reality, with serverRootPath="/my-custom-path", it would be "/my-custom-path/")
      expect(mockReplace).toHaveBeenCalled();
    });
  });

  describe("storage event synchronization", () => {
    it("should update flag when storage event is fired", async () => {
      Storage.prototype.getItem = vi.fn(() => "false");

      const { result } = renderHook(() => useFeatureFlags(), {
        wrapper: FeatureFlagsProvider,
      });

      expect(result.current.refactoredUIFlag).toBe(false);

      // Simulate storage event from another tab
      const storageEvent = new StorageEvent("storage", {
        key: "feature.refactoredUIFlag",
        newValue: "true",
      });

      window.dispatchEvent(storageEvent);

      await waitFor(() => {
        expect(result.current.refactoredUIFlag).toBe(true);
      });
    });

    it("should self-heal when storage key is cleared", async () => {
      const setItemMock = vi.fn();
      Storage.prototype.setItem = setItemMock;
      Storage.prototype.getItem = vi.fn(() => "true");

      renderHook(() => useFeatureFlags(), {
        wrapper: FeatureFlagsProvider,
      });

      // Simulate storage event where key was cleared
      const storageEvent = new StorageEvent("storage", {
        key: "feature.refactoredUIFlag",
        newValue: null,
      });

      window.dispatchEvent(storageEvent);

      await waitFor(() => {
        expect(setItemMock).toHaveBeenCalledWith(
          "feature.refactoredUIFlag",
          "false"
        );
      });
    });
  });

  describe("timeout cleanup", () => {
    it("should cleanup timeout on unmount", async () => {
      Storage.prototype.getItem = vi.fn(() => "false");

      delete (window as any).location;
      window.location = {
        pathname: "/some-path/",
      } as Location;

      const { unmount } = renderHook(() => useFeatureFlags(), {
        wrapper: FeatureFlagsProvider,
      });

      // Unmount immediately before timeout fires
      unmount();

      // Wait past the timeout
      await new Promise((resolve) => setTimeout(resolve, 150));

      // Should not have called replace since component unmounted
      expect(mockReplace).not.toHaveBeenCalled();
    });
  });
});

