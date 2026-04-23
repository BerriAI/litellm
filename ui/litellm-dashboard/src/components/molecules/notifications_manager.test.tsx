import { beforeEach, describe, expect, it, vi } from "vitest";

// The global `setupTests.ts` mocks `@/components/molecules/notifications_manager`
// as a safety rail for component tests. Unmock it here so we can exercise the
// real manager against sonner.
vi.unmock("@/components/molecules/notifications_manager");
vi.unmock("./notifications_manager");

// Sonner is a module-level singleton; mock it before importing the manager.
const mockToast = vi.hoisted(() => ({
  success: vi.fn(),
  error: vi.fn(),
  warning: vi.fn(),
  info: vi.fn(),
  loading: vi.fn(),
  dismiss: vi.fn(),
}));

vi.mock("sonner", () => ({
  toast: mockToast,
}));

// Dynamic import AFTER vi.unmock so the global mock doesn't take effect.
const { default: NotificationManager } = await import("./notifications_manager");

describe("NotificationManager (sonner-backed)", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe("Already Exists case", () => {
    it("should show error toast for 'already exists' backend message", () => {
      const error = {
        message: "Key with alias 'test10' already exists.",
        type: "bad_request_error",
        code: "400",
      };

      NotificationManager.fromBackend(error);

      expect(mockToast.error).toHaveBeenCalledWith(
        "Already Exists",
        expect.objectContaining({
          description: "Key with alias 'test10' already exists.",
          duration: 6000,
        }),
      );
    });
  });

  describe("direct call routing", () => {
    const cases = [
      { label: "error", call: () => NotificationManager.error("Test error"), mockFn: mockToast.error },
      { label: "warning", call: () => NotificationManager.warning("Test warning"), mockFn: mockToast.warning },
      { label: "info", call: () => NotificationManager.info("Test info"), mockFn: mockToast.info },
      { label: "success", call: () => NotificationManager.success("Test success"), mockFn: mockToast.success },
    ];

    cases.forEach(({ label, call, mockFn }) => {
      it(`should route ${label} to toast.${label}`, () => {
        call();
        expect(mockFn).toHaveBeenCalled();
      });
    });
  });

  describe("clear()", () => {
    it("dismisses all toasts", () => {
      NotificationManager.clear();
      expect(mockToast.dismiss).toHaveBeenCalled();
    });
  });
});
