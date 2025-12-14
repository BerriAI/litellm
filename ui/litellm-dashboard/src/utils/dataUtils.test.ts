import { describe, it, expect, beforeEach, vi, afterEach } from "vitest";
import { copyToClipboard, formatNumberWithCommas, getSpendString, updateExistingKeys } from "./dataUtils";

// Mock NotificationsManager
vi.mock("../../src/components/molecules/notifications_manager", () => ({
  default: {
    success: vi.fn(),
    fromBackend: vi.fn(),
  },
}));

// Import the mocked module
import NotificationsManager from "../components/molecules/notifications_manager";
const mockNotificationsManager = vi.mocked(NotificationsManager);

describe("dataUtils", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    // Reset document.execCommand mock
    delete (document as any).execCommand;
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  describe("updateExistingKeys", () => {
    it("should update only existing keys in target object", () => {
      const target = { a: 1, b: 2, c: 3 };
      const source = { a: 10, b: 20, d: 40 };

      const result = updateExistingKeys(target, source);

      expect(result).toEqual({ a: 10, b: 20, c: 3 });
      expect(result).not.toBe(target); // Should be a clone
    });

    it("should not modify original target object", () => {
      const target = { a: 1, b: 2 };
      const source = { a: 10, c: 30 };

      updateExistingKeys(target, source);

      expect(target).toEqual({ a: 1, b: 2 }); // Original unchanged
    });

    it("should handle empty source object", () => {
      const target = { a: 1, b: 2 };
      const source = {};

      const result = updateExistingKeys(target, source);

      expect(result).toEqual({ a: 1, b: 2 });
    });
  });

  describe("formatNumberWithCommas", () => {
    it("should format numbers with commas", () => {
      expect(formatNumberWithCommas(1234567)).toBe("1,234,567");
      expect(formatNumberWithCommas(1000)).toBe("1,000");
      expect(formatNumberWithCommas(123)).toBe("123");
    });

    it("should handle decimals", () => {
      expect(formatNumberWithCommas(1234.5678, 2)).toBe("1,234.57");
      expect(formatNumberWithCommas(1000.123, 3)).toBe("1,000.123");
    });

    it("should handle null and undefined values", () => {
      expect(formatNumberWithCommas(null)).toBe("-");
      expect(formatNumberWithCommas(undefined)).toBe("-");
    });

    it("should handle zero and non-finite values", () => {
      expect(formatNumberWithCommas(0)).toBe("0");
      expect(formatNumberWithCommas(0, 2)).toBe("0.00");
      expect(formatNumberWithCommas(Infinity)).toBe("-");
      expect(formatNumberWithCommas(Number.NaN)).toBe("-");
    });

    it("should abbreviate large numbers", () => {
      expect(formatNumberWithCommas(1_234_567, 1, true)).toBe("1.2M");
      expect(formatNumberWithCommas(12_345, 2, true)).toBe("12.35K");
      expect(formatNumberWithCommas(-1_200, 2, true)).toBe("-1.20K");
    });

    it("should show zero when showZero is true", () => {
      expect(formatNumberWithCommas(0, 0, false, true)).toBe("0");
      expect(formatNumberWithCommas(0, 2, false, true)).toBe("0.00");
      expect(formatNumberWithCommas(0, 0, true, true)).toBe("0");
    });

    it("should return '-' for zero when showZero is false", () => {
      expect(formatNumberWithCommas(0, 0, false, false)).toBe("-");
      expect(formatNumberWithCommas(0, 2, false, false)).toBe("-");
      expect(formatNumberWithCommas(0, 0, true, false)).toBe("-");
    });
  });

  describe("getSpendString", () => {
    it("should return '-' for null, undefined, zero, or non-finite", () => {
      expect(getSpendString(null)).toBe("-");
      expect(getSpendString(undefined)).toBe("-");
      expect(getSpendString(0)).toBe("-");
      expect(getSpendString(Number.NaN)).toBe("-");
    });

    it("should format spend with dollar sign", () => {
      expect(getSpendString(1234.5, 2)).toBe("$1,234.50");
      expect(getSpendString(-2500, 0)).toBe("$-2,500");
    });

    it("should return threshold string for very small values", () => {
      expect(getSpendString(0.0000004, 6)).toBe("< $0.000001");
      expect(getSpendString(-0.0000004, 6)).toBe("< $0.000001");
    });

    it("should respect custom decimals", () => {
      expect(getSpendString(0.01234, 3)).toBe("$0.012");
      expect(getSpendString(999.9999, 1)).toBe("$1,000.0");
    });
  });

  describe("copyToClipboard", () => {
    describe("when Clipboard API is available (HTTPS scenario)", () => {
      beforeEach(() => {
        // Mock modern Clipboard API
        Object.assign(navigator, {
          clipboard: {
            writeText: vi.fn().mockResolvedValue(undefined),
          },
        });
      });

      it("should use navigator.clipboard.writeText when available", async () => {
        const result = await copyToClipboard("test text");

        expect(navigator.clipboard.writeText).toHaveBeenCalledWith("test text");
        expect(mockNotificationsManager.success).toHaveBeenCalledWith("Copied to clipboard");
        expect(result).toBe(true);
      });

      it("should use custom message when provided", async () => {
        await copyToClipboard("test text", "Custom message");

        expect(mockNotificationsManager.success).toHaveBeenCalledWith("Custom message");
      });

      it("should return false for null/undefined text", async () => {
        expect(await copyToClipboard(null)).toBe(false);
        expect(await copyToClipboard(undefined)).toBe(false);
        expect(navigator.clipboard.writeText).not.toHaveBeenCalled();
      });

      it("should fall back to execCommand when clipboard API fails", async () => {
        // Make clipboard API fail
        navigator.clipboard.writeText = vi.fn().mockRejectedValue(new Error("Permission denied"));

        // Mock successful execCommand
        document.execCommand = vi.fn().mockReturnValue(true);

        // Mock DOM methods
        const mockTextArea = {
          value: "",
          style: {} as Record<string, string>,
          setAttribute: vi.fn(),
          focus: vi.fn(),
          select: vi.fn(),
        };
        document.createElement = vi.fn().mockReturnValue(mockTextArea);
        document.body.appendChild = vi.fn();
        document.body.removeChild = vi.fn();

        const result = await copyToClipboard("test text");

        expect(navigator.clipboard.writeText).toHaveBeenCalledWith("test text");
        expect(document.execCommand).toHaveBeenCalledWith("copy");
        expect(mockNotificationsManager.success).toHaveBeenCalledWith("Copied to clipboard");
        expect(result).toBe(true);
      });
    });

    describe("when Clipboard API is not available (HTTP scenario)", () => {
      beforeEach(() => {
        // Mock HTTP scenario - no clipboard API
        Object.assign(navigator, {
          clipboard: undefined,
        });

        // Mock successful execCommand
        document.execCommand = vi.fn().mockReturnValue(true);

        // Mock DOM methods
        const mockTextArea = {
          value: "",
          style: {} as Record<string, string>,
          setAttribute: vi.fn(),
          focus: vi.fn(),
          select: vi.fn(),
        };
        document.createElement = vi.fn().mockReturnValue(mockTextArea);
        document.body.appendChild = vi.fn();
        document.body.removeChild = vi.fn();
      });

      it("should fall back to execCommand when clipboard API is not available", async () => {
        const result = await copyToClipboard("test text");

        expect(document.createElement).toHaveBeenCalledWith("textarea");
        expect(document.execCommand).toHaveBeenCalledWith("copy");
        expect(mockNotificationsManager.success).toHaveBeenCalledWith("Copied to clipboard");
        expect(result).toBe(true);
      });

      it("should set textarea properties correctly", async () => {
        const mockTextArea = {
          value: "",
          style: {} as Record<string, string>,
          setAttribute: vi.fn(),
          focus: vi.fn(),
          select: vi.fn(),
        };
        document.createElement = vi.fn().mockReturnValue(mockTextArea);

        await copyToClipboard("test text");

        expect(mockTextArea.value).toBe("test text");
        expect(mockTextArea.style.position).toBe("fixed");
        expect(mockTextArea.style.left).toBe("-999999px");
        expect(mockTextArea.style.top).toBe("-999999px");
        expect(mockTextArea.setAttribute).toHaveBeenCalledWith("readonly", "");
        expect(mockTextArea.focus).toHaveBeenCalled();
        expect(mockTextArea.select).toHaveBeenCalled();
      });

      it("should handle execCommand failure", async () => {
        document.execCommand = vi.fn().mockReturnValue(false);

        const result = await copyToClipboard("test text");

        expect(document.execCommand).toHaveBeenCalledWith("copy");
        expect(mockNotificationsManager.fromBackend).toHaveBeenCalledWith("Failed to copy to clipboard");
        expect(result).toBe(false);
      });

      it("should handle DOM manipulation errors", async () => {
        document.createElement = vi.fn().mockImplementation(() => {
          throw new Error("DOM error");
        });

        const result = await copyToClipboard("test text");

        expect(mockNotificationsManager.fromBackend).toHaveBeenCalledWith("Failed to copy to clipboard");
        expect(result).toBe(false);
      });

      it("should clean up textarea element after successful copy", async () => {
        const mockTextArea = {
          value: "",
          style: {} as Record<string, string>,
          setAttribute: vi.fn(),
          focus: vi.fn(),
          select: vi.fn(),
        };
        document.createElement = vi.fn().mockReturnValue(mockTextArea);

        await copyToClipboard("test text");

        expect(document.body.appendChild).toHaveBeenCalledWith(mockTextArea);
        expect(document.body.removeChild).toHaveBeenCalledWith(mockTextArea);
      });
    });

    describe("edge cases", () => {
      beforeEach(() => {
        // Mock scenario where navigator exists but clipboard is null
        Object.assign(navigator, {
          clipboard: null,
        });

        document.execCommand = vi.fn().mockReturnValue(true);
        const mockTextArea = {
          value: "",
          style: {} as Record<string, string>,
          setAttribute: vi.fn(),
          focus: vi.fn(),
          select: vi.fn(),
        };
        document.createElement = vi.fn().mockReturnValue(mockTextArea);
        document.body.appendChild = vi.fn();
        document.body.removeChild = vi.fn();
      });

      it("should handle navigator.clipboard being null", async () => {
        const result = await copyToClipboard("test text");

        expect(document.execCommand).toHaveBeenCalledWith("copy");
        expect(result).toBe(true);
      });

      it("should handle navigator.clipboard.writeText being undefined", async () => {
        Object.assign(navigator, {
          clipboard: {}, // clipboard exists but writeText doesn't
        });

        const result = await copyToClipboard("test text");

        expect(document.execCommand).toHaveBeenCalledWith("copy");
        expect(result).toBe(true);
      });
    });
  });
});
