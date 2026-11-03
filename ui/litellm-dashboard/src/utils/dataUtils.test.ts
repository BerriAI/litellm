// @vitest-environment jsdom

import "@testing-library/jest-dom/vitest";
import { describe, it, expect, beforeEach, vi, afterEach } from "vitest";
import {
  copyToClipboard,
  formatNumberWithCommas,
  formatPerSecondCost,
  getSpendString,
  updateExistingKeys,
} from "./dataUtils";

// Import the mocked module
import { toast } from "@/lib/toast";
const mockToast = vi.mocked(toast);

describe("dataUtils", () => {
  beforeEach(() => {
    vi.clearAllMocks();
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

  describe("formatPerSecondCost", () => {
    it("should keep at least two decimals and append the per-second unit", () => {
      expect(formatPerSecondCost(0.4)).toBe("$0.40/s");
      expect(formatPerSecondCost(1)).toBe("$1.00/s");
    });

    it("should show sub-cent rates without rounding them to zero", () => {
      expect(formatPerSecondCost(0.015)).toBe("$0.015/s");
      expect(formatPerSecondCost(0.000025)).toBe("$0.000025/s");
    });
  });

  describe("copyToClipboard", () => {
    const clipboardDescriptor = Object.getOwnPropertyDescriptor(navigator, "clipboard");
    const execCommandDescriptor = Object.getOwnPropertyDescriptor(document, "execCommand");
    const writeText = vi.fn<(text: string) => Promise<void>>();
    const execCommand = vi.fn<(command: string) => boolean>();
    const selectedText = vi.fn(() => {
      const element = document.activeElement;
      return element instanceof HTMLTextAreaElement
        ? element.value.slice(element.selectionStart, element.selectionEnd)
        : undefined;
    });
    const trigger = document.createElement("button");
    const text = "Request α\nsecond line";

    beforeEach(() => {
      writeText.mockResolvedValue(undefined);
      execCommand.mockImplementation(() => {
        selectedText();
        return true;
      });
      Object.defineProperty(navigator, "clipboard", { configurable: true, value: { writeText } });
      Object.defineProperty(document, "execCommand", { configurable: true, value: execCommand });
      vi.spyOn(console, "error").mockImplementation(() => {});
      document.body.appendChild(trigger);
      trigger.focus();
    });

    afterEach(() => {
      document.body.replaceChildren();
      if (clipboardDescriptor) Object.defineProperty(navigator, "clipboard", clipboardDescriptor);
      else Reflect.deleteProperty(navigator, "clipboard");
      if (execCommandDescriptor) Object.defineProperty(document, "execCommand", execCommandDescriptor);
      else Reflect.deleteProperty(document, "execCommand");
    });

    it("confirms a modern clipboard write only after it succeeds", async () => {
      const pendingWrite = Promise.withResolvers<void>();
      writeText.mockReturnValue(pendingWrite.promise);
      const result = copyToClipboard(text, "Request copied");

      expect(writeText).toHaveBeenCalledWith(text);
      expect(mockToast.success).not.toHaveBeenCalled();
      pendingWrite.resolve();

      expect(await result).toBe(true);
      expect(mockToast.success).toHaveBeenCalledWith("Request copied");
      expect(execCommand).not.toHaveBeenCalled();
      expect(trigger).toHaveFocus();
    });

    it.each([null, undefined, ""])("does not copy absent content %s", async (value) => {
      expect(await copyToClipboard(value)).toBe(false);
      expect(writeText).not.toHaveBeenCalled();
      expect(execCommand).not.toHaveBeenCalled();
      expect(mockToast.success).not.toHaveBeenCalled();
    });

    it.each([undefined, null, {}])("copies the full selection when the Clipboard API is %s", async (clipboard) => {
      Object.defineProperty(navigator, "clipboard", { configurable: true, value: clipboard });

      expect(await copyToClipboard(text)).toBe(true);

      expect(execCommand).toHaveBeenCalledWith("copy");
      expect(selectedText).toHaveReturnedWith(text);
      expect(mockToast.success).toHaveBeenCalledWith("Copied to clipboard");
      expect(document.querySelector("textarea")).not.toBeInTheDocument();
      expect(trigger).toHaveFocus();
    });

    it("falls back after a rejected modern write", async () => {
      writeText.mockRejectedValue(new Error("Permission denied"));

      expect(await copyToClipboard(text, "Request copied")).toBe(true);

      expect(writeText).toHaveBeenCalledWith(text);
      expect(execCommand).toHaveBeenCalledWith("copy");
      expect(selectedText).toHaveReturnedWith(text);
      expect(mockToast.success).toHaveBeenCalledExactlyOnceWith("Request copied");
      expect(document.querySelector("textarea")).not.toBeInTheDocument();
      expect(trigger).toHaveFocus();
    });

    it.each(["false", "throw"])("cleans up and reports failure when the fallback returns %s", async (outcome) => {
      Object.defineProperty(navigator, "clipboard", { configurable: true, value: undefined });
      execCommand.mockImplementation(() => {
        selectedText();
        if (outcome === "throw") throw new Error("Copy denied");
        return false;
      });

      expect(await copyToClipboard(text)).toBe(false);

      expect(selectedText).toHaveReturnedWith(text);
      expect(mockToast.fromError).toHaveBeenCalledWith("Failed to copy to clipboard");
      expect(mockToast.success).not.toHaveBeenCalled();
      expect(document.querySelector("textarea")).not.toBeInTheDocument();
      expect(trigger).toHaveFocus();
    });

    it("reports failure when the fallback cannot create its selection", async () => {
      Object.defineProperty(navigator, "clipboard", { configurable: true, value: undefined });
      vi.spyOn(document, "createElement").mockImplementationOnce(() => {
        throw new Error("DOM error");
      });

      expect(await copyToClipboard(text)).toBe(false);
      expect(mockToast.fromError).toHaveBeenCalledWith("Failed to copy to clipboard");
      expect(mockToast.success).not.toHaveBeenCalled();
      expect(trigger).toHaveFocus();
    });
  });
});
