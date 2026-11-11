import i18n from "@/i18n/i18n";
import { toast } from "@/lib/toast";

export function updateExistingKeys<Source extends object>(target: Source, source: object): Source {
  const clonedTarget = structuredClone(target);

  for (const [key, value] of Object.entries(source)) {
    if (key in clonedTarget) {
      (clonedTarget as any)[key] = value;
    }
  }

  return clonedTarget;
}

export const formatNumberWithCommas = (
  value: number | null | undefined,
  decimals: number = 0,
  abbreviate: boolean = false,
  showZero: boolean = true,
): string => {
  if (value === null || value === undefined || !Number.isFinite(value) || (value === 0 && !showZero)) {
    return "-";
  }

  const opts: Intl.NumberFormatOptions = {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  };

  if (!abbreviate) {
    return value.toLocaleString("en-US", opts);
  }

  const sign = value < 0 ? "-" : "";
  const abs = Math.abs(value);
  let scaled = abs;
  let suffix = "";

  if (abs >= 1_000_000) {
    scaled = abs / 1_000_000;
    suffix = "M";
  } else if (abs >= 1_000) {
    scaled = abs / 1_000;
    suffix = "K";
  }

  return `${sign}${scaled.toLocaleString("en-US", opts)}${suffix}`;
};

export const getSpendString = (value: number | null | undefined, decimals: number = 6): string => {
  if (value === null || value === undefined || !Number.isFinite(value) || value === 0) {
    return "-";
  }

  const halfSmallestUnit = Number(`5e-${decimals + 1}`);
  if (Math.abs(value) < halfSmallestUnit) {
    const threshold = (1 / 10 ** decimals).toFixed(decimals);
    return `< $${threshold}`;
  }

  return `$${formatNumberWithCommas(value, decimals, false, false)}`;
};

export const formatPerSecondCost = (cost: number): string =>
  `$${cost.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 6 })}/s`;

export const copyToClipboard = async (
  text: string | null | undefined,
  messageText: string = i18n.t("utils.dataUtils.copiedToClipboard"),
): Promise<boolean> => {
  if (!text) return false;

  // Check if clipboard API is available
  if (navigator && navigator.clipboard && navigator.clipboard.writeText) {
    try {
      await navigator.clipboard.writeText(text);
      toast.success(messageText);
      return true;
    } catch (err) {
      console.error("Clipboard API failed: ", err);
      // Fall back to legacy method
      return fallbackCopyToClipboard(text, messageText);
    }
  } else {
    // Use fallback method when clipboard API is not available
    return fallbackCopyToClipboard(text, messageText);
  }
};

// Fallback method using document.execCommand (deprecated but widely supported)
const fallbackCopyToClipboard = (text: string, messageText: string): boolean => {
  try {
    const previouslyFocused = document.activeElement;
    const textArea = document.createElement("textarea");
    textArea.value = text;

    // Make the textarea invisible
    textArea.style.position = "fixed";
    textArea.style.left = "-999999px";
    textArea.style.top = "-999999px";
    textArea.setAttribute("readonly", "");

    document.body.appendChild(textArea);
    try {
      textArea.focus();
      textArea.select();

      if (document.execCommand("copy")) {
        toast.success(messageText);
        return true;
      }
    } finally {
      textArea.remove();
      if (previouslyFocused instanceof HTMLElement) previouslyFocused.focus();
    }
    toast.fromError(i18n.t("utils.dataUtils.copyFailed"));
    return false;
  } catch (err) {
    toast.fromError(i18n.t("utils.dataUtils.copyFailed"));
    console.error("Failed to copy: ", err);
    return false;
  }
};
