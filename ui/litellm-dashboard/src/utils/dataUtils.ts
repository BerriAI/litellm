import NotificationsManager from "@/components/molecules/notifications_manager";
import { message } from "antd";

export function updateExistingKeys<Source extends Object>(target: Source, source: Object): Source {
  const clonedTarget = structuredClone(target);

  for (const [key, value] of Object.entries(source)) {
    if (key in clonedTarget) {
      (clonedTarget as any)[key] = value;
    }
  }

  return clonedTarget;
}

export const formatNumberWithCommas = (value: number | null | undefined, decimals: number = 0): string => {
  if (value === null || value === undefined) {
    return "-";
  }
  return value.toLocaleString("en-US", {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  });
};

export const copyToClipboard = async (
  text: string | null | undefined,
  messageText: string = "Copied to clipboard",
): Promise<boolean> => {
  if (!text) return false;

  // Check if clipboard API is available
  if (navigator && navigator.clipboard && navigator.clipboard.writeText) {
    try {
      await navigator.clipboard.writeText(text);
      NotificationsManager.success(messageText);
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
    const textArea = document.createElement("textarea");
    textArea.value = text;

    // Make the textarea invisible
    textArea.style.position = "fixed";
    textArea.style.left = "-999999px";
    textArea.style.top = "-999999px";
    textArea.setAttribute("readonly", "");

    document.body.appendChild(textArea);
    textArea.focus();
    textArea.select();

    const successful = document.execCommand("copy");
    document.body.removeChild(textArea);

    if (successful) {
      NotificationsManager.success(messageText);
      return true;
    } else {
      throw new Error("execCommand failed");
    }
  } catch (err) {
    NotificationsManager.fromBackend("Failed to copy to clipboard");
    console.error("Failed to copy: ", err);
    return false;
  }
};
