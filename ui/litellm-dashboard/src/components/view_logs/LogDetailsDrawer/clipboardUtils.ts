import { message } from "antd";
import { MESSAGE_COPY_SUCCESS } from "./constants";

/**
 * Copies text to clipboard with fallback for non-secure contexts.
 * Shows success/error message to user.
 *
 * @param text - Text to copy to clipboard
 * @param label - Label for the copied content (e.g., "Request", "Metadata")
 * @returns Promise<boolean> - true if copy succeeded, false otherwise
 */
export async function copyToClipboard(text: string, label: string): Promise<boolean> {
  try {
    // Try modern clipboard API first
    if (navigator.clipboard && window.isSecureContext) {
      await navigator.clipboard.writeText(text);
      message.success(`${label} ${MESSAGE_COPY_SUCCESS}`);
      return true;
    } else {
      // Fallback for non-secure contexts (like 0.0.0.0)
      const textArea = document.createElement("textarea");
      textArea.value = text;
      textArea.style.position = "fixed";
      textArea.style.opacity = "0";
      document.body.appendChild(textArea);
      textArea.focus();
      textArea.select();

      const successful = document.execCommand("copy");
      document.body.removeChild(textArea);

      if (!successful) {
        throw new Error("execCommand failed");
      }
      message.success(`${label} ${MESSAGE_COPY_SUCCESS}`);
      return true;
    }
  } catch (error) {
    console.error("Copy failed:", error);
    message.error(`Failed to copy ${label}`);
    return false;
  }
}
