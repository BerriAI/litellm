import { MessageType, MultimodalContent } from "./types";

/**
 * Ensures an image src URL uses a safe scheme (blob:, data:, http:, https:).
 * Returns an empty string for anything else (e.g. javascript: URIs) to
 * prevent XSS via img src injection.
 *
 * Uses URL parsing so the returned value (`parsed.href`) is reconstructed from
 * parsed components, breaking the taint chain for static-analysis tools like
 * CodeQL that track the raw user-provided string.
 */
export const sanitizeImageSrc = (url: string | undefined): string => {
  if (!url) return "";
  try {
    const parsed = new URL(url);
    const proto = parsed.protocol;
    if (
      proto === "blob:" ||
      proto === "http:" ||
      proto === "https:"
    ) {
      return parsed.href;
    }
    // Restrict data: URIs to image and PDF MIME types only.
    // Split on both ';' and ',' to handle both `data:type;base64,...`
    // and `data:type,...` (non-base64 inline) formats.
    if (proto === "data:") {
      const mime = parsed.pathname.split(/[;,]/)[0].toLowerCase();
      if (mime.startsWith("image/") || mime === "application/pdf") {
        return parsed.href;
      }
    }
  } catch {
    // invalid URL — fall through
  }
  return "";
};

export const convertImageToBase64 = (file: File): Promise<string> => {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const result = reader.result as string;
      // Extract just the base64 data (remove the data:image/...;base64, prefix)
      const base64Data = result.split(",")[1];
      resolve(base64Data);
    };
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
};

export const createMultimodalMessage = async (
  inputMessage: string,
  file: File,
): Promise<{ role: string; content: MultimodalContent[] }> => {
  const base64Data = await convertImageToBase64(file);
  const mimeType = file.type || (file.name.toLowerCase().endsWith(".pdf") ? "application/pdf" : "image/jpeg");

  return {
    role: "user",
    content: [
      { type: "input_text", text: inputMessage },
      {
        type: "input_image",
        image_url: `data:${mimeType};base64,${base64Data}`,
      },
    ],
  };
};

export const createDisplayMessage = (
  inputMessage: string,
  hasFile: boolean,
  filePreviewUrl?: string,
  fileName?: string,
): MessageType => {
  let attachmentText = "";
  if (hasFile && fileName) {
    attachmentText = fileName.toLowerCase().endsWith(".pdf") ? "[PDF attached]" : "[Image attached]";
  }

  const displayMessage: MessageType = {
    role: "user",
    content: hasFile ? `${inputMessage} ${attachmentText}` : inputMessage,
  };

  if (hasFile && filePreviewUrl) {
    displayMessage.imagePreviewUrl = filePreviewUrl;
  }

  return displayMessage;
};

export const shouldShowAttachedImage = (message: MessageType): boolean => {
  return (
    message.role === "user" &&
    typeof message.content === "string" &&
    (message.content.includes("[Image attached]") || message.content.includes("[PDF attached]")) &&
    !!message.imagePreviewUrl
  );
};
