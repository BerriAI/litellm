import { MessageType } from "./types";

export interface ChatMultimodalContent {
  type: "text" | "image_url";
  text?: string;
  image_url?: {
    url: string;
    detail?: string;
  };
}

export const convertImageToBase64 = (file: File): Promise<string> => {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const result = reader.result as string;
      // For chat completions, we keep the full data URI format
      resolve(result);
    };
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
};

export const createChatMultimodalMessage = async (
  inputMessage: string,
  file: File,
): Promise<{ role: string; content: ChatMultimodalContent[] }> => {
  const base64DataUri = await convertImageToBase64(file);

  return {
    role: "user",
    content: [
      { type: "text", text: inputMessage },
      {
        type: "image_url",
        image_url: {
          url: base64DataUri,
        },
      },
    ],
  };
};

export const createChatDisplayMessage = (
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

export const shouldShowChatAttachedImage = (message: MessageType): boolean => {
  return (
    message.role === "user" &&
    typeof message.content === "string" &&
    (message.content.includes("[Image attached]") || message.content.includes("[PDF attached]")) &&
    !!message.imagePreviewUrl
  );
};
