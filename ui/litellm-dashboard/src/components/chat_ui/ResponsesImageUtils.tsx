import { MessageType, MultimodalContent } from "./types";

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
