import { MessageType } from "./types";

export interface ChatMultimodalContent {
  type: "text" | "image_url" | "file";
  text?: string;
  image_url?: {
    url: string;
    detail?: string;
  };
  file?: {
    file_data: string;
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
  const fileName = file.name.toLowerCase();
  
  // Check if file is PDF or text
  const isPdfOrText = fileName.endsWith('.pdf') || fileName.endsWith('.txt');
  
  // Ensure the data URI has the correct format with data: prefix
  // FileReader.readAsDataURL() should already provide this, but validate
  const fileData = base64DataUri.startsWith('data:') 
    ? base64DataUri 
    : `data:${file.type};base64,${base64DataUri}`;
  
  const fileContent: ChatMultimodalContent = isPdfOrText
    ? {
        type: "file",
        file: {
          file_data: fileData,
        },
      }
    : {
        type: "image_url",
        image_url: {
          url: base64DataUri,
        },
      };

  return {
    role: "user",
    content: [
      { type: "text", text: inputMessage },
      fileContent,
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
    const lowerFileName = fileName.toLowerCase();
    if (lowerFileName.endsWith(".pdf")) {
      attachmentText = "[PDF attached]";
    } else if (lowerFileName.endsWith(".txt")) {
      attachmentText = "[Text file attached]";
    } else {
      attachmentText = "[Image attached]";
    }
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
    (message.content.includes("[Image attached]") || 
     message.content.includes("[PDF attached]") || 
     message.content.includes("[Text file attached]")) &&
    !!message.imagePreviewUrl
  );
};
