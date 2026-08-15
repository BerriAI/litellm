import React from "react";
import { FileText } from "lucide-react";
import { MessageType } from "@/components/chat_ui/types";
import { shouldShowAttachedImage } from "./ResponsesImageUtils";

interface ResponsesImageRendererProps {
  message: MessageType;
}

const ResponsesImageRenderer: React.FC<ResponsesImageRendererProps> = ({ message }) => {
  if (!shouldShowAttachedImage(message)) {
    return null;
  }

  const isPdf = typeof message.content === "string" && message.content.includes("[PDF attached]");

  return (
    <div className="mb-2">
      {isPdf ? (
        <div className="flex h-32 w-64 items-center justify-center rounded-md border border-gray-200 bg-red-50">
          <FileText className="size-12 text-red-600" aria-label="PDF attachment" />
        </div>
      ) : (
        <img
          src={message.imagePreviewUrl}
          alt="User uploaded image"
          className="max-h-[200px] max-w-64 rounded-md border border-gray-200 shadow-xs"
        />
      )}
    </div>
  );
};

export default ResponsesImageRenderer;
