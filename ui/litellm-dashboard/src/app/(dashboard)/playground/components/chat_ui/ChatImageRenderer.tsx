import React from "react";
import Image from "next/image";
import { FileText } from "lucide-react";
import { MessageType } from "@/components/chat_ui/types";
import { shouldShowChatAttachedImage } from "./ChatImageUtils";

interface ChatImageRendererProps {
  message: MessageType;
}

const ChatImageRenderer: React.FC<ChatImageRendererProps> = ({ message }) => {
  if (!shouldShowChatAttachedImage(message)) {
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
        <Image
          src={message.imagePreviewUrl || ""}
          alt="User uploaded image"
          width={256}
          height={200}
          className="max-w-64 rounded-md border border-gray-200 shadow-xs"
          style={{ maxHeight: "200px", width: "auto", height: "auto" }}
        />
      )}
    </div>
  );
};

export default ChatImageRenderer;
