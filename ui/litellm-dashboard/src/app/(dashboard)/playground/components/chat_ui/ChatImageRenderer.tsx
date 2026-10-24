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
        <div className="flex h-32 w-64 items-center justify-center rounded-md border border-border bg-destructive/10">
          <FileText className="size-12 text-destructive" aria-label="PDF attachment" />
        </div>
      ) : (
        <Image
          src={message.imagePreviewUrl || ""}
          alt="User uploaded image"
          width={256}
          height={200}
          className="max-w-64 rounded-md border border-border shadow-xs"
          style={{ maxHeight: "200px", width: "auto", height: "auto" }}
        />
      )}
    </div>
  );
};

export default ChatImageRenderer;
