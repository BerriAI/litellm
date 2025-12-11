import React from "react";
import Image from "next/image";
import { MessageType } from "./types";
import { shouldShowChatAttachedImage } from "./ChatImageUtils";
import { FilePdfOutlined } from "@ant-design/icons";

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
        <div className="w-64 h-32 rounded-md border border-gray-200 bg-red-50 flex items-center justify-center">
          <FilePdfOutlined style={{ fontSize: "48px", color: "#dc2626" }} />
        </div>
      ) : (
        <Image
          src={message.imagePreviewUrl || ""}
          alt="User uploaded image"
          width={256}
          height={200}
          className="max-w-64 rounded-md border border-gray-200 shadow-sm"
          style={{ maxHeight: "200px", width: "auto", height: "auto" }}
        />
      )}
    </div>
  );
};

export default ChatImageRenderer;
