import React from "react";
import { MessageType } from "./types";
import { shouldShowAttachedImage } from "./ResponsesImageUtils";
import { FilePdfOutlined } from "@ant-design/icons";
import { useTranslation } from "react-i18next";

interface ResponsesImageRendererProps {
  message: MessageType;
}

const ResponsesImageRenderer: React.FC<ResponsesImageRendererProps> = ({ message }) => {
  const { t } = useTranslation();
  if (!shouldShowAttachedImage(message)) {
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
        <img
          src={message.imagePreviewUrl}
          alt={t("playground.chatImageRenderer.uploadedImageAlt")}
          className="max-w-64 rounded-md border border-gray-200 shadow-sm"
          style={{ maxHeight: "200px" }}
        />
      )}
    </div>
  );
};

export default ResponsesImageRenderer;
