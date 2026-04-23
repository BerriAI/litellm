import React from "react";
import { FileText } from "lucide-react";
import { MessageType } from "./types";
import { shouldShowAttachedImage } from "./ResponsesImageUtils";

interface ResponsesImageRendererProps {
  message: MessageType;
}

const ResponsesImageRenderer: React.FC<ResponsesImageRendererProps> = ({
  message,
}) => {
  if (!shouldShowAttachedImage(message)) {
    return null;
  }

  const isPdf =
    typeof message.content === "string" &&
    message.content.includes("[PDF attached]");

  return (
    <div className="mb-2">
      {isPdf ? (
        <div className="w-64 h-32 rounded-md border border-border bg-destructive/10 flex items-center justify-center">
          <FileText className="h-12 w-12 text-destructive" />
        </div>
      ) : (
        // eslint-disable-next-line @next/next/no-img-element
        <img
          src={message.imagePreviewUrl}
          alt="User uploaded image"
          className="max-w-64 rounded-md border border-border shadow-sm"
          style={{ maxHeight: "200px" }}
        />
      )}
    </div>
  );
};

export default ResponsesImageRenderer;
