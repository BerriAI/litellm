import React, { useId, useRef } from "react";
import { Paperclip } from "lucide-react";
import { toast } from "@/lib/toast";
import { Button } from "@/components/ui/button";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { CHAT_ATTACHMENT_ACCEPT, validateChatAttachment } from "./uploadValidation";

interface ChatImageUploadProps {
  chatUploadedImage: File | null;
  chatImagePreviewUrl: string | null;
  onImageUpload: (file: File) => void;
  onRemoveImage: () => void;
  disabled?: boolean;
}

const ChatImageUpload: React.FC<ChatImageUploadProps> = ({ chatUploadedImage, onImageUpload, disabled = false }) => {
  const inputRef = useRef<HTMLInputElement>(null);
  const inputId = useId();

  if (chatUploadedImage) {
    return null;
  }

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file) {
      return;
    }
    const result = validateChatAttachment(file);
    if (!result.ok) {
      toast.error(result.error);
      return;
    }
    onImageUpload(file);
  };

  return (
    <>
      <input
        id={inputId}
        ref={inputRef}
        type="file"
        accept={CHAT_ATTACHMENT_ACCEPT}
        className="sr-only"
        tabIndex={-1}
        disabled={disabled}
        onChange={handleFileChange}
      />
      <Tooltip>
        <TooltipTrigger
          render={
            <Button
              type="button"
              variant="ghost"
              size="icon-sm"
              disabled={disabled}
              aria-label="Attach image or PDF"
              className="text-muted-foreground hover:text-foreground"
              onClick={() => inputRef.current?.click()}
            />
          }
        >
          <Paperclip className="size-4" />
        </TooltipTrigger>
        <TooltipContent>Attach image or PDF</TooltipContent>
      </Tooltip>
    </>
  );
};

export default ChatImageUpload;
