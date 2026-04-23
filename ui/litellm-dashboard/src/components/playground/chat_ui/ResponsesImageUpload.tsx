import React, { useRef } from "react";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { Paperclip } from "lucide-react";

interface ResponsesImageUploadProps {
  responsesUploadedImage: File | null;
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  responsesImagePreviewUrl: string | null;
  onImageUpload: (file: File) => false;
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  onRemoveImage: () => void;
}

const ResponsesImageUpload: React.FC<ResponsesImageUploadProps> = ({
  responsesUploadedImage,
  onImageUpload,
}) => {
  const inputRef = useRef<HTMLInputElement>(null);

  if (responsesUploadedImage) return null;

  return (
    <>
      <input
        ref={inputRef}
        type="file"
        accept="image/*,.pdf"
        className="hidden"
        onChange={(e) => {
          const file = e.target.files?.[0];
          if (file) onImageUpload(file);
          e.target.value = "";
        }}
      />
      <TooltipProvider>
        <Tooltip>
          <TooltipTrigger asChild>
            <button
              type="button"
              onClick={() => inputRef.current?.click()}
              className="flex items-center justify-center w-8 h-8 text-muted-foreground hover:text-foreground hover:bg-muted rounded-md transition-colors"
            >
              <Paperclip className="h-4 w-4" />
            </button>
          </TooltipTrigger>
          <TooltipContent>Attach image or PDF</TooltipContent>
        </Tooltip>
      </TooltipProvider>
    </>
  );
};

export default ResponsesImageUpload;
