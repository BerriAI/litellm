import { FileText, X } from "lucide-react";
import { Button } from "@/components/ui/button";

interface FilePreviewCardProps {
  file: File;
  previewUrl: string | null;
  onRemove: () => void;
}

function FilePreviewCard({ file, previewUrl, onRemove }: FilePreviewCardProps) {
  const isPdf = file.name.toLowerCase().endsWith(".pdf");

  return (
    <div className="mb-2">
      <div className="flex items-center gap-3 p-3 bg-muted rounded-lg border border-border">
        <div className="relative inline-block">
          {isPdf ? (
            <div className="w-10 h-10 rounded-md bg-destructive flex items-center justify-center">
              <FileText className="size-4 text-destructive-foreground" aria-hidden="true" />
            </div>
          ) : (
            <img
              src={previewUrl || ""}
              alt="Upload preview"
              className="w-10 h-10 rounded-md border border-border object-cover"
            />
          )}
        </div>
        <div className="flex-1 min-w-0">
          <div className="text-sm font-medium text-foreground truncate">{file.name}</div>
          <div className="text-xs text-muted-foreground">{isPdf ? "PDF" : "Image"}</div>
        </div>
        <Button
          type="button"
          variant="ghost"
          size="icon-xs"
          aria-label={`Remove ${file.name}`}
          className="text-muted-foreground hover:text-foreground hover:bg-accent"
          onClick={onRemove}
        >
          <X className="size-3" />
        </Button>
      </div>
    </div>
  );
}

export default FilePreviewCard;
