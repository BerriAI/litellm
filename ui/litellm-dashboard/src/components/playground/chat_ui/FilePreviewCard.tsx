import { FileText, Trash2 } from "lucide-react";

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
              <FileText className="h-4 w-4 text-destructive-foreground" />
            </div>
          ) : (
            // eslint-disable-next-line @next/next/no-img-element
            <img
              src={previewUrl || ""}
              alt="Upload preview"
              className="w-10 h-10 rounded-md border border-border object-cover"
            />
          )}
        </div>
        <div className="flex-1 min-w-0">
          <div className="text-sm font-medium text-foreground truncate">
            {file.name}
          </div>
          <div className="text-xs text-muted-foreground">
            {isPdf ? "PDF" : "Image"}
          </div>
        </div>
        <button
          type="button"
          className="flex items-center justify-center w-6 h-6 text-muted-foreground hover:text-foreground hover:bg-accent rounded-full transition-colors"
          onClick={onRemove}
          aria-label="Remove attachment"
        >
          <Trash2 className="h-3 w-3" />
        </button>
      </div>
    </div>
  );
}

export default FilePreviewCard;
