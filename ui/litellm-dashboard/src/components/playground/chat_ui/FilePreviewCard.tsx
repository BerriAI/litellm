import { DeleteOutlined, FilePdfOutlined } from "@ant-design/icons";

interface FilePreviewCardProps {
  file: File;
  previewUrl: string | null;
  onRemove: () => void;
}

function FilePreviewCard({ file, previewUrl, onRemove }: FilePreviewCardProps) {
  const isPdf = file.name.toLowerCase().endsWith(".pdf");

  return (
    <div className="mb-2">
      <div className="flex items-center gap-3 p-3 bg-gray-50 rounded-lg border border-gray-200">
        <div className="relative inline-block">
          {isPdf ? (
            <div className="w-10 h-10 rounded-md bg-red-500 flex items-center justify-center">
              <FilePdfOutlined style={{ fontSize: "16px", color: "white" }} />
            </div>
          ) : (
            <img
              src={previewUrl || ""}
              alt="Upload preview"
              className="w-10 h-10 rounded-md border border-gray-200 object-cover"
            />
          )}
        </div>
        <div className="flex-1 min-w-0">
          <div className="text-sm font-medium text-gray-900 truncate">{file.name}</div>
          <div className="text-xs text-gray-500">
            {isPdf ? "PDF" : "Image"}
          </div>
        </div>
        <button
          className="flex items-center justify-center w-6 h-6 text-gray-400 hover:text-gray-600 hover:bg-gray-200 rounded-full transition-colors"
          onClick={onRemove}
        >
          <DeleteOutlined style={{ fontSize: "12px" }} />
        </button>
      </div>
    </div>
  );
}

export default FilePreviewCard;
