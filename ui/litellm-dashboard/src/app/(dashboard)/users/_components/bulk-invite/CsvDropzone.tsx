import { ExclamationIcon } from "@heroicons/react/outline";
import { FileText, FileWarning, Trash2, TriangleAlert, Upload } from "lucide-react";
import React, { useId, useState } from "react";

import { Button, buttonVariants } from "@/components/ui/button";

interface CsvDropzoneProps {
  selectedFile: File | null;
  fileError: string | null;
  csvStructureError: string | null;
  onFile: (file: File) => void;
  onRemove: () => void;
}

function SelectedFileCard({
  file,
  fileError,
  csvStructureError,
  onRemove,
}: Omit<CsvDropzoneProps, "selectedFile" | "onFile"> & { file: File }) {
  const tone = fileError ? "text-destructive" : "text-info";
  return (
    <div
      className={`mb-4 p-4 rounded-md border ${fileError ? "bg-destructive/10 border-destructive/20" : "bg-info/10 border-info/20"}`}
    >
      <div className="flex items-center justify-between">
        <div className="flex items-center min-w-0">
          {fileError ? (
            <FileWarning className="size-5 shrink-0 text-destructive mr-3" />
          ) : (
            <FileText className="size-5 shrink-0 text-info mr-3" />
          )}
          <div className="min-w-0">
            <strong className={`break-words ${tone}`}>{file.name}</strong>
            <span className={`block text-xs ${tone}`}>
              {(file.size / 1024).toFixed(1)} KB • {new Date().toLocaleDateString()}
            </span>
          </div>
        </div>
        <Button variant="outline" size="sm" onClick={onRemove} className="flex items-center">
          <Trash2 className="size-4" />
          Remove
        </Button>
      </div>

      {fileError && (
        <div className="mt-3 text-destructive text-sm flex items-start">
          <TriangleAlert className="size-3.5 shrink-0 mr-2 mt-0.5" />
          <span className="min-w-0 break-words">{fileError}</span>
        </div>
      )}
      {!fileError && !csvStructureError && (
        <div className="mt-3 flex items-center">
          <div className="w-full bg-border rounded-full h-1.5">
            <div className="bg-info h-1.5 rounded-full w-full animate-pulse"></div>
          </div>
          <span className="ml-2 text-xs text-info">Processing...</span>
        </div>
      )}
    </div>
  );
}

function DropTarget({ onFile }: Pick<CsvDropzoneProps, "onFile">) {
  const inputId = useId();
  const [isDraggingOver, setIsDraggingOver] = useState(false);

  const handleInputChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) onFile(file);
  };

  const handleDragOver = (event: React.DragEvent<HTMLLabelElement>) => {
    event.preventDefault();
    setIsDraggingOver(true);
  };

  const handleDrop = (event: React.DragEvent<HTMLLabelElement>) => {
    event.preventDefault();
    setIsDraggingOver(false);
    const file = event.dataTransfer.files?.[0];
    if (file) onFile(file);
  };

  return (
    <label
      htmlFor={inputId}
      className="block"
      onDragOver={handleDragOver}
      onDragLeave={() => setIsDraggingOver(false)}
      onDrop={handleDrop}
    >
      <div
        className={`border-2 border-dashed ${isDraggingOver ? "border-info" : "border-border"} rounded-lg p-8 text-center hover:border-info focus-within:border-info transition-colors cursor-pointer`}
      >
        <input id={inputId} type="file" accept=".csv" className="sr-only" onChange={handleInputChange} />
        <Upload className="size-[30px] text-muted-foreground mb-2" />
        <p className="mb-1">Drag and drop your CSV file here</p>
        <p className="text-sm text-muted-foreground mb-3">or</p>
        <span className={buttonVariants({ variant: "outline", size: "sm" })}>Browse files</span>
        <p className="text-xs text-muted-foreground mt-4">Only CSV files (.csv) are supported</p>
      </div>
    </label>
  );
}

export function CsvDropzone({ selectedFile, fileError, csvStructureError, onFile, onRemove }: CsvDropzoneProps) {
  return (
    <>
      {selectedFile ? (
        <SelectedFileCard
          file={selectedFile}
          fileError={fileError}
          csvStructureError={csvStructureError}
          onRemove={onRemove}
        />
      ) : (
        <DropTarget onFile={onFile} />
      )}

      {csvStructureError && (
        <div className="mb-4 p-4 bg-warning/10 border border-warning/20 rounded-md">
          <div className="flex items-start">
            <ExclamationIcon className="h-5 w-5 shrink-0 text-warning mr-2 mt-0.5" />
            <div className="min-w-0">
              <strong className="text-warning">CSV Structure Error</strong>
              <p className="text-warning mt-1 mb-0 break-words">{csvStructureError}</p>
              <p className="text-warning mt-2 mb-0">
                Please download our template and ensure your CSV follows the required format.
              </p>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
