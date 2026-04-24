import React from "react";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";
import MessageManager from "@/components/molecules/message_manager";
import { Copy, Eye, Trash2 } from "lucide-react";
import { DocumentUpload } from "./types";

interface DocumentsTableProps {
  documents: DocumentUpload[];
  onRemove: (uid: string) => void;
}

const DocumentsTable: React.FC<DocumentsTableProps> = ({
  documents,
  onRemove,
}) => {
  const handleCopyId = (uid: string) => {
    navigator.clipboard.writeText(uid);
    MessageManager.success("Document ID copied to clipboard");
  };

  const getStatusBadge = (status: DocumentUpload["status"]) => {
    const statusConfig: Record<
      DocumentUpload["status"],
      { className: string; text: string }
    > = {
      uploading: {
        className:
          "bg-blue-100 text-blue-700 dark:bg-blue-950 dark:text-blue-300",
        text: "Uploading",
      },
      done: {
        className:
          "bg-emerald-100 text-emerald-700 dark:bg-emerald-950 dark:text-emerald-300",
        text: "Ready",
      },
      error: {
        className:
          "bg-red-100 text-red-700 dark:bg-red-950 dark:text-red-300",
        text: "Error",
      },
      removed: {
        className: "bg-muted text-muted-foreground",
        text: "Removed",
      },
    };

    const config = statusConfig[status];
    return <Badge className={cn("text-xs", config.className)}>{config.text}</Badge>;
  };

  const formatFileSize = (bytes?: number) => {
    if (!bytes) return "-";
    const kb = bytes / 1024;
    if (kb < 1024) return `${kb.toFixed(2)} KB`;
    return `${(kb / 1024).toFixed(2)} MB`;
  };

  if (documents.length === 0) {
    return (
      <div className="py-6 text-center text-sm text-muted-foreground">
        No documents uploaded yet. Upload documents above to get started.
      </div>
    );
  }

  return (
    <Table>
      <TableHeader>
        <TableRow>
          <TableHead>Name</TableHead>
          <TableHead className="w-[150px]">Status</TableHead>
          <TableHead className="w-[120px]">Actions</TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
        {documents.map((doc) => (
          <TableRow key={doc.uid}>
            <TableCell>
              <div className="flex items-center space-x-2">
                <span className="text-sm">{doc.name}</span>
                {doc.size && (
                  <span className="text-xs text-muted-foreground">
                    ({formatFileSize(doc.size)})
                  </span>
                )}
              </div>
            </TableCell>
            <TableCell>{getStatusBadge(doc.status)}</TableCell>
            <TableCell>
              <div className="flex items-center space-x-2">
                <TooltipProvider>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <button
                        type="button"
                        onClick={() => console.log("View", doc)}
                        className="cursor-pointer text-muted-foreground hover:text-primary"
                        aria-label="View details"
                      >
                        <Eye className="h-4 w-4" />
                      </button>
                    </TooltipTrigger>
                    <TooltipContent>View details</TooltipContent>
                  </Tooltip>
                </TooltipProvider>
                <TooltipProvider>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <button
                        type="button"
                        onClick={() => handleCopyId(doc.uid)}
                        className="cursor-pointer text-muted-foreground hover:text-primary"
                        aria-label="Copy ID"
                      >
                        <Copy className="h-4 w-4" />
                      </button>
                    </TooltipTrigger>
                    <TooltipContent>Copy ID</TooltipContent>
                  </Tooltip>
                </TooltipProvider>
                <TooltipProvider>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <button
                        type="button"
                        onClick={() => onRemove(doc.uid)}
                        className="cursor-pointer text-muted-foreground hover:text-destructive"
                        aria-label="Delete document"
                      >
                        <Trash2 className="h-4 w-4" />
                      </button>
                    </TooltipTrigger>
                    <TooltipContent>Delete</TooltipContent>
                  </Tooltip>
                </TooltipProvider>
              </div>
            </TableCell>
          </TableRow>
        ))}
      </TableBody>
    </Table>
  );
};

export default DocumentsTable;
