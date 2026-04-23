import React from "react";
import { Table } from "antd";
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

  const columns = [
    {
      title: "Name",
      dataIndex: "name",
      key: "name",
      render: (name: string, record: DocumentUpload) => (
        <div className="flex items-center space-x-2">
          <span className="text-sm">{name}</span>
          {record.size && (
            <span className="text-xs text-muted-foreground">
              ({formatFileSize(record.size)})
            </span>
          )}
        </div>
      ),
    },
    {
      title: "Status",
      dataIndex: "status",
      key: "status",
      width: 150,
      render: (status: DocumentUpload["status"]) => getStatusBadge(status),
    },
    {
      title: "Actions",
      key: "actions",
      width: 120,
      render: (_: unknown, record: DocumentUpload) => (
        <div className="flex items-center space-x-2">
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <button
                  type="button"
                  onClick={() => console.log("View", record)}
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
                  onClick={() => handleCopyId(record.uid)}
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
                  onClick={() => onRemove(record.uid)}
                  className="cursor-pointer text-muted-foreground hover:text-destructive"
                  aria-label="Remove"
                >
                  <Trash2 className="h-4 w-4" />
                </button>
              </TooltipTrigger>
              <TooltipContent>Remove</TooltipContent>
            </Tooltip>
          </TooltipProvider>
        </div>
      ),
    },
  ];

  return (
    <Table
      dataSource={documents}
      columns={columns}
      rowKey="uid"
      pagination={false}
      locale={{
        emptyText:
          "No documents uploaded yet. Upload documents above to get started.",
      }}
      size="small"
    />
  );
};

export default DocumentsTable;
