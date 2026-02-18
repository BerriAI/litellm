import React from "react";
import { Table, Badge, Tooltip, message } from "antd";
import { EyeOutlined, CopyOutlined, DeleteOutlined } from "@ant-design/icons";
import { DocumentUpload } from "./types";

interface DocumentsTableProps {
  documents: DocumentUpload[];
  onRemove: (uid: string) => void;
}

const DocumentsTable: React.FC<DocumentsTableProps> = ({ documents, onRemove }) => {
  const handleCopyId = (uid: string) => {
    navigator.clipboard.writeText(uid);
    message.success("Document ID copied to clipboard");
  };

  const getStatusBadge = (status: DocumentUpload["status"]) => {
    const statusConfig = {
      uploading: { color: "blue", text: "Uploading" },
      done: { color: "green", text: "Ready" },
      error: { color: "red", text: "Error" },
      removed: { color: "default", text: "Removed" },
    };

    const config = statusConfig[status];
    return <Badge color={config.color} text={config.text} />;
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
          {record.size && <span className="text-xs text-gray-400">({formatFileSize(record.size)})</span>}
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
      render: (_: any, record: DocumentUpload) => (
        <div className="flex items-center space-x-2">
          <Tooltip title="View details">
            <EyeOutlined
              className="cursor-pointer text-gray-600 hover:text-blue-500"
              onClick={() => console.log("View", record)}
            />
          </Tooltip>
          <Tooltip title="Copy ID">
            <CopyOutlined
              className="cursor-pointer text-gray-600 hover:text-blue-500"
              onClick={() => handleCopyId(record.uid)}
            />
          </Tooltip>
          <Tooltip title="Remove">
            <DeleteOutlined
              className="cursor-pointer text-gray-600 hover:text-red-500"
              onClick={() => onRemove(record.uid)}
            />
          </Tooltip>
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
        emptyText: "No documents uploaded yet. Upload documents above to get started.",
      }}
      size="small"
    />
  );
};

export default DocumentsTable;
