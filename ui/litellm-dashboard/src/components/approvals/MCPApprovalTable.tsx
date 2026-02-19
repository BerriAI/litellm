import { useState } from "react";
import { Table, Button, Space, Tag, Tooltip, Modal, Input, Badge } from "antd";
import { EyeOutlined, CheckOutlined, CloseOutlined } from "@ant-design/icons";
import type { ColumnsType } from "antd/es/table";
import { MCPApprovalRequest } from "./mockData";
import ApprovalDetailModal from "./ApprovalDetailModal";

interface MCPApprovalTableProps {
  requests: MCPApprovalRequest[];
  onApprove: (approvalId: string) => void;
  onReject: (approvalId: string, reason: string) => void;
  readOnly?: boolean;
}

export default function MCPApprovalTable({
  requests,
  onApprove,
  onReject,
  readOnly = false
}: MCPApprovalTableProps) {
  const [selectedRequest, setSelectedRequest] = useState<MCPApprovalRequest | null>(null);
  const [showDetailModal, setShowDetailModal] = useState(false);
  const [showRejectModal, setShowRejectModal] = useState(false);
  const [rejectReason, setRejectReason] = useState("");
  const [rejectingId, setRejectingId] = useState<string | null>(null);

  const handleViewDetails = (record: MCPApprovalRequest) => {
    setSelectedRequest(record);
    setShowDetailModal(true);
  };

  const handleApproveClick = (approvalId: string) => {
    onApprove(approvalId);
  };

  const handleRejectClick = (approvalId: string) => {
    setRejectingId(approvalId);
    setShowRejectModal(true);
  };

  const handleRejectConfirm = () => {
    if (rejectingId && rejectReason.trim()) {
      onReject(rejectingId, rejectReason);
      setShowRejectModal(false);
      setRejectReason("");
      setRejectingId(null);
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case "pending":
        return "gold";
      case "approved":
        return "green";
      case "rejected":
        return "red";
      default:
        return "default";
    }
  };

  const getStatusText = (status: string) => {
    switch (status) {
      case "pending":
        return "Pending";
      case "approved":
        return "Approved";
      case "rejected":
        return "Rejected";
      default:
        return status;
    }
  };

  const columns: ColumnsType<MCPApprovalRequest> = [
    {
      title: "Status",
      dataIndex: "status",
      key: "status",
      width: 120,
      render: (status: string) => (
        <Tag color={getStatusColor(status)}>{getStatusText(status)}</Tag>
      )
    },
    {
      title: "Server Name",
      dataIndex: "server_name",
      key: "server_name",
      width: 200,
      render: (text: string, record: MCPApprovalRequest) => (
        <div>
          <div style={{ fontWeight: 500 }}>{text}</div>
          {record.alias && (
            <div style={{ fontSize: 12, color: "#888" }}>Alias: {record.alias}</div>
          )}
        </div>
      )
    },
    {
      title: "Description",
      dataIndex: "description",
      key: "description",
      ellipsis: true,
      render: (text: string) => (
        <Tooltip title={text}>
          <span>{text || "-"}</span>
        </Tooltip>
      )
    },
    {
      title: "Requested By",
      key: "requester",
      width: 200,
      render: (record: MCPApprovalRequest) => (
        <div>
          <div>{record.requester_email}</div>
          {record.team_name && (
            <Badge
              count={record.team_name}
              style={{ backgroundColor: "#1890ff", marginTop: 4 }}
            />
          )}
        </div>
      )
    },
    {
      title: "Requested Date",
      dataIndex: "created_at",
      key: "created_at",
      width: 150,
      render: (date: string) => new Date(date).toLocaleDateString()
    },
    {
      title: "Access Groups",
      dataIndex: "mcp_access_groups",
      key: "mcp_access_groups",
      width: 150,
      render: (groups: string[]) => (
        <Tooltip title={groups.join(", ")}>
          <Badge count={groups.length} style={{ backgroundColor: "#52c41a" }} />
        </Tooltip>
      )
    },
    {
      title: "Actions",
      key: "actions",
      width: 200,
      fixed: "right",
      render: (record: MCPApprovalRequest) => (
        <Space size="small">
          <Button
            size="small"
            icon={<EyeOutlined />}
            onClick={() => handleViewDetails(record)}
          >
            View
          </Button>
          {!readOnly && record.status === "pending" && (
            <>
              <Button
                size="small"
                type="primary"
                icon={<CheckOutlined />}
                onClick={() => handleApproveClick(record.approval_id)}
              >
                Approve
              </Button>
              <Button
                size="small"
                danger
                icon={<CloseOutlined />}
                onClick={() => handleRejectClick(record.approval_id)}
              >
                Reject
              </Button>
            </>
          )}
        </Space>
      )
    }
  ];

  return (
    <>
      <Table
        columns={columns}
        dataSource={requests}
        rowKey="approval_id"
        pagination={{ pageSize: 10 }}
        scroll={{ x: 1200 }}
      />

      {selectedRequest && (
        <ApprovalDetailModal
          request={selectedRequest}
          visible={showDetailModal}
          onClose={() => setShowDetailModal(false)}
          onApprove={!readOnly ? handleApproveClick : undefined}
          onReject={!readOnly ? handleRejectClick : undefined}
        />
      )}

      <Modal
        title="Reject MCP Server Request"
        open={showRejectModal}
        onOk={handleRejectConfirm}
        onCancel={() => {
          setShowRejectModal(false);
          setRejectReason("");
        }}
        okText="Confirm Rejection"
        okButtonProps={{ danger: true, disabled: !rejectReason.trim() }}
      >
        <div style={{ marginBottom: 8, fontWeight: 500 }}>
          Rejection Reason <span style={{ color: "#ff4d4f" }}>*</span>
        </div>
        <Input.TextArea
          rows={4}
          placeholder="Explain why this request is being rejected..."
          value={rejectReason}
          onChange={(e) => setRejectReason(e.target.value)}
          required
        />
        <div style={{ marginTop: 8, fontSize: 12, color: "#888" }}>
          The requester will be notified with this reason.
        </div>
      </Modal>
    </>
  );
}
