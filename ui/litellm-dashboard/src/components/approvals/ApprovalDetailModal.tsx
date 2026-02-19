import { Modal, Descriptions, Tag, Button, Space, Divider } from "antd";
import { MCPApprovalRequest } from "./mockData";

interface ApprovalDetailModalProps {
  request: MCPApprovalRequest;
  visible: boolean;
  onClose: () => void;
  onApprove?: (approvalId: string) => void;
  onReject?: (approvalId: string) => void;
}

export default function ApprovalDetailModal({
  request,
  visible,
  onClose,
  onApprove,
  onReject
}: ApprovalDetailModalProps) {
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

  return (
    <Modal
      title={`MCP Server Request: ${request.server_name}`}
      open={visible}
      onCancel={onClose}
      width={800}
      footer={
        <Space>
          <Button onClick={onClose}>Close</Button>
          {onApprove && request.status === "pending" && (
            <>
              <Button
                danger
                onClick={() => {
                  onReject?.(request.approval_id);
                  onClose();
                }}
              >
                Reject
              </Button>
              <Button
                type="primary"
                onClick={() => {
                  onApprove(request.approval_id);
                  onClose();
                }}
              >
                Approve
              </Button>
            </>
          )}
        </Space>
      }
    >
      <Descriptions column={1} bordered>
        <Descriptions.Item label="Status">
          <Tag color={getStatusColor(request.status)}>
            {request.status.toUpperCase()}
          </Tag>
        </Descriptions.Item>

        <Descriptions.Item label="Server Name">{request.server_name}</Descriptions.Item>
        <Descriptions.Item label="Alias">{request.alias || "-"}</Descriptions.Item>
        <Descriptions.Item label="Description">{request.description || "-"}</Descriptions.Item>
        <Descriptions.Item label="URL">{request.url}</Descriptions.Item>
        <Descriptions.Item label="Transport">{request.transport}</Descriptions.Item>
        <Descriptions.Item label="Auth Type">{request.auth_type || "None"}</Descriptions.Item>

        <Descriptions.Item label="Access Groups">
          {request.mcp_access_groups.length > 0 ? (
            request.mcp_access_groups.map(group => (
              <Tag key={group} color="blue" style={{ marginBottom: 4 }}>
                {group}
              </Tag>
            ))
          ) : (
            "-"
          )}
        </Descriptions.Item>

        <Descriptions.Item label="Allowed Tools">
          {request.allowed_tools.length > 0 ? (
            <div style={{ display: "flex", flexWrap: "wrap", gap: 4 }}>
              {request.allowed_tools.map(tool => (
                <Tag key={tool} style={{ marginBottom: 4 }}>
                  {tool}
                </Tag>
              ))}
            </div>
          ) : (
            "-"
          )}
        </Descriptions.Item>
      </Descriptions>

      <Divider>Requester Information</Divider>

      <Descriptions column={1} bordered>
        <Descriptions.Item label="Requester Email">{request.requester_email}</Descriptions.Item>
        <Descriptions.Item label="Team">{request.team_name || "-"}</Descriptions.Item>
        <Descriptions.Item label="Requested Date">
          {new Date(request.created_at).toLocaleString()}
        </Descriptions.Item>
      </Descriptions>

      {request.request_metadata && (
        <>
          <Divider>Request Metadata</Divider>
          <Descriptions column={1} bordered>
            {request.request_metadata.business_justification && (
              <Descriptions.Item label="Business Justification">
                {request.request_metadata.business_justification}
              </Descriptions.Item>
            )}
            {request.request_metadata.expected_usage && (
              <Descriptions.Item label="Expected Usage">
                {request.request_metadata.expected_usage}
              </Descriptions.Item>
            )}
          </Descriptions>
        </>
      )}

      {request.status !== "pending" && (
        <>
          <Divider>Review Information</Divider>
          <Descriptions column={1} bordered>
            <Descriptions.Item label="Reviewed At">
              {request.reviewed_at ? new Date(request.reviewed_at).toLocaleString() : "-"}
            </Descriptions.Item>
            {request.rejection_reason && (
              <Descriptions.Item label="Rejection Reason">
                <div style={{ color: "#ff4d4f" }}>{request.rejection_reason}</div>
              </Descriptions.Item>
            )}
          </Descriptions>
        </>
      )}
    </Modal>
  );
}
