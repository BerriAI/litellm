"use client";

import { useEffect, useState } from "react";
import { Card, Tabs, Badge, message } from "antd";
import { CheckCircleOutlined, CloseCircleOutlined, ClockCircleOutlined } from "@ant-design/icons";
import MCPApprovalTable from "@/components/approvals/MCPApprovalTable";
import { isAdminRole } from "@/components/utils/roles";
import { mockMCPApprovalRequests, MCPApprovalRequest } from "@/components/approvals/mockData";

export default function ApprovalsPage() {
  const [userRole, setUserRole] = useState<string | null>(null);
  const [approvalRequests, setApprovalRequests] = useState<MCPApprovalRequest[]>(mockMCPApprovalRequests);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Get user role from localStorage/session
    const role = localStorage.getItem("user_role") || null;
    setUserRole(role);
    setLoading(false);
  }, []);

  // Check if user is admin
  if (loading) {
    return (
      <div style={{ padding: "24px", textAlign: "center" }}>
        Loading...
      </div>
    );
  }

  if (!isAdminRole(userRole)) {
    return (
      <div style={{ padding: "24px" }}>
        <Card>
          <div style={{ textAlign: "center", padding: "40px" }}>
            <CloseCircleOutlined style={{ fontSize: 48, color: "#ff4d4f", marginBottom: 16 }} />
            <h2>Access Denied</h2>
            <p>Admin role required to access this page.</p>
          </div>
        </Card>
      </div>
    );
  }

  const pendingCount = approvalRequests.filter(r => r.status === "pending").length;
  const approvedCount = approvalRequests.filter(r => r.status === "approved").length;
  const rejectedCount = approvalRequests.filter(r => r.status === "rejected").length;

  const handleApprove = (approvalId: string) => {
    setApprovalRequests(prev =>
      prev.map(req =>
        req.approval_id === approvalId
          ? {
              ...req,
              status: "approved" as const,
              reviewed_at: new Date().toISOString(),
              reviewed_by: "current-admin"
            }
          : req
      )
    );
    message.success("MCP server request approved successfully");
  };

  const handleReject = (approvalId: string, reason: string) => {
    setApprovalRequests(prev =>
      prev.map(req =>
        req.approval_id === approvalId
          ? {
              ...req,
              status: "rejected" as const,
              rejection_reason: reason,
              reviewed_at: new Date().toISOString(),
              reviewed_by: "current-admin"
            }
          : req
      )
    );
    message.success("MCP server request rejected");
  };

  const tabItems = [
    {
      key: "pending",
      label: (
        <span>
          <ClockCircleOutlined /> Pending{" "}
          {pendingCount > 0 && <Badge count={pendingCount} style={{ marginLeft: 8 }} />}
        </span>
      ),
      children: (
        <MCPApprovalTable
          requests={approvalRequests.filter(r => r.status === "pending")}
          onApprove={handleApprove}
          onReject={handleReject}
        />
      ),
    },
    {
      key: "approved",
      label: (
        <span>
          <CheckCircleOutlined /> Approved{" "}
          {approvedCount > 0 && <Badge count={approvedCount} style={{ marginLeft: 8, backgroundColor: "#52c41a" }} />}
        </span>
      ),
      children: (
        <MCPApprovalTable
          requests={approvalRequests.filter(r => r.status === "approved")}
          onApprove={handleApprove}
          onReject={handleReject}
          readOnly={true}
        />
      ),
    },
    {
      key: "rejected",
      label: (
        <span>
          <CloseCircleOutlined /> Rejected{" "}
          {rejectedCount > 0 && <Badge count={rejectedCount} style={{ marginLeft: 8, backgroundColor: "#ff4d4f" }} />}
        </span>
      ),
      children: (
        <MCPApprovalTable
          requests={approvalRequests.filter(r => r.status === "rejected")}
          onApprove={handleApprove}
          onReject={handleReject}
          readOnly={true}
        />
      ),
    },
  ];

  return (
    <div style={{ padding: "24px" }}>
      <Card
        title="MCP Server Approvals"
        extra={
          <div style={{ fontSize: 14, color: "#888" }}>
            {pendingCount} pending approval{pendingCount !== 1 ? "s" : ""}
          </div>
        }
      >
        <Tabs items={tabItems} defaultActiveKey="pending" />
      </Card>
    </div>
  );
}
