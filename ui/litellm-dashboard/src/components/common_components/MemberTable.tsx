import { Member } from "@/components/networking";
import { CrownOutlined, InfoCircleOutlined, UserAddOutlined, UserOutlined } from "@ant-design/icons";
import { Button, Pagination, Space, Table, Tag, Tooltip, Typography } from "antd";
import type { ColumnsType } from "antd/es/table";
import React, { useState } from "react";
import TableIconActionButton from "./IconActionButton/TableIconActionButtons/TableIconActionButton";

const { Text } = Typography;

export interface MemberTableProps {
  members: Member[];
  canEdit: boolean;
  onEdit: (member: Member) => void;
  onDelete: (member: Member) => void;
  onAddMember?: () => void;
  roleColumnTitle?: string;
  roleTooltip?: string;
  extraColumns?: ColumnsType<Member>;
  showDeleteForMember?: (member: Member) => boolean;
  emptyText?: string;
  loading?: boolean;
  /** When true, renders top-right pagination controls instead of the default antd bottom pagination. */
  withPagination?: boolean;
  defaultPageSize?: number;
}

export default function MemberTable({
  members,
  canEdit,
  onEdit,
  onDelete,
  onAddMember,
  roleColumnTitle = "Role",
  roleTooltip,
  extraColumns = [],
  showDeleteForMember,
  emptyText,
  loading,
  withPagination = false,
  defaultPageSize = 50,
}: MemberTableProps) {
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(defaultPageSize);

  const total = members.length;
  const totalPages = Math.max(1, Math.ceil(total / pageSize));
  const safePage = Math.min(page, totalPages);
  const pagedMembers = withPagination ? members.slice((safePage - 1) * pageSize, safePage * pageSize) : members;
  const baseColumns: ColumnsType<Member> = [
    {
      title: "User Email",
      dataIndex: "user_email",
      key: "user_email",
      render: (email: string | null) => <Text>{email || "-"}</Text>,
    },
    {
      title: "User ID",
      dataIndex: "user_id",
      key: "user_id",
      render: (userId: string | null) =>
        userId === "default_user_id" ? (
          <Tag color="blue">Default Proxy Admin</Tag>
        ) : (
          <Text>{userId || "-"}</Text>
        ),
    },
    {
      title: roleTooltip ? (
        <Space direction="horizontal">
          {roleColumnTitle}
          <Tooltip title={roleTooltip}>
            <InfoCircleOutlined />
          </Tooltip>
        </Space>
      ) : (
        roleColumnTitle
      ),
      dataIndex: "role",
      key: "role",
      render: (role: string) => (
        <Space>
          {role?.toLowerCase() === "admin" || role?.toLowerCase() === "org_admin" ? (
            <CrownOutlined />
          ) : (
            <UserOutlined />
          )}
          <Text style={{ textTransform: "capitalize" }}>{role || "-"}</Text>
        </Space>
      ),
    },
    ...extraColumns,
    {
      title: "Actions",
      key: "actions",
      fixed: "right" as const,
      width: 120,
      render: (_: unknown, record: Member) =>
        canEdit ? (
          <Space>
            <TableIconActionButton
              variant="Edit"
              tooltipText="Edit member"
              dataTestId="edit-member"
              onClick={() => onEdit(record)}
            />
            {(!showDeleteForMember || showDeleteForMember(record)) && (
              <TableIconActionButton
                variant="Delete"
                tooltipText="Delete member"
                dataTestId="delete-member"
                onClick={() => onDelete(record)}
              />
            )}
          </Space>
        ) : null,
    },
  ];

  return (
    <Space direction="vertical" style={{ width: "100%" }}>
      <div className="flex items-center justify-between w-full">
        <span className="text-sm text-gray-700">
          {total} Member{total !== 1 ? "s" : ""}
        </span>
        {withPagination && (
          <Pagination
            size="small"
            current={safePage}
            pageSize={pageSize}
            total={total}
            showSizeChanger
            pageSizeOptions={["10", "25", "50", "100"]}
            showQuickJumper
            onChange={(p, ps) => { setPage(p); if (ps !== pageSize) { setPageSize(ps); setPage(1); } }}
          />
        )}
      </div>
      <Table
        columns={baseColumns}
        dataSource={pagedMembers}
        rowKey={(record) => record.user_id ?? record.user_email ?? JSON.stringify(record)}
        pagination={false}
        loading={loading}
        size="small"
        scroll={{ x: "max-content" }}
        locale={emptyText ? { emptyText } : undefined}
      />
      {onAddMember && canEdit && (
        <Button icon={<UserAddOutlined />} type="primary" onClick={onAddMember}>
          Add Member
        </Button>
      )}
    </Space>
  );
}
