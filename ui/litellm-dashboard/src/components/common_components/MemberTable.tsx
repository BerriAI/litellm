import { Member } from "@/components/networking";
import { Table } from "antd";
import type { ColumnsType } from "antd/es/table";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { Crown, Info, User, UserPlus } from "lucide-react";
import React from "react";
import TableIconActionButton from "./IconActionButton/TableIconActionButtons/TableIconActionButton";

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
}: MemberTableProps) {
  const baseColumns: ColumnsType<Member> = [
    {
      title: "User Email",
      dataIndex: "user_email",
      key: "user_email",
      render: (email: string | null) => <span>{email || "-"}</span>,
    },
    {
      title: "User ID",
      dataIndex: "user_id",
      key: "user_id",
      render: (userId: string | null) =>
        userId === "default_user_id" ? (
          <Badge className="bg-blue-100 text-blue-700 dark:bg-blue-950 dark:text-blue-300">
            Default Proxy Admin
          </Badge>
        ) : (
          <span>{userId || "-"}</span>
        ),
    },
    {
      title: roleTooltip ? (
        <div className="flex items-center gap-2">
          {roleColumnTitle}
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <Info className="h-3 w-3 text-muted-foreground" />
              </TooltipTrigger>
              <TooltipContent>{roleTooltip}</TooltipContent>
            </Tooltip>
          </TooltipProvider>
        </div>
      ) : (
        roleColumnTitle
      ),
      dataIndex: "role",
      key: "role",
      render: (role: string) => (
        <div className="flex items-center gap-2">
          {role?.toLowerCase() === "admin" ||
          role?.toLowerCase() === "org_admin" ? (
            <Crown className="h-3.5 w-3.5" />
          ) : (
            <User className="h-3.5 w-3.5" />
          )}
          <span className="capitalize">{role || "-"}</span>
        </div>
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
          <div className="flex items-center gap-2">
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
          </div>
        ) : null,
    },
  ];

  return (
    <div className="w-full flex flex-col gap-2">
      <span className="inline-flex text-sm text-foreground/80">
        {members.length} Member{members.length !== 1 ? "s" : ""}
      </span>
      <Table
        columns={baseColumns}
        dataSource={members}
        rowKey={(record) =>
          record.user_id ?? record.user_email ?? JSON.stringify(record)
        }
        pagination={false}
        size="small"
        scroll={{ x: "max-content" }}
        locale={emptyText ? { emptyText } : undefined}
      />
      {onAddMember && canEdit && (
        <div>
          <Button onClick={onAddMember}>
            <UserPlus className="h-4 w-4" />
            Add Member
          </Button>
        </div>
      )}
    </div>
  );
}
