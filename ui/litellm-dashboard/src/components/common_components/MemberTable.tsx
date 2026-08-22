import { SimpleTooltip } from "@/components/ui/tooltip";
import { Member } from "@/components/networking";
import { StatusBadge } from "@/components/shared/table_cells";
import { Button } from "@/components/ui/button";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Crown, Info, User, UserPlus } from "lucide-react";
import React from "react";
import TableIconActionButton from "./IconActionButton/TableIconActionButtons/TableIconActionButton";

export interface MemberTableColumn {
  title: React.ReactNode;
  key: React.Key;
  dataIndex?: keyof Member;
  render?: (value: Member[keyof Member], member: Member, index: number) => React.ReactNode;
}

export interface MemberTableProps {
  members: Member[];
  canEdit: boolean;
  onEdit: (member: Member) => void;
  onDelete: (member: Member) => void;
  onAddMember?: () => void;
  roleColumnTitle?: string;
  roleTooltip?: string;
  extraColumns?: MemberTableColumn[];
  showDeleteForMember?: (member: Member) => boolean;
  emptyText?: string;
}

const extraColumnCell = (column: MemberTableColumn, member: Member, index: number): React.ReactNode => {
  const value = column.dataIndex ? member[column.dataIndex] : undefined;
  return column.render ? column.render(value, member, index) : value;
};

const STICKY_ACTIONS_CLASS = "sticky right-0 w-[120px] bg-background";

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
  return (
    <div className="flex w-full flex-col gap-2">
      <span className="inline-flex text-sm text-foreground">
        {members.length} Member{members.length !== 1 ? "s" : ""}
      </span>
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>User Email</TableHead>
            <TableHead>User ID</TableHead>
            <TableHead>
              {roleTooltip ? (
                <span className="inline-flex items-center gap-2">
                  {roleColumnTitle}
                  <SimpleTooltip content={roleTooltip}>
                    <Info className="size-3.5" />
                  </SimpleTooltip>
                </span>
              ) : (
                roleColumnTitle
              )}
            </TableHead>
            {extraColumns.map((column) => (
              <TableHead key={column.key}>{column.title}</TableHead>
            ))}
            <TableHead className={STICKY_ACTIONS_CLASS}>Actions</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {members.length === 0 ? (
            <TableRow>
              <TableCell colSpan={extraColumns.length + 4} className="text-center text-muted-foreground">
                {emptyText ?? "No data"}
              </TableCell>
            </TableRow>
          ) : (
            members.map((member, memberIndex) => (
              <TableRow key={member.user_id ?? member.user_email ?? JSON.stringify(member)}>
                <TableCell>{member.user_email || "-"}</TableCell>
                <TableCell>
                  {member.user_id === "default_user_id" ? (
                    <StatusBadge tone="info" label="Default Proxy Admin" />
                  ) : (
                    member.user_id || "-"
                  )}
                </TableCell>
                <TableCell>
                  <span className="inline-flex items-center gap-2">
                    {member.role?.toLowerCase() === "admin" || member.role?.toLowerCase() === "org_admin" ? (
                      <Crown className="size-3.5" />
                    ) : (
                      <User className="size-3.5" />
                    )}
                    <span className="capitalize">{member.role || "-"}</span>
                  </span>
                </TableCell>
                {extraColumns.map((column) => (
                  <TableCell key={column.key}>{extraColumnCell(column, member, memberIndex)}</TableCell>
                ))}
                <TableCell className={STICKY_ACTIONS_CLASS}>
                  {canEdit ? (
                    <span className="inline-flex items-center gap-2">
                      <TableIconActionButton
                        variant="Edit"
                        tooltipText="Edit member"
                        dataTestId="edit-member"
                        onClick={() => onEdit(member)}
                      />
                      {(!showDeleteForMember || showDeleteForMember(member)) && (
                        <TableIconActionButton
                          variant="Delete"
                          tooltipText="Delete member"
                          dataTestId="delete-member"
                          onClick={() => onDelete(member)}
                        />
                      )}
                    </span>
                  ) : null}
                </TableCell>
              </TableRow>
            ))
          )}
        </TableBody>
      </Table>
      {onAddMember && canEdit && (
        <Button onClick={onAddMember} className="self-start">
          <UserPlus className="size-4" />
          Add Member
        </Button>
      )}
    </div>
  );
}
