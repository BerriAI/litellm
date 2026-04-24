import { Member } from "@/components/networking";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { Crown, Info, User, UserPlus } from "lucide-react";
import React from "react";
import TableIconActionButton from "./IconActionButton/TableIconActionButtons/TableIconActionButton";

export interface MemberTableExtraColumn {
  title: React.ReactNode;
  key: string;
  render: (_: unknown, record: Member) => React.ReactNode;
}

export interface MemberTableProps {
  members: Member[];
  canEdit: boolean;
  onEdit: (member: Member) => void;
  onDelete: (member: Member) => void;
  onAddMember?: () => void;
  roleColumnTitle?: string;
  roleTooltip?: string;
  extraColumns?: MemberTableExtraColumn[];
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
  const rowKey = (record: Member) =>
    record.user_id ?? record.user_email ?? JSON.stringify(record);

  return (
    <div className="w-full flex flex-col gap-2">
      <span className="inline-flex text-sm text-foreground/80">
        {members.length} Member{members.length !== 1 ? "s" : ""}
      </span>
      <div className="overflow-x-auto">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>User Email</TableHead>
              <TableHead>User ID</TableHead>
              <TableHead>
                {roleTooltip ? (
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
                )}
              </TableHead>
              {extraColumns.map((col) => (
                <TableHead key={col.key}>{col.title}</TableHead>
              ))}
              {canEdit && (
                <TableHead className="w-[120px] text-right">Actions</TableHead>
              )}
            </TableRow>
          </TableHeader>
          <TableBody>
            {members.length === 0 ? (
              <TableRow>
                <TableCell
                  colSpan={
                    3 + extraColumns.length + (canEdit ? 1 : 0)
                  }
                  className="text-center text-muted-foreground py-8"
                >
                  {emptyText || "No members"}
                </TableCell>
              </TableRow>
            ) : (
              members.map((record) => (
                <TableRow key={rowKey(record)}>
                  <TableCell>
                    <span>{record.user_email || "-"}</span>
                  </TableCell>
                  <TableCell>
                    {record.user_id === "default_user_id" ? (
                      <Badge className="bg-blue-100 text-blue-700 dark:bg-blue-950 dark:text-blue-300">
                        Default Proxy Admin
                      </Badge>
                    ) : (
                      <span>{record.user_id || "-"}</span>
                    )}
                  </TableCell>
                  <TableCell>
                    <div className="flex items-center gap-2">
                      {record.role?.toLowerCase() === "admin" ||
                      record.role?.toLowerCase() === "org_admin" ? (
                        <Crown className="h-3.5 w-3.5" />
                      ) : (
                        <User className="h-3.5 w-3.5" />
                      )}
                      <span className="capitalize">{record.role || "-"}</span>
                    </div>
                  </TableCell>
                  {extraColumns.map((col) => (
                    <TableCell key={col.key}>
                      {col.render(undefined, record)}
                    </TableCell>
                  ))}
                  {canEdit && (
                    <TableCell className="w-[120px]">
                      <div className="flex items-center gap-2 justify-end">
                        <TableIconActionButton
                          variant="Edit"
                          tooltipText="Edit member"
                          dataTestId="edit-member"
                          onClick={() => onEdit(record)}
                        />
                        {(!showDeleteForMember ||
                          showDeleteForMember(record)) && (
                          <TableIconActionButton
                            variant="Delete"
                            tooltipText="Delete member"
                            dataTestId="delete-member"
                            onClick={() => onDelete(record)}
                          />
                        )}
                      </div>
                    </TableCell>
                  )}
                </TableRow>
              ))
            )}
          </TableBody>
        </Table>
      </div>
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
