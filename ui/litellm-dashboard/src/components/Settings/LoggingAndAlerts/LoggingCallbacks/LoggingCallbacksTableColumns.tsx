"use client";

import { ColumnDef } from "@tanstack/react-table";
import { MoreHorizontal, Pencil, Play, Trash2 } from "lucide-react";

import { StatusBadge, type StatusTone } from "@/components/shared/table_cells";
import { Badge } from "@/components/ui/badge";
import { buttonVariants } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { cn } from "@/lib/cva.config";

import { AlertingObject } from "./types";

export type CallbackRow = AlertingObject & {
  mode?: "success" | "failure" | "info" | string;
};

export interface AvailableCallbackMeta {
  litellm_callback_name: string;
  litellm_callback_params: string[];
  ui_callback_name: string;
}

export type AvailableCallbacks = Record<string, AvailableCallbackMeta>;

export const callbackRowMode = (record: CallbackRow): string => record.type || record.mode || "success";

const isDestination = (record: AlertingObject): boolean => record.credentialName != null;

const CALLBACK_MODE_LABELS: Record<string, string> = {
  success: "Success",
  failure: "Failure",
  success_and_failure: "Success & Failure",
};

function callbackModeTone(mode: string): StatusTone {
  if (mode === "success") return "success";
  if (mode === "failure") return "error";
  return "info";
}

function destinationMode(record: AlertingObject) {
  if (record.autoEnable !== true) {
    return <span className="text-xs text-muted-foreground">Manual assignment</span>;
  }
  const access = record.access;
  const hasExplicitGrants = [
    access?.global === true,
    (access?.teams?.length ?? 0) > 0,
    (access?.orgs?.length ?? 0) > 0,
  ].some(Boolean);
  const tooltip = hasExplicitGrants
    ? "Exports automatically for all identities within the access scope without requiring explicit assignment."
    : "No explicit access grants. Treated as proxy-wide automatic export for backward compatibility. Add access.global=true or access.teams/orgs to scope this destination.";
  return <StatusBadge tone="warning" label="Auto-enabled" tooltip={tooltip} />;
}

function ScopeCell({ callback }: { callback: AlertingObject }) {
  const scope = callback.resolvedScope;
  const hasResolvedScope = scope?.global === true || [...(scope?.teams ?? []), ...(scope?.orgs ?? [])].length > 0;
  if (!scope || !hasResolvedScope) {
    return <span className="text-muted-foreground">—</span>;
  }
  if (scope.global) {
    return <Badge variant="secondary">Global access</Badge>;
  }
  const items = [
    ...scope.teams.map((label) => ({ kind: "team" as const, label })),
    ...scope.orgs.map((label) => ({ kind: "org" as const, label })),
  ];
  const shown = items.slice(0, 4);
  const remainder = items.length - shown.length;
  return (
    <div className="flex flex-wrap gap-1">
      {shown.map((item) => (
        <Badge key={`${item.kind}-${item.label}`} variant="outline">
          {item.kind}: {item.label}
        </Badge>
      ))}
      {remainder > 0 && <Badge variant="secondary">+{remainder} more</Badge>}
    </div>
  );
}

interface CallbackRowActionsProps {
  callback: CallbackRow;
  onTest: (callback: AlertingObject) => void | Promise<void>;
  onEdit: (callback: AlertingObject) => void;
  onDelete: (callback: AlertingObject) => void;
  onEditAccess: (callback: AlertingObject) => void;
}

function CallbackRowActions({ callback, onTest, onEdit, onDelete, onEditAccess }: CallbackRowActionsProps) {
  const destination = isDestination(callback);
  return (
    <DropdownMenu>
      <DropdownMenuTrigger
        aria-label="Open callback actions"
        data-testid={`callback-actions-${callback.name}-${callbackRowMode(callback)}`}
        className={cn(buttonVariants({ variant: "ghost", size: "icon-sm" }), "text-muted-foreground")}
      >
        <MoreHorizontal className="size-4" />
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="w-52">
        {destination ? (
          <DropdownMenuItem data-testid="destination-action-edit-access" onClick={() => onEditAccess(callback)}>
            <Pencil />
            Edit scope
          </DropdownMenuItem>
        ) : (
          <>
            <DropdownMenuItem data-testid="callback-action-test" onClick={() => void onTest(callback)}>
              <Play />
              Test
            </DropdownMenuItem>
            <DropdownMenuItem data-testid="callback-action-edit" onClick={() => onEdit(callback)}>
              <Pencil />
              Edit
            </DropdownMenuItem>
          </>
        )}
        <DropdownMenuSeparator />
        <DropdownMenuItem
          variant="destructive"
          data-testid={destination ? "destination-action-delete" : "callback-action-delete"}
          onClick={() => onDelete(callback)}
        >
          <Trash2 />
          Delete
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}

interface LoggingCallbacksTableColumnsDeps {
  availableCallbacks: AvailableCallbacks;
  onTest: (callback: AlertingObject) => void | Promise<void>;
  onEdit: (callback: AlertingObject) => void;
  onDelete: (callback: AlertingObject) => void;
  onEditAccess: (callback: AlertingObject) => void;
}

export const getLoggingCallbacksTableColumns = ({
  availableCallbacks,
  onTest,
  onEdit,
  onDelete,
  onEditAccess,
}: LoggingCallbacksTableColumnsDeps): ColumnDef<CallbackRow>[] => [
  {
    id: "name",
    accessorKey: "name",
    meta: { title: "Callback Name" },
    header: "Callback Name",
    enableSorting: false,
    cell: ({ row }) => {
      const id = row.original.name;
      const displayName = availableCallbacks[id]?.ui_callback_name || id;
      return (
        <div>
          <span className="block max-w-72 truncate text-sm font-medium" title={displayName}>
            {displayName}
          </span>
          {row.original.destinationLabel && (
            <span className="block text-xs text-muted-foreground">{row.original.destinationLabel}</span>
          )}
        </div>
      );
    },
  },
  {
    id: "mode",
    meta: { title: "Mode", skeleton: "badge" },
    header: "Mode",
    size: 240,
    enableSorting: false,
    cell: ({ row }) => {
      if (isDestination(row.original)) {
        return destinationMode(row.original);
      }
      const mode = callbackRowMode(row.original);
      return <StatusBadge tone={callbackModeTone(mode)} label={CALLBACK_MODE_LABELS[mode] || mode} />;
    },
  },
  {
    id: "access",
    meta: { title: "Scope", skeleton: "badge" },
    header: "Scope",
    size: 280,
    enableSorting: false,
    cell: ({ row }) =>
      isDestination(row.original) ? (
        <ScopeCell callback={row.original} />
      ) : (
        <span className="text-muted-foreground">—</span>
      ),
  },
  {
    id: "actions",
    meta: { className: "text-right", headerClassName: "text-right" },
    header: () => <span className="sr-only">Actions</span>,
    size: 64,
    enableSorting: false,
    enableHiding: false,
    cell: ({ row }) => (
      <div className="flex justify-end">
        <CallbackRowActions
          callback={row.original}
          onTest={onTest}
          onEdit={onEdit}
          onDelete={onDelete}
          onEditAccess={onEditAccess}
        />
      </div>
    ),
  },
];
