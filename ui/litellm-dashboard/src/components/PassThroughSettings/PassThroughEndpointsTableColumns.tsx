"use client";

import { ColumnDef } from "@tanstack/react-table";
import { Eye, EyeOff, Info, MoreHorizontal, Pencil, Trash2 } from "lucide-react";
import React, { useState } from "react";

import { CellTooltip, IdentityCell, StatusBadge } from "@/components/shared/table_cells";
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

import type { passThroughItem } from "./PassThroughSettings";

function HeaderWithTooltip({ title, tooltip }: { title: string; tooltip: string }) {
  return (
    <div className="flex items-center gap-1">
      <span>{title}</span>
      <CellTooltip content={tooltip} trigger={<Info className="size-3.5 cursor-help text-muted-foreground" />} />
    </div>
  );
}

function HeadersCell({ value }: { value: object }) {
  const [showHeaders, setShowHeaders] = useState(false);
  const headerString = JSON.stringify(value);

  return (
    <div className="flex items-center gap-2">
      <span className="block max-w-60 truncate font-mono text-xs">{showHeaders ? headerString : "••••••••"}</span>
      <button
        type="button"
        onClick={() => setShowHeaders(!showHeaders)}
        aria-label={showHeaders ? "Hide headers" : "Show headers"}
        className="rounded-sm p-1 hover:bg-muted"
      >
        {showHeaders ? (
          <EyeOff className="size-4 text-muted-foreground" />
        ) : (
          <Eye className="size-4 text-muted-foreground" />
        )}
      </button>
    </div>
  );
}

function MethodsCell({ methods }: { methods: string[] | undefined }) {
  if (!methods || methods.length === 0) {
    return <Badge variant="secondary">ALL</Badge>;
  }
  return (
    <div className="flex flex-wrap gap-1">
      {methods.map((method) => (
        <Badge key={method} variant="outline" className="font-mono text-xs font-normal">
          {method}
        </Badge>
      ))}
    </div>
  );
}

interface EndpointRowActionsProps {
  endpoint: passThroughItem;
  onEndpointClick: (endpointId: string) => void;
  onDeleteClick: (endpointId: string) => void;
}

function EndpointRowActions({ endpoint, onEndpointClick, onDeleteClick }: EndpointRowActionsProps) {
  const endpointId = endpoint.id;
  return (
    <DropdownMenu>
      <DropdownMenuTrigger
        aria-label="Open endpoint actions"
        data-testid={`endpoint-actions-${endpointId || endpoint.path}`}
        className={cn(buttonVariants({ variant: "ghost", size: "icon-sm" }), "text-muted-foreground")}
      >
        <MoreHorizontal className="size-4" />
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="w-52">
        <DropdownMenuItem
          data-testid="endpoint-action-edit"
          disabled={!endpointId}
          onClick={() => endpointId && onEndpointClick(endpointId)}
        >
          <Pencil />
          Edit
        </DropdownMenuItem>
        <DropdownMenuSeparator />
        <DropdownMenuItem
          variant="destructive"
          data-testid="endpoint-action-delete"
          disabled={!endpointId}
          onClick={() => endpointId && onDeleteClick(endpointId)}
        >
          <Trash2 />
          Delete
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}

interface PassThroughEndpointsTableColumnsDeps {
  onEndpointClick: (endpointId: string) => void;
  onDeleteClick: (endpointId: string) => void;
}

export const getPassThroughEndpointsTableColumns = ({
  onEndpointClick,
  onDeleteClick,
}: PassThroughEndpointsTableColumnsDeps): ColumnDef<passThroughItem>[] => [
  {
    id: "id",
    accessorKey: "id",
    meta: { title: "ID" },
    header: "ID",
    size: 190,
    enableSorting: false,
    cell: ({ row }) => {
      const endpointId = row.original.id;
      if (!endpointId) return <span className="font-mono text-xs text-muted-foreground">—</span>;
      return (
        <IdentityCell
          title={endpointId}
          titleClassName="font-mono text-xs font-normal"
          onClick={() => onEndpointClick(endpointId)}
        />
      );
    },
  },
  {
    id: "path",
    accessorKey: "path",
    meta: { title: "Path" },
    header: "Path",
    size: 200,
    enableSorting: false,
    cell: ({ row }) => (
      <span className="block max-w-60 truncate text-sm font-medium" title={row.original.path}>
        {row.original.path}
      </span>
    ),
  },
  {
    id: "target",
    accessorKey: "target",
    meta: { title: "Target" },
    header: "Target",
    size: 240,
    enableSorting: false,
    cell: ({ row }) => (
      <span className="block max-w-72 truncate text-sm" title={row.original.target}>
        {row.original.target}
      </span>
    ),
  },
  {
    id: "methods",
    meta: { title: "Methods", skeleton: "chips" },
    header: () => <HeaderWithTooltip title="Methods" tooltip="HTTP methods supported by this endpoint" />,
    size: 150,
    enableSorting: false,
    cell: ({ row }) => <MethodsCell methods={row.original.methods} />,
  },
  {
    id: "auth",
    accessorKey: "auth",
    meta: { title: "Authentication", skeleton: "badge" },
    header: () => <HeaderWithTooltip title="Authentication" tooltip="LiteLLM Virtual Key required to call endpoint" />,
    size: 140,
    enableSorting: false,
    cell: ({ row }) => (
      <StatusBadge tone={row.original.auth ? "success" : "neutral"} label={row.original.auth ? "Yes" : "No"} />
    ),
  },
  {
    id: "headers",
    meta: { title: "Headers" },
    header: "Headers",
    size: 180,
    enableSorting: false,
    cell: ({ row }) => <HeadersCell value={row.original.headers || {}} />,
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
        <EndpointRowActions endpoint={row.original} onEndpointClick={onEndpointClick} onDeleteClick={onDeleteClick} />
      </div>
    ),
  },
];
