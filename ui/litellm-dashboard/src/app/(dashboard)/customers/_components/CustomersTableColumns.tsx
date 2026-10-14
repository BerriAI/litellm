"use client";

import { useId } from "react";

import type { ColumnDef } from "@tanstack/react-table";
import { Ban, Copy, Layers, MoreHorizontal, Pencil, Server, ShieldCheck, Trash2, Network, Wrench } from "lucide-react";

import type { EndUser } from "@/app/(dashboard)/hooks/customers/useCustomers";
import { ALL_PROXY_MCP_SERVERS_SENTINEL, NO_MCP_SERVERS_SENTINEL } from "@/components/mcp_tools/constants";
import { DataTableSortHeader } from "@/components/shared/DataTable";
import { IdentityCell, ModelsCell, SpendBudgetCell, StatusBadge } from "@/components/shared/table_cells";
import { buttonVariants } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { cn } from "@/lib/cva.config";
import { copyToClipboard } from "@/utils/dataUtils";

import { getCustomerResourceCounts } from "./customerResources";

export type CustomerTableActions = {
  canModify: boolean;
  premiumUser: boolean;
  isBlocking: boolean;
  onView: (customerId: string) => void;
  onEdit: (customerId: string) => void;
  onDelete: (customer: EndUser) => void;
  onSetBlocked: (customer: EndUser) => void;
};

export function CustomerBlockedBadge() {
  return (
    <TooltipProvider>
      <Tooltip>
        <TooltipTrigger render={<span className="inline-flex" />}>
          <Badge variant="destructive">
            <Ban className="size-3" />
            Blocked
          </Badge>
        </TooltipTrigger>
        <TooltipContent role="tooltip">
          {"Enforced only when LiteLLM Enterprise's blocked user check is enabled"}
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
}

export function CustomerBlockAction({
  customer,
  premiumUser,
  isBlocking = false,
  onSetBlocked,
}: {
  customer: EndUser;
  premiumUser: boolean;
  isBlocking?: boolean;
  onSetBlocked: (customer: EndUser) => void;
}) {
  const descriptionId = useId();
  const label = customer.blocked ? "Unblock" : "Block";
  const content = (
    <>
      {customer.blocked ? <ShieldCheck /> : <Ban />}
      {label}
    </>
  );
  if (premiumUser) {
    return (
      <DropdownMenuItem
        disabled={isBlocking}
        variant={customer.blocked ? "default" : "destructive"}
        onClick={() => onSetBlocked(customer)}
      >
        {content}
      </DropdownMenuItem>
    );
  }
  return (
    <TooltipProvider delay={300}>
      <Tooltip>
        <TooltipTrigger
          aria-describedby={descriptionId}
          render={<DropdownMenuItem aria-disabled="true" closeOnClick={false} className="opacity-50" />}
        >
          {content}
        </TooltipTrigger>
        <TooltipContent id={descriptionId} role="tooltip">
          This is a LiteLLM Enterprise feature, and requires a valid key to use.
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
}

const countLabel = (count: number, noun: string): string => (count ? `${count} ${noun}${count === 1 ? "" : "s"}` : "");

export function describeMcpGrant(permission: EndUser["object_permission"]): string {
  const servers = permission?.mcp_servers ?? [];
  if (servers.includes(NO_MCP_SERVERS_SENTINEL)) return "No MCP access";
  const counts = getCustomerResourceCounts({ object_permission: permission });
  const toolPermissionKeys = Object.keys(permission?.mcp_tool_permissions ?? {}).filter(
    (server) => server !== ALL_PROXY_MCP_SERVERS_SENTINEL && server !== NO_MCP_SERVERS_SENTINEL,
  );
  const parts = [
    countLabel(counts.servers, "server"),
    countLabel(counts.accessGroups, "group"),
    countLabel(counts.toolsets, "toolset"),
    countLabel(toolPermissionKeys.length, "tool limit"),
  ].filter(Boolean);
  const hasUnsupportedAll =
    servers.includes(ALL_PROXY_MCP_SERVERS_SENTINEL) ||
    Object.hasOwn(permission?.mcp_tool_permissions ?? {}, ALL_PROXY_MCP_SERVERS_SENTINEL);
  return parts.join(", ") || (hasUnsupportedAll ? "No MCP access" : "No restriction");
}

const RESOURCE_ITEMS = [
  { key: "models", label: "models", icon: Layers, className: "bg-info/10 text-info ring-sky-600/20" },
  {
    key: "servers",
    label: "MCP servers",
    icon: Server,
    className:
      "bg-violet-50 text-violet-700 ring-violet-600/20 dark:bg-violet-950 dark:text-violet-300 dark:ring-violet-400/30",
  },
  {
    key: "accessGroups",
    label: "MCP access groups",
    icon: Network,
    className: "bg-cyan-50 text-cyan-700 ring-cyan-600/20 dark:bg-cyan-950 dark:text-cyan-300 dark:ring-cyan-400/30",
  },
  { key: "toolsets", label: "MCP toolsets", icon: Wrench, className: "bg-warning/10 text-warning ring-amber-600/20" },
] as const;

export function CustomerResourcesCell({ customer }: { customer: EndUser }) {
  const counts = getCustomerResourceCounts(customer);
  return (
    <div className="flex items-center gap-1.5">
      {RESOURCE_ITEMS.map(({ key, label, icon: Icon, className }) => {
        const count = counts[key];
        const countLabel = count === 1 ? label.slice(0, -1) : label;
        const title = count === "all" ? `All ${label}` : `${count} ${countLabel}`;
        return (
          <span
            key={key}
            title={title}
            aria-label={title}
            className={cn(
              "inline-flex items-center gap-1.5 rounded-md px-2 py-1 text-xs font-medium ring-1 ring-inset [&_svg]:size-3.5",
              className,
            )}
          >
            <Icon aria-hidden="true" />
            <span className="tabular-nums">{count === "all" ? "All" : count}</span>
          </span>
        );
      })}
    </div>
  );
}

function CustomerRowActions({
  customer,
  canModify,
  premiumUser,
  onEdit,
  onDelete,
  onSetBlocked,
  isBlocking,
}: CustomerTableActions & { customer: EndUser }) {
  return (
    <DropdownMenu>
      <DropdownMenuTrigger
        aria-label={`Actions for ${customer.user_id}`}
        className={cn(buttonVariants({ variant: "ghost", size: "icon-sm" }), "text-muted-foreground")}
      >
        <MoreHorizontal className="size-4" />
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end">
        {canModify && (
          <>
            <DropdownMenuItem onClick={() => onEdit(customer.user_id)}>
              <Pencil />
              Edit
            </DropdownMenuItem>
            <CustomerBlockAction
              customer={customer}
              premiumUser={premiumUser}
              isBlocking={isBlocking}
              onSetBlocked={onSetBlocked}
            />
          </>
        )}
        <DropdownMenuItem onClick={() => void copyToClipboard(customer.user_id, "Customer ID copied")}>
          <Copy />
          Copy ID
        </DropdownMenuItem>
        {canModify && (
          <>
            <DropdownMenuSeparator />
            <DropdownMenuItem variant="destructive" onClick={() => onDelete(customer)}>
              <Trash2 />
              Delete
            </DropdownMenuItem>
          </>
        )}
      </DropdownMenuContent>
    </DropdownMenu>
  );
}

export function getCustomerTableColumns(actions: CustomerTableActions): ColumnDef<EndUser>[] {
  return [
    {
      id: "user_id",
      accessorKey: "user_id",
      meta: { title: "Customer ID", skeleton: "twoLine" },
      header: ({ column }) => <DataTableSortHeader column={column} title="Customer ID" />,
      size: 260,
      cell: ({ row }) => (
        <IdentityCell
          title={row.original.alias || row.original.user_id}
          subtitle={
            row.original.alias && row.original.alias !== row.original.user_id ? row.original.user_id : undefined
          }
          titleClassName={
            row.original.alias && row.original.alias !== row.original.user_id ? undefined : "font-mono text-xs"
          }
          onClick={() => actions.onView(row.original.user_id)}
        />
      ),
    },
    {
      id: "resources",
      meta: { title: "Resources", skeleton: "chips" },
      header: "Resources",
      size: 250,
      enableSorting: false,
      cell: ({ row }) => <CustomerResourcesCell customer={row.original} />,
    },
    {
      id: "spend",
      accessorKey: "spend",
      meta: { title: "Spend / Budget", skeleton: "meter", numeric: true },
      header: ({ column }) => <DataTableSortHeader column={column} title="Spend / Budget" />,
      size: 200,
      cell: ({ row }) => (
        <SpendBudgetCell
          spend={row.original.spend}
          maxBudget={row.original.litellm_budget_table?.max_budget}
          spendDecimals={2}
          budgetDecimals={2}
        />
      ),
    },
    {
      id: "max_budget",
      accessorFn: (customer) => customer.litellm_budget_table?.max_budget,
      enableHiding: false,
      enableSorting: false,
    },
    {
      id: "models",
      meta: { title: "Models", skeleton: "chips" },
      header: "Models",
      size: 240,
      enableSorting: false,
      cell: ({ row }) => <ModelsCell models={row.original.models} unrestrictedLabel="All models the key can use" />,
    },
    {
      id: "limits",
      meta: { title: "TPM / RPM Limits" },
      header: "TPM / RPM Limits",
      size: 150,
      enableSorting: false,
      cell: ({ row }) =>
        row.original.litellm_budget_table?.tpm_limit == null && row.original.litellm_budget_table?.rpm_limit == null ? (
          "-"
        ) : (
          <div className="flex flex-col text-xs text-muted-foreground">
            <span>TPM: {row.original.litellm_budget_table?.tpm_limit ?? "Unlimited"}</span>
            <span>RPM: {row.original.litellm_budget_table?.rpm_limit ?? "Unlimited"}</span>
          </div>
        ),
    },
    {
      id: "mcp_access",
      meta: { title: "MCP Access" },
      header: "MCP Access",
      size: 210,
      enableSorting: false,
      cell: ({ row }) => <span className="text-sm">{describeMcpGrant(row.original.object_permission)}</span>,
    },
    {
      id: "blocked",
      accessorKey: "blocked",
      meta: { title: "Status", skeleton: "badge" },
      header: ({ column }) => <DataTableSortHeader column={column} title="Status" />,
      size: 120,
      cell: ({ row }) =>
        row.original.blocked ? <CustomerBlockedBadge /> : <StatusBadge tone="success" label="Active" />,
    },
    {
      id: "actions",
      header: () => <span className="sr-only">Actions</span>,
      size: 64,
      enableSorting: false,
      enableHiding: false,
      cell: ({ row }) => (
        <div className="flex justify-end">
          <CustomerRowActions {...actions} customer={row.original} />
        </div>
      ),
    },
  ];
}
