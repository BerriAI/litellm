"use client";

import { InfoCircleOutlined } from "@ant-design/icons";
import { formatNumberWithCommas } from "@/utils/dataUtils";
import { ChevronDownIcon, ChevronRightIcon } from "@heroicons/react/outline";
import { ColumnDef } from "@tanstack/react-table";
import { Badge, Button, Icon, Text } from "@tremor/react";
import { Popover, Tooltip } from "antd";
import { getModelDisplayName } from "../key_team_helpers/fetch_available_models_team_key";
import { KeyResponse } from "../key_team_helpers/key_list";

export interface VirtualKeysColumnsOptions {
  setSelectedKey: (key: KeyResponse | Record<string, unknown>) => void;
  teams?: Array<{ team_id: string; team_alias: string }> | null;
  expandedAccordions: Record<string, boolean>;
  setExpandedAccordions: React.Dispatch<React.SetStateAction<Record<string, boolean>>>;
}

/**
 * Shared column definitions for virtual keys tables (VirtualKeysTable, TeamKeysTab).
 * Uses token ?? token_id for Key ID to support team keys where token is not returned.
 */
export function getVirtualKeysColumns(options: VirtualKeysColumnsOptions): ColumnDef<KeyResponse | Record<string, unknown>>[] {
  const { setSelectedKey, teams, expandedAccordions, setExpandedAccordions } = options;

  return [
    {
      id: "expander",
      header: () => null,
      size: 40,
      enableSorting: false,
      cell: ({ row }) =>
        row.getCanExpand() ? (
          <button onClick={row.getToggleExpandedHandler()} style={{ cursor: "pointer" }}>
            {row.getIsExpanded() ? "▼" : "▶"}
          </button>
        ) : null,
    },
    {
      id: "token",
      accessorFn: (row) => (row as KeyResponse).token ?? (row as Record<string, unknown>).token_id,
      header: "Key ID",
      size: 100,
      enableSorting: true,
      cell: (info) => {
        const value = info.getValue() as string;
        const width = info.cell.column.getSize();
        const row = info.row.original;
        return (
          <Tooltip title={value}>
            <Button
              size="xs"
              variant="light"
              className="font-mono text-blue-500 bg-blue-50 hover:bg-blue-100 text-xs font-normal px-2 py-0.5 text-left overflow-hidden truncate block"
              style={{ maxWidth: width, overflow: "hidden" }}
              onClick={() => setSelectedKey(row)}
            >
              {value ?? "-"}
            </Button>
          </Tooltip>
        );
      },
    },
    {
      id: "key_alias",
      accessorKey: "key_alias",
      header: "Key Alias",
      size: 150,
      enableSorting: true,
      cell: (info) => {
        const value = info.getValue() as string;
        const width = info.cell.column.getSize();
        return (
          <Tooltip title={value}>
            <span className={`font-mono text-xs truncate block`} style={{ maxWidth: width, overflow: "hidden" }}>
              {value ?? "-"}
            </span>
          </Tooltip>
        );
      },
    },
    {
      id: "key_name",
      accessorKey: "key_name",
      header: "Secret Key",
      size: 120,
      enableSorting: false,
      cell: (info) => <span className="font-mono text-xs">{info.getValue() as string ?? "-"}</span>,
    },
    {
      id: "team_alias",
      accessorKey: "team_id",
      header: "Team Alias",
      size: 120,
      enableSorting: false,
      cell: ({ getValue }) => {
        const teamId = getValue() as string;
        const team = teams?.find((t) => t.team_id === teamId);
        return team?.team_alias || "Unknown";
      },
    },
    {
      id: "team_id",
      accessorKey: "team_id",
      header: "Team ID",
      size: 80,
      enableSorting: false,
      cell: (info) => {
        const value = info.getValue() as string | null;
        const width = info.cell.column.getSize();
        return (
          <Tooltip title={value}>
            <span className={`font-mono text-xs truncate block`} style={{ maxWidth: width, overflow: "hidden" }}>
              {value ?? "-"}
            </span>
          </Tooltip>
        );
      },
    },
    {
      id: "organization_id",
      accessorKey: "organization_id",
      header: "Organization ID",
      size: 140,
      enableSorting: false,
      cell: (info) => (info.getValue() ? info.renderValue() : "-"),
    },
    {
      id: "user_email",
      accessorFn: (row) => {
        const r = row as Record<string, unknown>;
        const user = r.user as { user_email?: string } | undefined;
        return user?.user_email ?? r.user_email ?? null;
      },
      header: "User Email",
      size: 160,
      enableSorting: false,
      cell: (info) => {
        const value = info.getValue() as string | null;
        const width = info.cell.column.getSize();
        return (
          <Tooltip title={value}>
            <span className={`font-mono text-xs truncate block`} style={{ maxWidth: width, overflow: "hidden" }}>
              {value ?? "-"}
            </span>
          </Tooltip>
        );
      },
    },
    {
      id: "user_id",
      accessorKey: "user_id",
      header: "User ID",
      size: 70,
      enableSorting: false,
      cell: (info) => {
        const userId = info.getValue() as string | null;
        const displayValue = userId === "default_user_id" ? "Default Proxy Admin" : userId;
        const width = info.cell.column.getSize();
        return (
          <Tooltip title={displayValue}>
            <span className={`font-mono text-xs truncate block`} style={{ maxWidth: width, overflow: "hidden" }}>
              {displayValue ?? "-"}
            </span>
          </Tooltip>
        );
      },
    },
    {
      id: "created_at",
      accessorKey: "created_at",
      header: "Created At",
      size: 120,
      enableSorting: true,
      cell: (info) => {
        const value = info.getValue();
        return value ? new Date(value as string).toLocaleDateString() : "-";
      },
    },
    {
      id: "created_by",
      accessorKey: "created_by",
      header: "Created By",
      size: 70,
      enableSorting: false,
      cell: (info) => {
        const value = info.getValue() as string | null;
        const displayValue = value === "default_user_id" ? "Default Proxy Admin" : value;
        const width = info.cell.column.getSize();
        return (
          <Tooltip title={displayValue}>
            <span className={`font-mono text-xs truncate block`} style={{ maxWidth: width, overflow: "hidden" }}>
              {displayValue ?? "-"}
            </span>
          </Tooltip>
        );
      },
    },
    {
      id: "updated_at",
      accessorKey: "updated_at",
      header: "Updated At",
      size: 120,
      enableSorting: true,
      cell: (info) => {
        const value = info.getValue();
        return value ? new Date(value as string).toLocaleDateString() : "Never";
      },
    },
    {
      id: "last_active",
      accessorKey: "last_active",
      header: () => (
        <span className="flex items-center gap-1">
          Last Active
          <Popover
            content="This is a new field and is not backfilled. Only new key usage will update this value."
            trigger="hover"
          >
            <InfoCircleOutlined className="text-gray-400 text-xs cursor-help" />
          </Popover>
        </span>
      ),
      size: 130,
      enableSorting: false,
      cell: (info) => {
        const value = info.getValue();
        if (!value) return "Unknown";
        const date = new Date(value as string);
        return (
          <Tooltip title={date.toLocaleString(undefined, { dateStyle: "medium", timeStyle: "long" })}>
            <span>{date.toLocaleDateString()}</span>
          </Tooltip>
        );
      },
    },
    {
      id: "expires",
      accessorKey: "expires",
      header: "Expires",
      size: 120,
      enableSorting: false,
      cell: (info) => {
        const value = info.getValue();
        return value ? new Date(value as string).toLocaleDateString() : "Never";
      },
    },
    {
      id: "spend",
      accessorKey: "spend",
      header: "Spend (USD)",
      size: 100,
      enableSorting: true,
      cell: (info) => formatNumberWithCommas((info.getValue() as number) ?? 0, 4),
    },
    {
      id: "max_budget",
      accessorKey: "max_budget",
      header: "Budget (USD)",
      size: 110,
      enableSorting: true,
      cell: (info) => {
        const maxBudget = info.getValue() as number | null | undefined;
        if (maxBudget === null || maxBudget === undefined) {
          return "Unlimited";
        }
        return `$${formatNumberWithCommas(maxBudget)}`;
      },
    },
    {
      id: "budget_reset_at",
      accessorKey: "budget_reset_at",
      header: "Budget Reset",
      size: 130,
      enableSorting: false,
      cell: (info) => {
        const value = info.getValue();
        return value ? new Date(value as string).toLocaleString() : "Never";
      },
    },
    {
      id: "models",
      accessorKey: "models",
      header: "Models",
      size: 200,
      enableSorting: false,
      cell: (info) => {
        const models = info.getValue() as string[];
        return (
          <div className="flex flex-col py-2">
            {Array.isArray(models) ? (
              <div className="flex flex-col">
                {models.length === 0 ? (
                  <Badge size={"xs"} className="mb-1" color="red">
                    <Text>All Proxy Models</Text>
                  </Badge>
                ) : (
                  <>
                    <div className="flex items-start">
                      {models.length > 3 && (
                        <div>
                          <Icon
                            icon={expandedAccordions[info.row.id] ? ChevronDownIcon : ChevronRightIcon}
                            className="cursor-pointer"
                            size="xs"
                            onClick={() => {
                              setExpandedAccordions((prev) => ({
                                ...prev,
                                [info.row.id]: !prev[info.row.id],
                              }));
                            }}
                          />
                        </div>
                      )}
                      <div className="flex flex-wrap gap-1">
                        {models.slice(0, 3).map((model, index) =>
                          model === "all-proxy-models" ? (
                            <Badge key={index} size={"xs"} color="red">
                              <Text>All Proxy Models</Text>
                            </Badge>
                          ) : (
                            <Badge key={index} size={"xs"} color="blue">
                              <Text>
                                {model.length > 30
                                  ? `${getModelDisplayName(model).slice(0, 30)}...`
                                  : getModelDisplayName(model)}
                              </Text>
                            </Badge>
                          ),
                        )}
                        {models.length > 3 && !expandedAccordions[info.row.id] && (
                          <Badge size={"xs"} color="gray" className="cursor-pointer">
                            <Text>
                              +{models.length - 3} {models.length - 3 === 1 ? "more model" : "more models"}
                            </Text>
                          </Badge>
                        )}
                        {expandedAccordions[info.row.id] && (
                          <div className="flex flex-wrap gap-1">
                            {models.slice(3).map((model, index) =>
                              model === "all-proxy-models" ? (
                                <Badge key={index + 3} size={"xs"} color="red">
                                  <Text>All Proxy Models</Text>
                                </Badge>
                              ) : (
                                <Badge key={index + 3} size={"xs"} color="blue">
                                  <Text>
                                    {model.length > 30
                                      ? `${getModelDisplayName(model).slice(0, 30)}...`
                                      : getModelDisplayName(model)}
                                  </Text>
                                </Badge>
                              ),
                            )}
                          </div>
                        )}
                      </div>
                    </div>
                  </>
                )}
              </div>
            ) : null}
          </div>
        );
      },
    },
    {
      id: "rate_limits",
      header: "Rate Limits",
      size: 140,
      enableSorting: false,
      cell: ({ row }) => {
        const key = row.original as Record<string, unknown>;
        const tpm = key.tpm_limit;
        const rpm = key.rpm_limit;
        return (
          <div>
            <div>TPM: {tpm != null ? String(tpm) : "Unlimited"}</div>
            <div>RPM: {rpm != null ? String(rpm) : "Unlimited"}</div>
          </div>
        );
      },
    },
  ];
}
