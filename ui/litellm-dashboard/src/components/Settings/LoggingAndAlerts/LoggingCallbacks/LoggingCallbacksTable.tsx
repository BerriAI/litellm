import { Button } from "@tremor/react";
import type { TableProps } from "antd";
import { Table, Tag, Tooltip } from "antd";
import Title from "antd/es/typography/Title";
import React from "react";
import { StatusBadge, type StatusTone } from "@/components/shared/table_cells";
import TableIconActionButton from "../../../common_components/IconActionButton/TableIconActionButtons/TableIconActionButton";
import { AlertingObject } from "./types";

type LoggingCallbacksProps = {
  callbacks: AlertingObject[];
  availableCallbacks?: Record<
    string,
    {
      litellm_callback_name: string;
      litellm_callback_params: string[];
      ui_callback_name: string;
    }
  >;
  onTest?: (callback: AlertingObject) => void | Promise<void>;
  onEdit?: (callback: AlertingObject) => void;
  onDelete?: (callback: AlertingObject) => void;
  onEditAccess?: (callback: AlertingObject) => void;
  onAdd?: () => void;
};

const isDestination = (record: AlertingObject): boolean => record.credentialName != null;

const SCOPE_BADGES_LIMIT = 4;

// Renders the explicit access grants for a destination.
//
// The Scope column shows only what is statically configured in credential_info.access.
// It does NOT reflect runtime enablement behavior (auto_enable) — that is the
// Mode column's job. This keeps the two concepts cleanly separated:
//   - access.global=true  → "Global access"  (visible/assignable by all)
//   - access.teams=[...]  → per-team badges
//   - access.orgs=[...]   → per-org badges
//   - empty/absent access → "—" in all cases, including auto_enable=true
const ScopeCell: React.FC<{ record: AlertingObject }> = ({ record }) => {
  const scope = record.resolvedScope;

  if (!scope || (!scope.global && scope.teams.length === 0 && scope.orgs.length === 0)) {
    return <span className="text-gray-400">—</span>;
  }
  if (scope.global) {
    return <Tag color="blue">Global access</Tag>;
  }
  const items = [
    ...scope.teams.map((label) => ({ kind: "team" as const, label })),
    ...scope.orgs.map((label) => ({ kind: "org" as const, label })),
  ];
  const shown = items.slice(0, SCOPE_BADGES_LIMIT);
  const remainder = items.length - shown.length;
  return (
    <div className="flex flex-wrap gap-1">
      {shown.map((item, i) => (
        <Tag key={`${item.kind}-${item.label}-${i}`} color={item.kind === "team" ? "blue" : "geekblue"}>
          {item.kind}: {item.label}
        </Tag>
      ))}
      {remainder > 0 && <Tag>+{remainder} more</Tag>}
    </div>
  );
};

type CallbackRow = AlertingObject & {
  id?: string;
  mode?: "success" | "failure" | "info" | string;
};

const CALLBACK_MODES: { value: string; label: string }[] = [
  { value: "success", label: "Success" },
  { value: "failure", label: "Failure" },
  { value: "success_and_failure", label: "Success & Failure" },
];

export const LoggingCallbacksTable: React.FC<LoggingCallbacksProps> = ({
  callbacks,
  availableCallbacks = {},
  onTest = () => {},
  onEdit = () => {},
  onDelete = () => {},
  onEditAccess = () => {},
  onAdd = () => {},
}) => {
  const columns: TableProps<CallbackRow>["columns"] = [
    {
      title: <span className="font-medium text-gray-700">Callback Name</span>,
      dataIndex: "name",
      key: "name",
      render: (_: string, record: CallbackRow) => {
        const id = record.name;
        const displayName = availableCallbacks[id]?.ui_callback_name || id;
        return (
          <div>
            <div className="font-medium text-gray-800">{displayName}</div>
            {record.destinationLabel && <div className="text-xs text-gray-500">{record.destinationLabel}</div>}
          </div>
        );
      },
    },
    {
      title: <span className="font-medium text-gray-700">Mode</span>,
      key: "mode",
      render: (_: unknown, record: CallbackRow) => {
        // Destination rows show their enablement behaviour: auto_enable=true
        // means the destination exports automatically (scoped by access grants);
        // false means it only exports when explicitly named in logging_exporters.
        if (isDestination(record)) {
          if (record.autoEnable === true) {
            const access = record.access;
            const hasExplicitGrants =
              access?.global === true ||
              (Array.isArray(access?.teams) && access.teams.length > 0) ||
              (Array.isArray(access?.orgs) && access.orgs.length > 0);
            const tooltipTitle = hasExplicitGrants
              ? "Exports automatically for all identities within the access scope without requiring explicit assignment."
              : "No explicit access grants. Treated as proxy-wide automatic export for backward compatibility. Add access.global=true or access.teams/orgs to scope this destination.";
            return (
              <Tooltip title={tooltipTitle}>
                <Tag color="orange" style={{ cursor: "help" }}>
                  Auto-enabled
                </Tag>
              </Tooltip>
            );
          }
          return <span className="text-gray-400 text-xs">Manual assignment</span>;
        }
        // Backend sends `type` (success | failure); legacy in-memory rows
        // from add-callback flow set `mode`. Read both so newly-added rows
        // and server-fetched rows both render correctly.
        const mode = record.type || record.mode || "success";
        const label = CALLBACK_MODES.find((m) => m.value === mode)?.label || mode;
        const tone: StatusTone = mode === "success" ? "success" : mode === "failure" ? "error" : "info";
        return <StatusBadge tone={tone} label={label} />;
      },
      width: 200,
    },
    {
      title: <span className="font-medium text-gray-700">Scope</span>,
      key: "access",
      render: (_: unknown, record: CallbackRow) =>
        isDestination(record) ? <ScopeCell record={record} /> : <span className="text-gray-400">—</span>,
      width: 280,
    },
    {
      title: <span className="font-medium text-gray-700 text-right w-full block">Actions</span>,
      key: "actions",
      align: "right",
      render: (_: unknown, record: CallbackRow) =>
        isDestination(record) ? (
          <div className="flex justify-end gap-2">
            <TableIconActionButton
              variant="Edit"
              tooltipText="Edit scope"
              dataTestId="edit-access"
              onClick={() => onEditAccess(record)}
            />
            <TableIconActionButton
              variant="Delete"
              tooltipText="Delete destination"
              dataTestId="delete-destination"
              onClick={() => onDelete(record)}
            />
          </div>
        ) : (
          <div className="flex justify-end gap-2">
            <TableIconActionButton
              variant="Test"
              tooltipText="Test Callback"
              dataTestId="test-callback"
              onClick={() => onTest(record)}
            />
            <TableIconActionButton
              variant="Edit"
              tooltipText="Edit Callback"
              dataTestId="edit-callback"
              onClick={() => onEdit(record)}
            />
            <TableIconActionButton
              variant="Delete"
              tooltipText="Delete Callback"
              dataTestId="delete-callback"
              onClick={() => onDelete(record)}
            />
          </div>
        ),
      width: 200,
    },
  ];
  return (
    <>
      <div className="w-full mt-4">
        <Button onClick={onAdd} className="mx-auto">
          + Add Callback
        </Button>
        <div className="flex justify-between items-center my-2">
          <Title level={4}>Active Logging Callbacks</Title>
        </div>
        {/* Empty state */}
        {callbacks.length === 0 ? (
          <div className="flex flex-col items-center justify-center p-8 bg-gray-50 border border-gray-200 rounded-lg">
            <div className="text-center">
              <h3 className="text-lg font-medium text-gray-700 mb-2">No callbacks configured</h3>
              <p className="text-gray-500">Add your first callback to start logging data to external services.</p>
            </div>
          </div>
        ) : (
          <div className="bg-white border border-gray-200 rounded-lg overflow-hidden">
            <Table
              columns={columns}
              dataSource={callbacks as CallbackRow[]}
              // `generic_api` can appear as both a success and a failure
              // callback simultaneously — keying by `name` alone produced
              // duplicate React keys. Compose with type to keep keys unique.
              rowKey={(record) => `${record.name}-${record.type || record.mode || "success"}`}
              pagination={false}
              rowClassName={() => "hover:bg-gray-50"}
            />
          </div>
        )}
      </div>
    </>
  );
};
