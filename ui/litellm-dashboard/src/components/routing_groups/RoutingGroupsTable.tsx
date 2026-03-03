"use client";

import React from "react";
import { Badge } from "@tremor/react";
import { Table, TableHead, TableHeaderCell, TableBody, TableRow, TableCell } from "@tremor/react";
import { Button, Popconfirm, Tooltip } from "antd";
import { DeleteOutlined, EditOutlined } from "@ant-design/icons";

interface RoutingGroupRow {
  routing_group_id: string;
  routing_group_name: string;
  routing_strategy: string;
  deployments: unknown[];
  is_active: boolean;
  created_at: string;
  description?: string;
  [key: string]: unknown;
}

interface RoutingGroupsTableProps {
  data: RoutingGroupRow[];
  loading: boolean;
  onEdit: (group: Record<string, unknown>) => void;
  onDelete: (groupId: string) => void;
}

const strategyLabels: Record<string, string> = {
  "priority-failover": "Priority Failover",
  weighted: "Weighted",
  "cost-based-routing": "Cost-Based",
  "latency-based-routing": "Latency-Based",
  "least-busy": "Least Busy",
  "usage-based-routing-v2": "Usage-Based",
  "simple-shuffle": "Round Robin",
};

const strategyColors: Record<string, "blue" | "emerald" | "amber" | "cyan" | "violet" | "orange" | "indigo" | "gray"> = {
  "priority-failover": "blue",
  weighted: "emerald",
  "cost-based-routing": "amber",
  "latency-based-routing": "cyan",
  "least-busy": "violet",
  "usage-based-routing-v2": "orange",
  "simple-shuffle": "indigo",
};

export default function RoutingGroupsTable({
  data,
  loading,
  onEdit,
  onDelete,
}: RoutingGroupsTableProps) {
  return (
    <div className="rounded-lg custom-border relative">
      <div className="overflow-x-auto">
        <Table className="[&_td]:py-2 [&_th]:py-2" style={{ minWidth: "100%", tableLayout: "fixed" }}>
          <TableHead>
            <TableRow>
              <TableHeaderCell className="py-1 h-8" style={{ width: 200 }}>Name</TableHeaderCell>
              <TableHeaderCell className="py-1 h-8" style={{ width: 160 }}>Strategy</TableHeaderCell>
              <TableHeaderCell className="py-1 h-8" style={{ width: 120 }}>Deployments</TableHeaderCell>
              <TableHeaderCell className="py-1 h-8" style={{ width: 100 }}>Status</TableHeaderCell>
              <TableHeaderCell className="py-1 h-8" style={{ width: 140 }}>Created At</TableHeaderCell>
              <TableHeaderCell className="py-1 h-8" style={{ width: 100 }}></TableHeaderCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {loading ? (
              <TableRow>
                <TableCell colSpan={6} className="h-8 text-center">
                  <div className="text-center text-gray-500 py-4">Loading routing groups...</div>
                </TableCell>
              </TableRow>
            ) : data.length === 0 ? (
              <TableRow>
                <TableCell colSpan={6} className="h-8 text-center">
                  <div className="text-center text-gray-500 py-4">No routing groups yet. Create one to get started.</div>
                </TableCell>
              </TableRow>
            ) : (
              data.map((row) => {
                const strat = row.routing_strategy;
                const stratLabel = strategyLabels[strat] ?? strat;
                const stratColor = strategyColors[strat] ?? "gray";
                const deps = row.deployments;
                const depCount = Array.isArray(deps) ? deps.length : 0;
                const active = row.is_active !== false;
                const created = row.created_at;

                return (
                  <TableRow
                    key={row.routing_group_id}
                    className="cursor-pointer hover:bg-gray-50"
                    onClick={() => onEdit(row)}
                  >
                    <TableCell className="py-0.5" style={{ width: 200 }}>
                      <span className="text-sm font-semibold text-blue-600 hover:underline">
                        {row.routing_group_name}
                      </span>
                    </TableCell>
                    <TableCell className="py-0.5" style={{ width: 160 }}>
                      <Badge size="xs" color={stratColor}>{stratLabel}</Badge>
                    </TableCell>
                    <TableCell className="py-0.5" style={{ width: 120 }}>
                      <span className="text-sm text-gray-600">{depCount} model{depCount !== 1 ? "s" : ""}</span>
                    </TableCell>
                    <TableCell className="py-0.5" style={{ width: 100 }}>
                      <Badge size="xs" color={active ? "emerald" : "gray"}>
                        {active ? "Active" : "Inactive"}
                      </Badge>
                    </TableCell>
                    <TableCell className="py-0.5" style={{ width: 140 }}>
                      {created ? (
                        <span className="text-sm text-gray-500">{new Date(created).toLocaleDateString()}</span>
                      ) : (
                        <span className="text-sm text-gray-400">-</span>
                      )}
                    </TableCell>
                    <TableCell className="py-0.5" style={{ width: 100 }}>
                      <div className="flex items-center gap-2" onClick={(e) => e.stopPropagation()}>
                        <Tooltip title="Edit">
                          <Button
                            size="small"
                            type="text"
                            icon={<EditOutlined />}
                            onClick={() => onEdit(row)}
                            className="text-gray-500 hover:text-gray-700"
                          />
                        </Tooltip>
                        <Popconfirm
                          title="Delete routing group?"
                          description="This action cannot be undone."
                          onConfirm={() => onDelete(row.routing_group_id)}
                          okText="Delete"
                          okButtonProps={{ danger: true }}
                          cancelText="Cancel"
                        >
                          <Tooltip title="Delete">
                            <Button size="small" type="text" danger icon={<DeleteOutlined />} />
                          </Tooltip>
                        </Popconfirm>
                      </div>
                    </TableCell>
                  </TableRow>
                );
              })
            )}
          </TableBody>
        </Table>
      </div>
    </div>
  );
}
