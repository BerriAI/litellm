import React from "react";
import type { ColumnDef } from "@tanstack/react-table";

import { MoneyCell } from "@/components/shared/table_cells";

import { SpendByCategoryPanel } from "../SpendByCategoryPanel";
import type { ProjectSpendRow } from "./projectUsageAggregations";

interface ProjectSpendBreakdownProps {
  loading: boolean;
  isDateChanging: boolean;
  projectSpend: ProjectSpendRow[];
}

const columns: ColumnDef<ProjectSpendRow>[] = [
  {
    header: "Project",
    accessorKey: "project_alias",
  },
  {
    header: "Spend",
    accessorKey: "spend",
    meta: { numeric: true },
    cell: ({ row }) => <MoneyCell value={row.original.spend} decimals={2} />,
  },
  {
    header: "Successful",
    accessorKey: "successful_requests",
    meta: { numeric: true, className: "text-success" },
    cell: ({ row }) => row.original.successful_requests.toLocaleString(),
  },
  {
    header: "Failed",
    accessorKey: "failed_requests",
    meta: { numeric: true, className: "text-destructive" },
    cell: ({ row }) => row.original.failed_requests.toLocaleString(),
  },
  {
    header: "Tokens",
    accessorKey: "tokens",
    meta: { numeric: true },
    cell: ({ row }) => row.original.tokens.toLocaleString(),
  },
];

const ProjectSpendBreakdown: React.FC<ProjectSpendBreakdownProps> = ({ loading, isDateChanging, projectSpend }) => (
  <SpendByCategoryPanel
    title="Spend by Project"
    loading={loading}
    isDateChanging={isDateChanging}
    data={projectSpend}
    indexKey="project_alias"
    columns={columns}
    getRowId={(row) => row.project_id}
    noDataMessage="No project usage data"
  />
);

export default ProjectSpendBreakdown;
