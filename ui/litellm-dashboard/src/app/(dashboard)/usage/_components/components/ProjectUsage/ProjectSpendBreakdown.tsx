import React from "react";
import type { ColumnDef } from "@tanstack/react-table";

import { DonutChart } from "@/components/shared/charts";
import { DataTable } from "@/components/shared/DataTable";
import { MoneyCell } from "@/components/shared/table_cells";
import { ChartLoader } from "@/components/shared/chart_loader";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { formatNumberWithCommas } from "@/utils/dataUtils";

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
  <Card className="h-full">
    <CardHeader>
      <CardTitle>Spend by Project</CardTitle>
    </CardHeader>
    <CardContent>
      {loading ? (
        <ChartLoader isDateChanging={isDateChanging} />
      ) : (
        <div className="grid grid-cols-2">
          <DonutChart
            className="mt-4 h-40"
            data={projectSpend}
            index="project_alias"
            category="spend"
            valueFormatter={(value) => `$${formatNumberWithCommas(value, 2)}`}
            colors={["cyan"]}
            showLabel
            startAngle={90}
            endAngle={-270}
          />
          <DataTable
            columns={columns}
            data={projectSpend}
            getRowId={(row) => row.project_id}
            noDataMessage="No project usage data"
            size="compact"
          />
        </div>
      )}
    </CardContent>
  </Card>
);

export default ProjectSpendBreakdown;
