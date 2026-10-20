import { BarChart } from "@/components/shared/charts";
import { DataTable } from "@/components/shared/DataTable";
import { MoneyCell } from "@/components/shared/table_cells";
import { ProviderLogo } from "@/components/molecules/models/ProviderLogo";
import { useState } from "react";
import { formatNumberWithCommas } from "@/utils/dataUtils";
import { Segmented } from "../overview/Primitives";
import { providerForModel } from "../overview/modelProvider";

type TopModel = {
  key: string;
  spend: number;
  successful_requests: number;
  failed_requests: number;
  tokens: number;
};

interface TopModelViewProps {
  topModels: TopModel[];
  topModelsLimit: number;
  setTopModelsLimit: (limit: number) => void;
}

const TOP_MODEL_LIMITS = [5, 10, 25, 50];

type ViewMode = "chart" | "table";

const LIMIT_OPTIONS = TOP_MODEL_LIMITS.map((limit) => ({ value: String(limit), label: String(limit) }));
const VIEW_OPTIONS = [
  { value: "table", label: "Table View" },
  { value: "chart", label: "Chart View" },
] as const satisfies readonly { value: ViewMode; label: string }[];

const BAR_COLOR = "#2b3fd6";
const QUIET_HEADER = { headerClassName: "font-normal" };

function ModelName({ name }: { name: string | undefined }) {
  if (!name) return <>-</>;
  const provider = providerForModel(name);
  return (
    <span className="flex min-w-0 items-center gap-2">
      {provider && (
        <span className="inline-flex size-6 shrink-0 items-center justify-center rounded-md border bg-background">
          <ProviderLogo provider={provider} className="size-3.5 rounded-[3px]" />
        </span>
      )}
      <span className="truncate font-medium text-foreground">{name}</span>
    </span>
  );
}

export default function TopModelView({ topModels, topModelsLimit, setTopModelsLimit }: TopModelViewProps) {
  const [modelViewMode, setModelViewMode] = useState<ViewMode>("table");

  const columns = [
    {
      header: "Model",
      accessorKey: "key",
      meta: QUIET_HEADER,
      cell: (info: any) => <ModelName name={info.getValue()} />,
    },
    {
      header: "Spend (USD)",
      accessorKey: "spend",
      meta: { numeric: true, ...QUIET_HEADER },
      cell: (info: any) => <MoneyCell value={info.getValue()} decimals={2} />,
    },
    {
      header: "Successful",
      accessorKey: "successful_requests",
      meta: { numeric: true, ...QUIET_HEADER },
      cell: (info: any) => <span className="text-success">{info.getValue()?.toLocaleString() || 0}</span>,
    },
    {
      header: "Failed",
      accessorKey: "failed_requests",
      meta: { numeric: true, ...QUIET_HEADER },
      cell: (info: any) => <span className="text-destructive">{info.getValue()?.toLocaleString() || 0}</span>,
    },
    {
      header: "Tokens",
      accessorKey: "tokens",
      meta: { numeric: true, ...QUIET_HEADER },
      cell: (info: any) => info.getValue()?.toLocaleString() || 0,
    },
  ];
  const processedTopModels = topModels.slice(0, topModelsLimit);

  return (
    <>
      <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
        <Segmented
          label="Number of models to show"
          value={String(topModelsLimit)}
          options={LIMIT_OPTIONS}
          onChange={(limit) => setTopModelsLimit(Number(limit))}
        />
        <Segmented
          label="Top model view mode"
          value={modelViewMode}
          options={VIEW_OPTIONS}
          onChange={setModelViewMode}
        />
      </div>
      {modelViewMode === "chart" ? (
        <div className="relative max-h-[600px] overflow-y-auto">
          <BarChart
            className="cursor-pointer hover:opacity-90"
            style={{ height: Math.min(processedTopModels.length, topModelsLimit) * 52 }}
            data={processedTopModels}
            index="key"
            categories={["spend"]}
            colors={[BAR_COLOR]}
            valueFormatter={(value) => `$${formatNumberWithCommas(value, 2)}`}
            layout="vertical"
            yAxisWidth={200}
            tickGap={5}
            showLegend={false}
          />
        </div>
      ) : (
        <DataTable columns={columns} data={processedTopModels} isLoading={false} maxBodyHeight={600} size="compact" />
      )}
    </>
  );
}
