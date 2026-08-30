import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { BarChart } from "@/components/shared/charts";
import { DataTable } from "@/components/shared/DataTable";
import { IdCell, MoneyCell } from "@/components/shared/table_cells";
import { ChevronDownIcon, ChevronUpIcon } from "@heroicons/react/outline";
import { Label } from "@/components/ui/label";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { SimpleTooltip } from "@/components/ui/tooltip";
import React, { useState } from "react";
import { formatNumberWithCommas } from "../../../../utils/dataUtils";
import { transformKeyInfo } from "../../../key_team_helpers/transform_key_info";
import { keyInfoV1Call } from "../../../networking";
import KeyInfoView from "../../../templates/key_info_view";
import { TagUsage } from "../../types";

const TOP_KEYS_LIMITS = [5, 10, 25, 50] as const;

interface TopKeyViewProps {
  topKeys: any[];
  teams: any[] | null;
  showTags?: boolean;
  topKeysLimit: number;
  setTopKeysLimit: (limit: number) => void;
}

const TopKeyView: React.FC<TopKeyViewProps> = ({ topKeys, teams, showTags = false, topKeysLimit, setTopKeysLimit }) => {
  const { accessToken } = useAuthorized();
  const [isModalOpen, setIsModalOpen] = useState<boolean>(false);
  const [selectedKey, setSelectedKey] = useState<string | null>(null);
  const [keyData, setKeyData] = useState<any | undefined>(undefined);
  const [viewMode, setViewMode] = useState<"chart" | "table">("table");
  const [expandedTags, setExpandedTags] = useState<Set<string>>(new Set());

  const toggleTagsExpansion = (apiKey: string) => {
    setExpandedTags((prev) => {
      const newSet = new Set(prev);
      if (newSet.has(apiKey)) {
        newSet.delete(apiKey);
      } else {
        newSet.add(apiKey);
      }
      return newSet;
    });
  };

  const handleKeyClick = async (item: any) => {
    if (!accessToken) return;

    try {
      const keyInfo = await keyInfoV1Call(accessToken, item.api_key);
      const transformedKeyData = transformKeyInfo(keyInfo);

      setKeyData(transformedKeyData);
      setSelectedKey(item.api_key);
      setIsModalOpen(true); // Open modal when key is clicked
    } catch (error) {
      console.error("Error fetching key info:", error);
    }
  };

  const handleClose = () => {
    setIsModalOpen(false);
    setSelectedKey(null);
    setKeyData(undefined);
  };

  // Handle clicking outside the modal
  const handleOutsideClick = (e: React.MouseEvent<HTMLDivElement>) => {
    if (e.target === e.currentTarget) {
      handleClose();
    }
  };

  // Handle escape key
  React.useEffect(() => {
    const handleEscapeKey = (e: KeyboardEvent) => {
      if (e.key === "Escape" && isModalOpen) {
        handleClose();
      }
    };

    document.addEventListener("keydown", handleEscapeKey);
    return () => document.removeEventListener("keydown", handleEscapeKey);
  }, [isModalOpen]);

  // Define columns for the table view
  const baseColumns = [
    {
      header: "Key ID",
      accessorKey: "api_key",
      cell: (info: any) => <IdCell value={info.getValue()} onClick={() => handleKeyClick(info.row.original)} />,
    },
    {
      header: "Key Alias",
      accessorKey: "key_alias",
      cell: (info: any) => info.getValue() || "-",
    },
  ];

  const tagsColumn = {
    header: "Tags",
    accessorKey: "tags",
    cell: (info: any) => {
      const tags = info.getValue() as TagUsage[] | undefined;
      const apiKey = info.row.original.api_key;
      const isExpanded = expandedTags.has(apiKey);

      if (!tags || tags.length === 0) {
        return "-";
      }

      const sortedTags = tags.sort((a, b) => b.usage - a.usage);
      const displayTags = isExpanded ? sortedTags : sortedTags.slice(0, 2);
      const hasMoreTags = tags.length > 2;

      return (
        <div className="overflow-hidden">
          <div className="flex flex-wrap items-center gap-1">
            {displayTags.map((tag, index) => (
              <SimpleTooltip
                key={index}
                content={
                  <div>
                    <div>
                      <span className="text-muted-foreground">Tag Name:</span> {tag.tag}
                    </div>
                    <div>
                      <span className="text-muted-foreground">Spend:</span>{" "}
                      {tag.usage > 0 && tag.usage < 0.01 ? "<$0.01" : `$${formatNumberWithCommas(tag.usage, 2)}`}
                    </div>
                  </div>
                }
              >
                <span className="px-2 py-1 bg-muted rounded-full text-xs">{tag.tag.slice(0, 7)}...</span>
              </SimpleTooltip>
            ))}
            {hasMoreTags && (
              <button
                onClick={() => toggleTagsExpansion(apiKey)}
                className="ml-1 p-1 hover:bg-accent rounded-full transition-colors"
                title={isExpanded ? "Show fewer tags" : "Show all tags"}
              >
                {isExpanded ? (
                  <ChevronUpIcon className="h-3 w-3 text-muted-foreground" />
                ) : (
                  <ChevronDownIcon className="h-3 w-3 text-muted-foreground" />
                )}
              </button>
            )}
          </div>
        </div>
      );
    },
  };

  const spendColumn = {
    header: "Spend (USD)",
    accessorKey: "spend",
    meta: { numeric: true },
    cell: (info: any) => <MoneyCell value={info.getValue()} decimals={2} />,
  };

  const columns = showTags ? [...baseColumns, tagsColumn, spendColumn] : [...baseColumns, spendColumn];

  const processedTopKeys = topKeys.map((k) => ({
    ...k,
    display_key_alias: k.key_alias && k.key_alias.length > 10 ? `${k.key_alias.slice(0, 10)}...` : k.key_alias || "-",
  }));

  return (
    <>
      <div className="mb-4 flex justify-between items-center">
        <RadioGroup
          aria-label="Number of top keys to show"
          value={String(topKeysLimit)}
          onValueChange={(limit: unknown) => setTopKeysLimit(Number(limit))}
          className="inline-flex w-fit items-center gap-1 rounded-lg bg-muted p-[3px]"
        >
          {TOP_KEYS_LIMITS.map((limit) => (
            <Label
              key={limit}
              className="cursor-pointer rounded-md px-3 py-1 font-medium text-foreground/60 transition-colors has-data-checked:bg-background has-data-checked:text-foreground has-data-checked:shadow-sm"
            >
              <RadioGroupItem value={String(limit)} className="sr-only" />
              {limit}
            </Label>
          ))}
        </RadioGroup>
        <div className="flex space-x-2">
          <button
            onClick={() => setViewMode("table")}
            className={`px-3 py-1 text-sm rounded-md ${viewMode === "table" ? "bg-info/15 text-info" : "bg-muted text-foreground"}`}
          >
            Table View
          </button>
          <button
            onClick={() => setViewMode("chart")}
            className={`px-3 py-1 text-sm rounded-md ${viewMode === "chart" ? "bg-info/15 text-info" : "bg-muted text-foreground"}`}
          >
            Chart View
          </button>
        </div>
      </div>

      {viewMode === "chart" ? (
        <div className="relative max-h-[600px] overflow-y-auto">
          <BarChart
            className="mt-4 cursor-pointer hover:opacity-90"
            style={{ height: Math.min(processedTopKeys.length, topKeysLimit) * 52 }}
            data={processedTopKeys}
            index="display_key_alias"
            categories={["spend"]}
            colors={["cyan"]}
            yAxisWidth={120}
            tickGap={5}
            layout="vertical"
            showLegend={false}
            valueFormatter={(value) => `$${formatNumberWithCommas(value, 2)}`}
            onValueChange={(item) => handleKeyClick(item)}
            showTooltip={true}
            customTooltip={(props) => {
              const item = props.payload?.[0]?.payload;
              return (
                <div className="relative z-floating p-3 bg-black/90 shadow-lg rounded-lg text-white max-w-xs">
                  <div className="space-y-1.5">
                    <div className="text-sm">
                      <span className="text-muted-foreground">Key Alias: </span>
                      <span className="font-mono text-gray-100 break-all">{item?.key_alias}</span>
                    </div>
                    <div className="text-sm">
                      <span className="text-muted-foreground">Key ID: </span>
                      <span className="font-mono text-gray-100 break-all">{item?.api_key}</span>
                    </div>
                    <div className="text-sm">
                      <span className="text-muted-foreground">Spend: </span>
                      <span className="text-white font-medium">${formatNumberWithCommas(item?.spend, 2)}</span>
                    </div>
                  </div>
                </div>
              );
            }}
          />
        </div>
      ) : (
        <DataTable columns={columns} data={topKeys} isLoading={false} maxBodyHeight={600} size="compact" />
      )}

      {isModalOpen && selectedKey && keyData && (
        <div
          className="fixed inset-0 bg-black/50 flex items-center justify-center z-overlay"
          onClick={handleOutsideClick}
        >
          <div className="bg-card rounded-lg shadow-xl relative w-11/12 max-w-6xl max-h-[90vh] overflow-y-auto min-h-[750px]">
            {/* Close button */}
            <button
              onClick={handleClose}
              className="absolute top-4 right-4 text-muted-foreground hover:text-foreground focus:outline-hidden"
              aria-label="Close"
            >
              <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>

            {/* Content */}
            <div className="p-6 h-full">
              <KeyInfoView keyId={selectedKey} onClose={handleClose} keyData={keyData} teams={teams} />
            </div>
          </div>
        </div>
      )}
    </>
  );
};

export default TopKeyView;
