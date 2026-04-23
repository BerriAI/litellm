import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { ChevronDown, ChevronUp, X } from "lucide-react";
// eslint-disable-next-line litellm-ui/no-banned-ui-imports
import { BarChart } from "@tremor/react";
import { Segmented } from "antd";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";
import React, { useState } from "react";
import { formatNumberWithCommas } from "../../../../utils/dataUtils";
import { transformKeyInfo } from "../../../key_team_helpers/transform_key_info";
import { keyInfoV1Call } from "../../../networking";
import KeyInfoView from "../../../templates/key_info_view";
import { DataTable } from "../../../view_logs/table";
import { TagUsage } from "../../types";

interface TopKeyViewProps {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  topKeys: any[];
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  teams: any[] | null;
  showTags?: boolean;
  topKeysLimit: number;
  setTopKeysLimit: (limit: number) => void;
}

const TopKeyView: React.FC<TopKeyViewProps> = ({ topKeys, teams, showTags = false, topKeysLimit, setTopKeysLimit }) => {
  const { accessToken, userRole, userId: userID, premiumUser } = useAuthorized();
  const [isModalOpen, setIsModalOpen] = useState<boolean>(false);
  const [selectedKey, setSelectedKey] = useState<string | null>(null);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
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

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
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
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      cell: (info: any) => (
        <div className="overflow-hidden">
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <button
                  type="button"
                  className="font-mono text-blue-500 bg-blue-50 hover:bg-blue-100 dark:bg-blue-950/30 dark:hover:bg-blue-950/60 text-xs font-normal px-2 py-0.5 text-left overflow-hidden truncate max-w-[200px] rounded"
                  onClick={() => handleKeyClick(info.row.original)}
                >
                  {info.getValue()
                    ? `${(info.getValue() as string).slice(0, 7)}...`
                    : "-"}
                </button>
              </TooltipTrigger>
              <TooltipContent>{info.getValue() as string}</TooltipContent>
            </Tooltip>
          </TooltipProvider>
        </div>
      ),
    },
    {
      header: "Key Alias",
      accessorKey: "key_alias",
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      cell: (info: any) => info.getValue() || "-",
    },
  ];

  const tagsColumn = {
    header: "Tags",
    accessorKey: "tags",
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
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
              <TooltipProvider key={index}>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <span className="px-2 py-1 bg-muted rounded-full text-xs">
                      {tag.tag.slice(0, 7)}...
                    </span>
                  </TooltipTrigger>
                  <TooltipContent>
                    <div>
                      <div>
                        <span className="text-muted-foreground">
                          Tag Name:
                        </span>{" "}
                        {tag.tag}
                      </div>
                      <div>
                        <span className="text-muted-foreground">Spend:</span>{" "}
                        {tag.usage > 0 && tag.usage < 0.01
                          ? "<$0.01"
                          : `$${formatNumberWithCommas(tag.usage, 2)}`}
                      </div>
                    </div>
                  </TooltipContent>
                </Tooltip>
              </TooltipProvider>
            ))}
            {hasMoreTags && (
              <button
                type="button"
                onClick={() => toggleTagsExpansion(apiKey)}
                className="ml-1 p-1 hover:bg-muted rounded-full transition-colors"
                title={isExpanded ? "Show fewer tags" : "Show all tags"}
              >
                {isExpanded ? (
                  <ChevronUp className="h-3 w-3 text-muted-foreground" />
                ) : (
                  <ChevronDown className="h-3 w-3 text-muted-foreground" />
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
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    cell: (info: any) => {
      const value = info.getValue();
      return value > 0 && value < 0.01
        ? "<$0.01"
        : `$${formatNumberWithCommas(value, 2)}`;
    },
  };

  const columns = showTags ? [...baseColumns, tagsColumn, spendColumn] : [...baseColumns, spendColumn];

  const processedTopKeys = topKeys.map((k) => ({
    ...k,
    display_key_alias: k.key_alias && k.key_alias.length > 10 ? `${k.key_alias.slice(0, 10)}...` : k.key_alias || "-",
  }));

  return (
    <>
      <div className="mb-4 flex justify-between items-center">
        <Segmented
          options={[
            { label: "5", value: 5 },
            { label: "10", value: 10 },
            { label: "25", value: 25 },
            { label: "50", value: 50 },
          ]}
          value={topKeysLimit}
          onChange={(value) => setTopKeysLimit(value as number)}
        />
        <div className="flex space-x-2">
          <button
            type="button"
            onClick={() => setViewMode("table")}
            className={cn(
              "px-3 py-1 text-sm rounded-md",
              viewMode === "table"
                ? "bg-blue-100 text-blue-700 dark:bg-blue-950 dark:text-blue-300"
                : "bg-muted text-muted-foreground",
            )}
          >
            Table View
          </button>
          <button
            type="button"
            onClick={() => setViewMode("chart")}
            className={cn(
              "px-3 py-1 text-sm rounded-md",
              viewMode === "chart"
                ? "bg-blue-100 text-blue-700 dark:bg-blue-950 dark:text-blue-300"
                : "bg-muted text-muted-foreground",
            )}
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
                <div className="relative z-50 p-3 bg-black/90 shadow-lg rounded-lg text-white max-w-xs">
                  <div className="space-y-1.5">
                    <div className="text-sm">
                      <span className="text-gray-300">Key Alias: </span>
                      <span className="font-mono text-gray-100 break-all">{item?.key_alias}</span>
                    </div>
                    <div className="text-sm">
                      <span className="text-gray-300">Key ID: </span>
                      <span className="font-mono text-gray-100 break-all">{item?.api_key}</span>
                    </div>
                    <div className="text-sm">
                      <span className="text-gray-300">Spend: </span>
                      <span className="text-white font-medium">${formatNumberWithCommas(item?.spend, 2)}</span>
                    </div>
                  </div>
                </div>
              );
            }}
          />
        </div>
      ) : (
        <div className="border border-border rounded-lg overflow-hidden max-h-[600px] overflow-y-auto">
          <DataTable
            columns={columns}
            data={topKeys}
            renderSubComponent={() => <></>}
            getRowCanExpand={() => false}
            isLoading={false}
          />
        </div>
      )}

      {isModalOpen && selectedKey && keyData && (
        <div
          className="fixed inset-0 bg-black/50 flex items-center justify-center z-50"
          onClick={handleOutsideClick}
        >
          <div className="bg-background rounded-lg shadow-xl relative w-11/12 max-w-6xl max-h-[90vh] overflow-y-auto min-h-[750px]">
            <button
              type="button"
              onClick={handleClose}
              className="absolute top-4 right-4 text-muted-foreground hover:text-foreground focus:outline-none"
              aria-label="Close"
            >
              <X className="w-6 h-6" />
            </button>

            <div className="p-6 h-full">
              <KeyInfoView
                keyId={selectedKey}
                onClose={handleClose}
                keyData={keyData}
                teams={teams}
              />
            </div>
          </div>
        </div>
      )}
    </>
  );
};

export default TopKeyView;
