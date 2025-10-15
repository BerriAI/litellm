import React, { useState } from "react";
import { BarChart } from "@tremor/react";
import KeyInfoView from "./templates/key_info_view";
import { keyInfoV1Call } from "./networking";
import { transformKeyInfo } from "../components/key_team_helpers/transform_key_info";
import { DataTable } from "./view_logs/table";
import { Tooltip } from "antd";
import { Button } from "@tremor/react";
import { formatNumberWithCommas } from "../utils/dataUtils";
import { TagUsage } from "./usage/types";
import { ChevronDownIcon, ChevronUpIcon } from "@heroicons/react/outline";

interface TopKeyViewProps {
  topKeys: any[];
  accessToken: string | null;
  userID: string | null;
  userRole: string | null;
  teams: any[] | null;
  premiumUser: boolean;
  showTags?: boolean;
}

const TopKeyView: React.FC<TopKeyViewProps> = ({
  topKeys,
  accessToken,
  userID,
  userRole,
  teams,
  premiumUser,
  showTags = false,
}) => {
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
      cell: (info: any) => (
        <div className="overflow-hidden">
          <Tooltip title={info.getValue() as string}>
            <Button
              size="xs"
              variant="light"
              className="font-mono text-blue-500 bg-blue-50 hover:bg-blue-100 text-xs font-normal px-2 py-0.5 text-left overflow-hidden truncate max-w-[200px]"
              onClick={() => handleKeyClick(info.row.original)}
            >
              {info.getValue() ? `${(info.getValue() as string).slice(0, 7)}...` : "-"}
            </Button>
          </Tooltip>
        </div>
      ),
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
              <Tooltip
                key={index}
                title={
                  <div>
                    <div>
                      <span className="text-gray-300">Tag Name:</span> {tag.tag}
                    </div>
                    <div>
                      <span className="text-gray-300">Spend:</span>{" "}
                      {tag.usage > 0 && tag.usage < 0.01 ? "<$0.01" : `$${formatNumberWithCommas(tag.usage, 2)}`}
                    </div>
                  </div>
                }
              >
                <span className="px-2 py-1 bg-gray-100 rounded-full text-xs">{tag.tag.slice(0, 7)}...</span>
              </Tooltip>
            ))}
            {hasMoreTags && (
              <button
                onClick={() => toggleTagsExpansion(apiKey)}
                className="ml-1 p-1 hover:bg-gray-200 rounded-full transition-colors"
                title={isExpanded ? "Show fewer tags" : "Show all tags"}
              >
                {isExpanded ? (
                  <ChevronUpIcon className="h-3 w-3 text-gray-500" />
                ) : (
                  <ChevronDownIcon className="h-3 w-3 text-gray-500" />
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
    cell: (info: any) => {
      const value = info.getValue();
      return value > 0 && value < 0.01 ? "<$0.01" : `$${formatNumberWithCommas(value, 2)}`;
    },
  };

  const columns = showTags ? [...baseColumns, tagsColumn, spendColumn] : [...baseColumns, spendColumn];

  const processedTopKeys = topKeys.map((k) => ({
    ...k,
    display_key_alias: k.key_alias && k.key_alias.length > 10 ? `${k.key_alias.slice(0, 10)}...` : k.key_alias || "-",
  }));

  return (
    <>
      <div className="mb-4 flex justify-end items-center">
        <div className="flex space-x-2">
          <button
            onClick={() => setViewMode("table")}
            className={`px-3 py-1 text-sm rounded-md ${viewMode === "table" ? "bg-blue-100 text-blue-700" : "bg-gray-100 text-gray-700"}`}
          >
            Table View
          </button>
          <button
            onClick={() => setViewMode("chart")}
            className={`px-3 py-1 text-sm rounded-md ${viewMode === "chart" ? "bg-blue-100 text-blue-700" : "bg-gray-100 text-gray-700"}`}
          >
            Chart View
          </button>
        </div>
      </div>

      {viewMode === "chart" ? (
        <div className="relative">
          <BarChart
            className="mt-4 h-40 cursor-pointer hover:opacity-90"
            data={processedTopKeys}
            index="display_key_alias"
            categories={["spend"]}
            colors={["cyan"]}
            yAxisWidth={120}
            tickGap={5}
            layout="vertical"
            showXAxis={false}
            showLegend={false}
            valueFormatter={(value) => (value ? `$${formatNumberWithCommas(value, 2)}` : "No Key Alias")}
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
        <div className="border rounded-lg overflow-hidden">
          <DataTable
            columns={columns}
            data={topKeys}
            renderSubComponent={() => <></>}
            getRowCanExpand={() => false}
            isLoading={false}
          />
        </div>
      )}

      {isModalOpen &&
        selectedKey &&
        keyData &&
        (console.log("Rendering modal with:", { isModalOpen, selectedKey, keyData }),
        (
          <div
            className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50"
            onClick={handleOutsideClick}
          >
            <div className="bg-white rounded-lg shadow-xl relative w-11/12 max-w-6xl max-h-[90vh] overflow-y-auto min-h-[750px]">
              {/* Close button */}
              <button
                onClick={handleClose}
                className="absolute top-4 right-4 text-gray-500 hover:text-gray-700 focus:outline-none"
                aria-label="Close"
              >
                <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>

              {/* Content */}
              <div className="p-6 h-full">
                <KeyInfoView
                  keyId={selectedKey}
                  onClose={handleClose}
                  keyData={keyData}
                  accessToken={accessToken}
                  userID={userID}
                  userRole={userRole}
                  teams={teams}
                  premiumUser={premiumUser}
                />
              </div>
            </div>
          </div>
        ))}
    </>
  );
};

export default TopKeyView;
