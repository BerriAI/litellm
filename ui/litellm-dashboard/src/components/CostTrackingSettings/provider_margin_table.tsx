import React, { useState } from "react";
import { TextInput, Icon, Text } from "@tremor/react";
import { TrashIcon, PencilAltIcon, CheckIcon, XIcon } from "@heroicons/react/outline";
import { SimpleTable } from "../common_components/simple_table";
import { MarginConfig } from "./types";
import { getProviderDisplayInfo, handleImageError } from "./provider_display_helpers";

interface ProviderMarginTableProps {
  marginConfig: MarginConfig;
  onMarginChange: (provider: string, value: number | { percentage?: number; fixed_amount?: number }) => void;
  onRemoveProvider: (provider: string, providerDisplayName: string) => void;
}

interface ProviderMarginRow {
  provider: string;
  margin: number | { percentage?: number; fixed_amount?: number };
}

const ProviderMarginTable: React.FC<ProviderMarginTableProps> = ({
  marginConfig,
  onMarginChange,
  onRemoveProvider,
}) => {
  const [editingProvider, setEditingProvider] = useState<string | null>(null);
  const [editPercentage, setEditPercentage] = useState<string>("");
  const [editFixedAmount, setEditFixedAmount] = useState<string>("");

  const handleStartEdit = (provider: string, currentMargin: number | { percentage?: number; fixed_amount?: number }) => {
    setEditingProvider(provider);
    if (typeof currentMargin === "number") {
      // Simple percentage format
      setEditPercentage((currentMargin * 100).toString());
      setEditFixedAmount("");
    } else {
      // Complex format with percentage and/or fixed_amount
      setEditPercentage(currentMargin.percentage ? (currentMargin.percentage * 100).toString() : "");
      setEditFixedAmount(currentMargin.fixed_amount ? currentMargin.fixed_amount.toString() : "");
    }
  };

  const handleSaveEdit = (provider: string) => {
    const percentValue = editPercentage ? parseFloat(editPercentage) : undefined;
    const fixedValue = editFixedAmount ? parseFloat(editFixedAmount) : undefined;

    if (percentValue !== undefined && !isNaN(percentValue) && percentValue >= 0 && percentValue <= 1000) {
      if (fixedValue !== undefined && !isNaN(fixedValue) && fixedValue >= 0) {
        // Both percentage and fixed amount
        onMarginChange(provider, { percentage: percentValue / 100, fixed_amount: fixedValue });
      } else {
        // Only percentage
        onMarginChange(provider, percentValue / 100);
      }
    } else if (fixedValue !== undefined && !isNaN(fixedValue) && fixedValue >= 0) {
      // Only fixed amount
      onMarginChange(provider, { fixed_amount: fixedValue });
    }
    setEditingProvider(null);
    setEditPercentage("");
    setEditFixedAmount("");
  };

  const handleCancelEdit = () => {
    setEditingProvider(null);
    setEditPercentage("");
    setEditFixedAmount("");
  };

  const handleKeyDown = (e: React.KeyboardEvent, provider: string) => {
    if (e.key === 'Enter') {
      handleSaveEdit(provider);
    } else if (e.key === 'Escape') {
      handleCancelEdit();
    }
  };

  const formatMargin = (margin: number | { percentage?: number; fixed_amount?: number }): string => {
    if (typeof margin === "number") {
      return `${(margin * 100).toFixed(1)}%`;
    }
    const parts: string[] = [];
    if (margin.percentage !== undefined) {
      parts.push(`${(margin.percentage * 100).toFixed(1)}%`);
    }
    if (margin.fixed_amount !== undefined) {
      parts.push(`$${margin.fixed_amount.toFixed(6)}`);
    }
    return parts.join(" + ") || "0%";
  };

  // Convert margin config to array and sort (global first, then alphabetically)
  const data: ProviderMarginRow[] = Object.entries(marginConfig)
    .map(([provider, margin]) => ({ provider, margin }))
    .sort((a, b) => {
      if (a.provider === "global") return -1;
      if (b.provider === "global") return 1;
      const displayA = getProviderDisplayInfo(a.provider).displayName;
      const displayB = getProviderDisplayInfo(b.provider).displayName;
      return displayA.localeCompare(displayB);
    });

  return (
    <SimpleTable
      data={data}
      columns={[
        {
          header: "Provider",
          cell: (row) => {
            if (row.provider === "global") {
              return (
                <div className="flex items-center space-x-2">
                  <span className="font-medium">Global (All Providers)</span>
                </div>
              );
            }
            const { displayName, logo } = getProviderDisplayInfo(row.provider);
            return (
              <div className="flex items-center space-x-2">
                {logo && (
                  <img
                    src={logo}
                    alt={`${displayName} logo`}
                    className="w-5 h-5"
                    onError={(e) => handleImageError(e, displayName)}
                  />
                )}
                <span className="font-medium">{displayName}</span>
              </div>
            );
          },
        },
        {
          header: "Margin",
          cell: (row) => (
            <div className="flex items-center gap-2">
              {editingProvider === row.provider ? (
                <>
                  <div className="flex items-center gap-2">
                    <TextInput
                      value={editPercentage}
                      onValueChange={setEditPercentage}
                      placeholder="10"
                      className="w-20"
                      autoFocus
                    />
                    <span className="text-gray-600">%</span>
                    <span className="text-gray-400">+</span>
                    <span className="text-gray-600">$</span>
                    <TextInput
                      value={editFixedAmount}
                      onValueChange={setEditFixedAmount}
                      placeholder="0.001"
                      className="w-24"
                    />
                  </div>
                  <Icon
                    icon={CheckIcon}
                    size="sm"
                    onClick={() => handleSaveEdit(row.provider)}
                    className="cursor-pointer text-green-600 hover:text-green-700"
                  />
                  <Icon
                    icon={XIcon}
                    size="sm"
                    onClick={handleCancelEdit}
                    className="cursor-pointer text-gray-600 hover:text-gray-700"
                  />
                </>
              ) : (
                <>
                  <Text className="font-medium">{formatMargin(row.margin)}</Text>
                  <Icon
                    icon={PencilAltIcon}
                    size="sm"
                    onClick={() => handleStartEdit(row.provider, row.margin)}
                    className="cursor-pointer text-blue-600 hover:text-blue-700"
                  />
                </>
              )}
            </div>
          ),
          width: "350px",
        },
        {
          header: "Actions",
          cell: (row) => {
            const displayName = row.provider === "global" ? "Global" : getProviderDisplayInfo(row.provider).displayName;
            return (
              <Icon
                icon={TrashIcon}
                size="sm"
                onClick={() => onRemoveProvider(row.provider, displayName)}
                className="cursor-pointer hover:text-red-600"
              />
            );
          },
          width: "80px",
        },
      ]}
      getRowKey={(row) => row.provider}
      emptyMessage="No provider margins configured"
    />
  );
};

export default ProviderMarginTable;

