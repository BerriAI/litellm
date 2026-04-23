import React, { useState } from "react";
import { Input } from "@/components/ui/input";
import { Check, Pencil, Trash2, X } from "lucide-react";
import { SimpleTable } from "../common_components/simple_table";
import { MarginConfig } from "./types";
import {
  getProviderDisplayInfo,
  handleImageError,
} from "./provider_display_helpers";

interface ProviderMarginTableProps {
  marginConfig: MarginConfig;
  onMarginChange: (
    provider: string,
    value: number | { percentage?: number; fixed_amount?: number },
  ) => void;
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

  const handleStartEdit = (
    provider: string,
    currentMargin: number | { percentage?: number; fixed_amount?: number },
  ) => {
    setEditingProvider(provider);
    if (typeof currentMargin === "number") {
      setEditPercentage((currentMargin * 100).toString());
      setEditFixedAmount("");
    } else {
      setEditPercentage(
        currentMargin.percentage
          ? (currentMargin.percentage * 100).toString()
          : "",
      );
      setEditFixedAmount(
        currentMargin.fixed_amount ? currentMargin.fixed_amount.toString() : "",
      );
    }
  };

  const handleSaveEdit = (provider: string) => {
    const percentValue = editPercentage ? parseFloat(editPercentage) : undefined;
    const fixedValue = editFixedAmount ? parseFloat(editFixedAmount) : undefined;

    if (
      percentValue !== undefined &&
      !isNaN(percentValue) &&
      percentValue >= 0 &&
      percentValue <= 1000
    ) {
      if (
        fixedValue !== undefined &&
        !isNaN(fixedValue) &&
        fixedValue >= 0
      ) {
        onMarginChange(provider, {
          percentage: percentValue / 100,
          fixed_amount: fixedValue,
        });
      } else {
        onMarginChange(provider, percentValue / 100);
      }
    } else if (
      fixedValue !== undefined &&
      !isNaN(fixedValue) &&
      fixedValue >= 0
    ) {
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

  const formatMargin = (
    margin: number | { percentage?: number; fixed_amount?: number },
  ): string => {
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
                  // eslint-disable-next-line @next/next/no-img-element
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
                    <Input
                      value={editPercentage}
                      onChange={(e) => setEditPercentage(e.target.value)}
                      placeholder="10"
                      className="w-20 h-8"
                      autoFocus
                    />
                    <span className="text-muted-foreground">%</span>
                    <span className="text-muted-foreground">+</span>
                    <span className="text-muted-foreground">$</span>
                    <Input
                      value={editFixedAmount}
                      onChange={(e) => setEditFixedAmount(e.target.value)}
                      placeholder="0.001"
                      className="w-24 h-8"
                    />
                  </div>
                  <button
                    type="button"
                    onClick={() => handleSaveEdit(row.provider)}
                    className="cursor-pointer text-emerald-600 hover:text-emerald-700 dark:text-emerald-400"
                    aria-label="Save"
                  >
                    <Check className="h-4 w-4" />
                  </button>
                  <button
                    type="button"
                    onClick={handleCancelEdit}
                    className="cursor-pointer text-muted-foreground hover:text-foreground"
                    aria-label="Cancel"
                  >
                    <X className="h-4 w-4" />
                  </button>
                </>
              ) : (
                <>
                  <span className="font-medium">
                    {formatMargin(row.margin)}
                  </span>
                  <button
                    type="button"
                    onClick={() => handleStartEdit(row.provider, row.margin)}
                    className="cursor-pointer text-primary hover:text-primary/80"
                    aria-label="Edit"
                  >
                    <Pencil className="h-4 w-4" />
                  </button>
                </>
              )}
            </div>
          ),
          width: "350px",
        },
        {
          header: "Actions",
          cell: (row) => {
            const displayName =
              row.provider === "global"
                ? "Global"
                : getProviderDisplayInfo(row.provider).displayName;
            return (
              <button
                type="button"
                onClick={() => onRemoveProvider(row.provider, displayName)}
                className="cursor-pointer hover:text-destructive"
                aria-label="Remove"
              >
                <Trash2 className="h-4 w-4" />
              </button>
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
