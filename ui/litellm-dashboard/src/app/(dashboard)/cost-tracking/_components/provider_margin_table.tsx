import React, { useState } from "react";
import { Check, SquarePen, Trash2, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { SimpleTable } from "@/components/common_components/simple_table";
import { MarginConfig } from "./types";
import { getProviderLogoAndName } from "@/components/provider_info_helpers";
import { Logo } from "@/components/molecules/logo/Logo";

interface ProviderMarginTableProps {
  marginConfig: MarginConfig;
  onMarginChange: (provider: string, value: number | { percentage?: number; fixed_amount?: number }) => void;
  onRemoveProvider: (provider: string, providerDisplayName: string) => void;
}

interface ProviderMarginRow {
  provider: string;
  margin: number | { percentage?: number; fixed_amount?: number };
}

const marginRowDisplayName = (provider: string): string =>
  provider === "global" ? "Global" : getProviderLogoAndName(provider).displayName;

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
      const displayA = getProviderLogoAndName(a.provider).displayName;
      const displayB = getProviderLogoAndName(b.provider).displayName;
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
            const { displayName } = getProviderLogoAndName(row.provider);
            return (
              <div className="flex items-center space-x-2">
                <Logo provider={row.provider} label={displayName} className="w-5 h-5" />
                <span className="font-medium">{displayName}</span>
              </div>
            );
          },
        },
        {
          header: "Margin",
          cell: (row) => {
            const displayName = marginRowDisplayName(row.provider);
            return (
              <div className="flex items-center gap-2">
                {editingProvider === row.provider ? (
                  <>
                    <div className="flex items-center gap-2">
                      <Input
                        value={editPercentage}
                        onChange={(e) => setEditPercentage(e.target.value)}
                        placeholder="10"
                        className="w-20"
                        autoFocus
                      />
                      <span className="text-muted-foreground">%</span>
                      <span className="text-muted-foreground">+</span>
                      <span className="text-muted-foreground">$</span>
                      <Input
                        value={editFixedAmount}
                        onChange={(e) => setEditFixedAmount(e.target.value)}
                        placeholder="0.001"
                        className="w-24"
                      />
                    </div>
                    <Button
                      variant="ghost"
                      size="icon-sm"
                      aria-label={`Save margin for ${displayName}`}
                      onClick={() => handleSaveEdit(row.provider)}
                      className="cursor-pointer text-success hover:text-success/80"
                    >
                      <Check className="size-5" />
                    </Button>
                    <Button
                      variant="ghost"
                      size="icon-sm"
                      aria-label={`Cancel editing margin for ${displayName}`}
                      onClick={handleCancelEdit}
                      className="cursor-pointer text-muted-foreground hover:text-foreground"
                    >
                      <X className="size-5" />
                    </Button>
                  </>
                ) : (
                  <>
                    <p className="font-medium">{formatMargin(row.margin)}</p>
                    <Button
                      variant="ghost"
                      size="icon-sm"
                      aria-label={`Edit margin for ${displayName}`}
                      onClick={() => handleStartEdit(row.provider, row.margin)}
                      className="cursor-pointer text-info hover:text-info/80"
                    >
                      <SquarePen className="size-5" />
                    </Button>
                  </>
                )}
              </div>
            );
          },
          width: "350px",
        },
        {
          header: "Actions",
          cell: (row) => {
            const displayName = marginRowDisplayName(row.provider);
            return (
              <Button
                variant="ghost"
                size="icon-sm"
                aria-label={`Remove margin for ${displayName}`}
                onClick={() => onRemoveProvider(row.provider, displayName)}
                className="cursor-pointer hover:text-destructive"
              >
                <Trash2 className="size-5" />
              </Button>
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
