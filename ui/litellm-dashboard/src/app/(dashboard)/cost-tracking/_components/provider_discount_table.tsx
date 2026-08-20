import React, { useState } from "react";
import { Check, SquarePen, Trash2, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { SimpleTable } from "@/components/common_components/simple_table";
import { DiscountConfig } from "./types";
import { getProviderLogoAndName } from "@/components/provider_info_helpers";
import { Logo } from "@/components/molecules/logo/Logo";

interface ProviderDiscountTableProps {
  discountConfig: DiscountConfig;
  onDiscountChange: (provider: string, value: string) => void;
  onRemoveProvider: (provider: string, providerDisplayName: string) => void;
}

interface ProviderDiscountRow {
  provider: string;
  discount: number;
}

const ProviderDiscountTable: React.FC<ProviderDiscountTableProps> = ({
  discountConfig,
  onDiscountChange,
  onRemoveProvider,
}) => {
  const [editingProvider, setEditingProvider] = useState<string | null>(null);
  const [editValue, setEditValue] = useState<string>("");

  const handleStartEdit = (provider: string, currentDiscount: number) => {
    setEditingProvider(provider);
    setEditValue((currentDiscount * 100).toString());
  };

  const handleSaveEdit = (provider: string) => {
    const percentValue = parseFloat(editValue);
    if (!isNaN(percentValue) && percentValue >= 0 && percentValue <= 100) {
      onDiscountChange(provider, (percentValue / 100).toString());
    }
    setEditingProvider(null);
    setEditValue("");
  };

  const handleCancelEdit = () => {
    setEditingProvider(null);
    setEditValue("");
  };

  const handleKeyDown = (e: React.KeyboardEvent, provider: string) => {
    if (e.key === "Enter") {
      handleSaveEdit(provider);
    } else if (e.key === "Escape") {
      handleCancelEdit();
    }
  };

  // Convert discount config to array and sort
  const data: ProviderDiscountRow[] = Object.entries(discountConfig)
    .map(([provider, discount]) => ({ provider, discount }))
    .sort((a, b) => {
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
          header: "Discount Percentage",
          cell: (row) => {
            const { displayName } = getProviderLogoAndName(row.provider);
            return (
              <div className="flex items-center gap-2">
                {editingProvider === row.provider ? (
                  <>
                    <Input
                      value={editValue}
                      onChange={(e) => setEditValue(e.target.value)}
                      onKeyDown={(e) => handleKeyDown(e, row.provider)}
                      placeholder="5"
                      className="w-20"
                      autoFocus
                    />
                    <span className="text-muted-foreground">%</span>
                    <Button
                      variant="ghost"
                      size="icon-sm"
                      aria-label={`Save discount for ${displayName}`}
                      onClick={() => handleSaveEdit(row.provider)}
                      className="cursor-pointer text-success hover:text-success/80"
                    >
                      <Check className="size-5" />
                    </Button>
                    <Button
                      variant="ghost"
                      size="icon-sm"
                      aria-label={`Cancel editing discount for ${displayName}`}
                      onClick={handleCancelEdit}
                      className="cursor-pointer text-muted-foreground hover:text-foreground"
                    >
                      <X className="size-5" />
                    </Button>
                  </>
                ) : (
                  <>
                    <p className="font-medium">{(row.discount * 100).toFixed(1)}%</p>
                    <Button
                      variant="ghost"
                      size="icon-sm"
                      aria-label={`Edit discount for ${displayName}`}
                      onClick={() => handleStartEdit(row.provider, row.discount)}
                      className="cursor-pointer text-info hover:text-info/80"
                    >
                      <SquarePen className="size-5" />
                    </Button>
                  </>
                )}
              </div>
            );
          },
          width: "250px",
        },
        {
          header: "Actions",
          cell: (row) => {
            const { displayName } = getProviderLogoAndName(row.provider);
            return (
              <Button
                variant="ghost"
                size="icon-sm"
                aria-label={`Remove discount for ${displayName}`}
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
      emptyMessage="No provider discounts configured"
    />
  );
};

export default ProviderDiscountTable;
