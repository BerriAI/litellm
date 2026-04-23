import React, { useState } from "react";
import { Input } from "@/components/ui/input";
import { Check, Pencil, Trash2, X } from "lucide-react";
import { SimpleTable } from "../common_components/simple_table";
import { DiscountConfig } from "./types";
import {
  getProviderDisplayInfo,
  handleImageError,
} from "./provider_display_helpers";

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

  const data: ProviderDiscountRow[] = Object.entries(discountConfig)
    .map(([provider, discount]) => ({ provider, discount }))
    .sort((a, b) => {
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
          header: "Discount Percentage",
          cell: (row) => (
            <div className="flex items-center gap-2">
              {editingProvider === row.provider ? (
                <>
                  <Input
                    value={editValue}
                    onChange={(e) => setEditValue(e.target.value)}
                    onKeyDown={(e) => handleKeyDown(e, row.provider)}
                    placeholder="5"
                    className="w-20 h-8"
                    autoFocus
                  />
                  <span className="text-muted-foreground">%</span>
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
                    {(row.discount * 100).toFixed(1)}%
                  </span>
                  <button
                    type="button"
                    onClick={() => handleStartEdit(row.provider, row.discount)}
                    className="cursor-pointer text-primary hover:text-primary/80"
                    aria-label="Edit"
                  >
                    <Pencil className="h-4 w-4" />
                  </button>
                </>
              )}
            </div>
          ),
          width: "250px",
        },
        {
          header: "Actions",
          cell: (row) => {
            const { displayName } = getProviderDisplayInfo(row.provider);
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
      emptyMessage="No provider discounts configured"
    />
  );
};

export default ProviderDiscountTable;
