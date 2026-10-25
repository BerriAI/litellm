import { useLanguage } from "@/contexts/LanguageContext";
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
  key: string;
  provider: string;
  modelPattern: string | null;
  discount: number;
}

const splitDiscountKey = (key: string): { provider: string; modelPattern: string | null } => {
  const slashIndex = key.indexOf("/");
  if (slashIndex < 0) {
    return { provider: key, modelPattern: null };
  }
  return { provider: key.slice(0, slashIndex), modelPattern: key.slice(slashIndex + 1) };
};

const ProviderDiscountTable: React.FC<ProviderDiscountTableProps> = ({
  discountConfig,
  onDiscountChange,
  onRemoveProvider,
}) => {
  const { t } = useLanguage();
  const [editingKey, setEditingKey] = useState<string | null>(null);
  const [editValue, setEditValue] = useState<string>("");

  const handleStartEdit = (key: string, currentDiscount: number) => {
    setEditingKey(key);
    setEditValue((currentDiscount * 100).toString());
  };

  const handleSaveEdit = (key: string) => {
    const percentValue = parseFloat(editValue);
    if (!isNaN(percentValue) && percentValue >= 0 && percentValue <= 100) {
      onDiscountChange(key, (percentValue / 100).toString());
    }
    setEditingKey(null);
    setEditValue("");
  };

  const handleCancelEdit = () => {
    setEditingKey(null);
    setEditValue("");
  };

  const handleKeyDown = (e: React.KeyboardEvent, key: string) => {
    if (e.key === "Enter") {
      handleSaveEdit(key);
    } else if (e.key === "Escape") {
      handleCancelEdit();
    }
  };

  const rowLabel = (row: ProviderDiscountRow): string => {
    const { displayName } = getProviderLogoAndName(row.provider);
    return row.modelPattern ? `${displayName} (${row.modelPattern})` : displayName;
  };

  // Convert discount config to array and sort
  const data: ProviderDiscountRow[] = Object.entries(discountConfig)
    .map(([key, discount]) => ({ key, ...splitDiscountKey(key), discount }))
    .sort((a, b) => {
      const displayA = getProviderLogoAndName(a.provider).displayName;
      const displayB = getProviderLogoAndName(b.provider).displayName;
      return displayA.localeCompare(displayB) || (a.modelPattern ?? "").localeCompare(b.modelPattern ?? "");
    });

  return (
    <SimpleTable
      data={data}
      columns={[
        {
          header: t("Provider"),
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
          header: t("Models"),
          cell: (row) =>
            row.modelPattern ? (
              <span className="font-mono text-sm">{row.modelPattern}</span>
            ) : (
              <span className="text-muted-foreground">{t("All models")}</span>
            ),
        },
        {
          header: t("Discount Percentage"),
          numeric: true,
          cell: (row) => {
            const label = rowLabel(row);
            return (
              <div className="flex items-center justify-end gap-2">
                {editingKey === row.key ? (
                  <>
                    <Input
                      value={editValue}
                      onChange={(e) => setEditValue(e.target.value)}
                      onKeyDown={(e) => handleKeyDown(e, row.key)}
                      placeholder="5"
                      className="w-20"
                      autoFocus
                    />
                    <span className="text-muted-foreground">%</span>
                    <Button
                      variant="ghost"
                      size="icon-sm"
                      aria-label={t("Save discount for {value1}", { value1: String(label) })}
                      onClick={() => handleSaveEdit(row.key)}
                      className="cursor-pointer text-success hover:text-success/80"
                    >
                      <Check className="size-5" />
                    </Button>
                    <Button
                      variant="ghost"
                      size="icon-sm"
                      aria-label={t("Cancel editing discount for {value1}", { value1: String(label) })}
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
                      aria-label={t("Edit discount for {value1}", { value1: String(label) })}
                      onClick={() => handleStartEdit(row.key, row.discount)}
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
          header: t("Actions"),
          cell: (row) => {
            const label = rowLabel(row);
            return (
              <Button
                variant="ghost"
                size="icon-sm"
                aria-label={t("Remove discount for {value1}", { value1: String(label) })}
                onClick={() => onRemoveProvider(row.key, label)}
                className="cursor-pointer hover:text-destructive"
              >
                <Trash2 className="size-5" />
              </Button>
            );
          },
          width: "80px",
        },
      ]}
      getRowKey={(row) => row.key}
      emptyMessage={t("No provider discounts configured")}
    />
  );
};

export default ProviderDiscountTable;
