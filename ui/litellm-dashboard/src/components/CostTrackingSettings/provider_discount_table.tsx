import React, { useState } from "react";
import { TextInput, Icon, Text } from "@tremor/react";
import { TrashIcon, PencilAltIcon, CheckIcon, XIcon } from "@heroicons/react/outline";
import { SimpleTable } from "../common_components/simple_table";
import { DiscountConfig } from "./types";
import { getProviderDisplayInfo, handleImageError } from "./provider_display_helpers";

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
    if (e.key === 'Enter') {
      handleSaveEdit(provider);
    } else if (e.key === 'Escape') {
      handleCancelEdit();
    }
  };

  // Convert discount config to array and sort
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
                  <TextInput
                    value={editValue}
                    onValueChange={setEditValue}
                    onKeyDown={(e) => handleKeyDown(e, row.provider)}
                    placeholder="5"
                    className="w-20"
                    autoFocus
                  />
                  <span className="text-gray-600">%</span>
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
                  <Text className="font-medium">{(row.discount * 100).toFixed(1)}%</Text>
                  <Icon
                    icon={PencilAltIcon}
                    size="sm"
                    onClick={() => handleStartEdit(row.provider, row.discount)}
                    className="cursor-pointer text-blue-600 hover:text-blue-700"
                  />
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
      emptyMessage="No provider discounts configured"
    />
  );
};

export default ProviderDiscountTable;

