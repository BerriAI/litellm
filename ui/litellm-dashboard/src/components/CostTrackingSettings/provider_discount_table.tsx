import React from "react";
import { TextInput, Icon } from "@tremor/react";
import { TrashIcon } from "@heroicons/react/outline";
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
              <TextInput
                value={(row.discount * 100).toString()}
                onValueChange={(value) => {
                  // Convert percentage back to decimal
                  const percentValue = parseFloat(value);
                  if (!isNaN(percentValue)) {
                    onDiscountChange(row.provider, (percentValue / 100).toString());
                  }
                }}
                placeholder="5"
                className="w-20"
              />
              <span className="text-gray-600">%</span>
            </div>
          ),
          width: "200px",
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

