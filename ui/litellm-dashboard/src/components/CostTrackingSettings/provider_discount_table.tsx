import React from "react";
import { TextInput, Button } from "@tremor/react";
import { SimpleTable } from "../common_components/simple_table";
import { DiscountConfig } from "./types";
import { getProviderDisplayInfo, handleImageError } from "./provider_display_helpers";

interface ProviderDiscountTableProps {
  discountConfig: DiscountConfig;
  onDiscountChange: (provider: string, value: string) => void;
  onRemoveProvider: (provider: string) => void;
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
          header: "Discount Value",
          cell: (row) => (
            <TextInput
              value={row.discount.toString()}
              onValueChange={(value) => onDiscountChange(row.provider, value)}
              placeholder="0.05"
              className="w-32"
            />
          ),
          width: "200px",
        },
        {
          header: "Percentage",
          cell: (row) => (
            <span className="text-gray-700 font-medium">
              {(row.discount * 100).toFixed(1)}%
            </span>
          ),
          width: "120px",
        },
        {
          header: "Actions",
          cell: (row) => (
            <Button
              size="xs"
              variant="secondary"
              color="red"
              onClick={() => onRemoveProvider(row.provider)}
            >
              Remove
            </Button>
          ),
          width: "120px",
        },
      ]}
      getRowKey={(row) => row.provider}
      emptyMessage="No provider discounts configured"
    />
  );
};

export default ProviderDiscountTable;

