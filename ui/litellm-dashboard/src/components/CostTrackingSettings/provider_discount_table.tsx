import React from "react";
import {
  Table,
  TableHead,
  TableRow,
  TableHeaderCell,
  TableBody,
  TableCell,
  TextInput,
  Button,
} from "@tremor/react";
import { DiscountConfig } from "./types";
import { getProviderDisplayInfo, handleImageError } from "./provider_display_helpers";

interface ProviderDiscountTableProps {
  discountConfig: DiscountConfig;
  onDiscountChange: (provider: string, value: string) => void;
  onRemoveProvider: (provider: string) => void;
}

const ProviderDiscountTable: React.FC<ProviderDiscountTableProps> = ({
  discountConfig,
  onDiscountChange,
  onRemoveProvider,
}) => {
  return (
    <Table>
      <TableHead>
        <TableRow>
          <TableHeaderCell>Provider</TableHeaderCell>
          <TableHeaderCell>Discount Value</TableHeaderCell>
          <TableHeaderCell>Percentage</TableHeaderCell>
          <TableHeaderCell>Actions</TableHeaderCell>
        </TableRow>
      </TableHead>
      <TableBody>
        {Object.entries(discountConfig)
          .sort(([a], [b]) => {
            const displayA = getProviderDisplayInfo(a).displayName;
            const displayB = getProviderDisplayInfo(b).displayName;
            return displayA.localeCompare(displayB);
          })
          .map(([provider, discount]) => {
            const { displayName, logo } = getProviderDisplayInfo(provider);
            return (
              <TableRow key={provider}>
                <TableCell>
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
                </TableCell>
                <TableCell>
                  <TextInput
                    value={discount.toString()}
                    onValueChange={(value) => onDiscountChange(provider, value)}
                    placeholder="0.05"
                    className="w-32"
                  />
                </TableCell>
                <TableCell>
                  <span className="text-gray-700 font-medium">
                    {(discount * 100).toFixed(1)}%
                  </span>
                </TableCell>
                <TableCell>
                  <Button
                    size="xs"
                    variant="secondary"
                    color="red"
                    onClick={() => onRemoveProvider(provider)}
                  >
                    Remove
                  </Button>
                </TableCell>
              </TableRow>
            );
          })}
      </TableBody>
    </Table>
  );
};

export default ProviderDiscountTable;

