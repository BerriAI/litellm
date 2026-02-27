import { Button } from "antd";
import { RotateCcw } from "lucide-react";
import React from "react";

interface ResetFiltersButtonProps {
  onClick: () => void;
  label?: string;
}

export const ResetFiltersButton: React.FC<ResetFiltersButtonProps> = ({ onClick, label = "Reset Filters" }) => {
  return (
    <Button type="default" onClick={onClick} icon={<RotateCcw size={16} />}>
      {label}
    </Button>
  );
};
