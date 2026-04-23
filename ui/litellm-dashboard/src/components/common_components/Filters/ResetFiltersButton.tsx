import { Button } from "@/components/ui/button";
import { RotateCcw } from "lucide-react";
import React from "react";

interface ResetFiltersButtonProps {
  onClick: () => void;
  label?: string;
}

export const ResetFiltersButton: React.FC<ResetFiltersButtonProps> = ({
  onClick,
  label = "Reset Filters",
}) => {
  return (
    <Button variant="outline" onClick={onClick}>
      <RotateCcw size={16} />
      {label}
    </Button>
  );
};
