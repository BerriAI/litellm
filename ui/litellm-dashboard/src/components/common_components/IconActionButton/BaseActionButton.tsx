import { cx } from "@/lib/cva.config";
import { Icon } from "@tremor/react";
import React from "react";

interface BaseActionButtonProps {
  icon: React.ComponentType<React.ComponentProps<"svg">>;
  onClick: () => void;
  className?: string;
  disabled?: boolean;
  dataTestId?: string;
}

export default function BaseActionButton({ icon, onClick, className, disabled, dataTestId }: BaseActionButtonProps) {
  return disabled ? (
    <Icon icon={icon} size="sm" className={"opacity-50 cursor-not-allowed"} data-testid={dataTestId} />
  ) : (
    <Icon
      icon={icon}
      size="sm"
      onClick={onClick}
      className={cx("cursor-pointer", className)}
      data-testid={dataTestId}
    />
  );
}
