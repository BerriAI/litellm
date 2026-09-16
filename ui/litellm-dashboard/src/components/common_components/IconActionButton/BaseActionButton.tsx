import { cx } from "@/lib/cva.config";
import React from "react";

interface BaseActionButtonProps {
  icon: React.ComponentType<React.ComponentProps<"svg">>;
  onClick: () => void;
  className?: string;
  disabled?: boolean;
  dataTestId?: string;
}

export default function BaseActionButton({
  icon: Icon,
  onClick,
  className,
  disabled,
  dataTestId,
}: BaseActionButtonProps) {
  return disabled ? (
    <span
      className="inline-flex shrink-0 cursor-not-allowed items-center justify-center p-1.5 opacity-50"
      data-testid={dataTestId}
    >
      <Icon className="size-5 shrink-0" />
    </span>
  ) : (
    <span
      className={cx("inline-flex shrink-0 cursor-pointer items-center justify-center p-1.5", className)}
      onClick={onClick}
      data-testid={dataTestId}
    >
      <Icon className="size-5 shrink-0" />
    </span>
  );
}
