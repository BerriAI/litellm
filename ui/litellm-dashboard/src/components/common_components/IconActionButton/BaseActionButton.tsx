import { cx } from "@/lib/cva.config";
import React from "react";

interface BaseActionButtonProps {
  icon: React.ComponentType<React.ComponentProps<"svg">>;
  onClick: () => void;
  label?: string;
  className?: string;
  disabled?: boolean;
  dataTestId?: string;
}

export default function BaseActionButton({
  icon: Icon,
  onClick,
  label,
  className,
  disabled,
  dataTestId,
}: BaseActionButtonProps) {
  return (
    <button
      type="button"
      aria-label={label}
      aria-disabled={disabled || undefined}
      className={cx(
        "inline-flex shrink-0 items-center justify-center rounded-md p-1.5 outline-none focus-visible:ring-2 focus-visible:ring-ring",
        disabled ? "cursor-not-allowed opacity-50" : cx("cursor-pointer", className),
      )}
      onClick={disabled ? undefined : onClick}
      data-testid={dataTestId}
    >
      <Icon className="size-5 shrink-0" aria-hidden />
    </button>
  );
}
