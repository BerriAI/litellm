import { cx } from "@/lib/cva.config";
import React from "react";

interface BaseActionButtonProps {
  // Accepts lucide-react components and plain SVG components alike.
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  icon: React.ComponentType<any>;
  onClick: () => void;
  className?: string;
  disabled?: boolean;
  dataTestId?: string;
}

/**
 * Compact icon-button used inside table action cells. Renders the icon as
 * a 16px lucide-style svg inside a transparent button so it picks up the
 * caller's hover/text-color classes (no @tremor Icon dependency).
 */
export default function BaseActionButton({
  icon: IconComp,
  onClick,
  className,
  disabled,
  dataTestId,
}: BaseActionButtonProps) {
  return (
    <button
      type="button"
      onClick={disabled ? undefined : onClick}
      disabled={disabled}
      data-testid={dataTestId}
      className={cx(
        "inline-flex items-center justify-center transition-colors",
        disabled
          ? "opacity-50 cursor-not-allowed"
          : "cursor-pointer text-muted-foreground",
        className,
      )}
      aria-disabled={disabled || undefined}
    >
      <IconComp className="w-4 h-4" />
    </button>
  );
}
