"use client";

import { Button } from "@/components/ui/button";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { cn } from "@/lib/cva.config";
import { copyToClipboard } from "@/utils/dataUtils";
import { Check, Copy } from "lucide-react";
import React, { useEffect, useState } from "react";

interface CopyButtonProps {
  value: string | null | undefined;
  label: string;
  className?: string;
  iconClassName?: string;
  variant?: "icon" | "action";
  iconOnly?: boolean;
  copiedLabel?: string;
}

const VARIANTS = {
  icon: {
    resetMs: 1200,
    iconClassName: "size-[15px]",
    className: "text-muted-foreground hover:text-primary",
  },
  action: {
    resetMs: 1600,
    iconClassName: "size-3",
    className: "text-xs text-muted-foreground hover:text-foreground",
  },
} as const;

const CopyButton: React.FC<CopyButtonProps> = ({
  value,
  label,
  className,
  iconClassName,
  variant = "icon",
  iconOnly = false,
  copiedLabel = "Copied",
}) => {
  const [copied, setCopied] = useState(false);
  const isAction = variant === "action";
  const showLabel = isAction && !iconOnly;
  const { resetMs, className: variantClassName, iconClassName: defaultIconClassName } = VARIANTS[variant];
  const iconSize = iconClassName ?? defaultIconClassName;

  useEffect(() => {
    if (!copied) return;
    const timer = setTimeout(() => setCopied(false), resetMs);
    return () => clearTimeout(timer);
  }, [copied, resetMs]);

  if (value == null || (!value && !isAction)) return null;

  const handleCopy = async () => {
    if (isAction) {
      setCopied(await copyToClipboard(value, copiedLabel));
      return;
    }
    if (!navigator.clipboard) return;
    try {
      await navigator.clipboard.writeText(value);
      setCopied(true);
    } catch {
      setCopied(false);
    }
  };

  const button = (
    <Button
      type="button"
      variant={showLabel ? "outline" : "ghost"}
      size={showLabel ? "xs" : "icon-xs"}
      onClick={handleCopy}
      aria-label={label}
      title={isAction ? undefined : label}
      className={cn(variantClassName, showLabel && "gap-1.5", className)}
    >
      {copied ? <Check className={iconSize} /> : <Copy className={iconSize} />}
      {showLabel && <span>{copied ? copiedLabel : label}</span>}
    </Button>
  );

  if (!isAction || !iconOnly) return button;
  return (
    <TooltipProvider>
      <Tooltip>
        <TooltipTrigger render={button} />
        <TooltipContent>{copied ? copiedLabel : label}</TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
};

export default CopyButton;
