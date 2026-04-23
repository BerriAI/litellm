import React, { useState } from "react";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { Check, Copy } from "lucide-react";
import { cn } from "@/lib/utils";
import DefaultProxyAdminTag from "./DefaultProxyAdminTag";

interface LabeledFieldProps {
  label: string;
  value: string;
  icon?: React.ReactNode;
  truncate?: boolean;
  copyable?: boolean;
  defaultUserIdCheck?: boolean;
}

const CopyableValue: React.FC<{
  value: string;
  label: string;
  truncate?: boolean;
}> = ({ value, label, truncate }) => {
  const [copied, setCopied] = useState(false);
  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(value);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
      // eslint-disable-next-line @typescript-eslint/no-unused-vars
    } catch (_err) {
      /* noop */
    }
  };
  return (
    <div className="inline-flex items-center gap-1.5">
      <span
        className={cn(
          "font-semibold",
          truncate && "block max-w-[160px] truncate",
        )}
      >
        {value}
      </span>
      <TooltipProvider>
        <Tooltip open={copied || undefined}>
          <TooltipTrigger asChild>
            <button
              type="button"
              onClick={handleCopy}
              className="text-muted-foreground hover:text-foreground"
              aria-label={`Copy ${label}`}
            >
              {copied ? (
                <Check className="h-3 w-3 text-emerald-500" />
              ) : (
                <Copy className="h-3 w-3" />
              )}
            </button>
          </TooltipTrigger>
          <TooltipContent>{copied ? "Copied!" : `Copy ${label}`}</TooltipContent>
        </Tooltip>
      </TooltipProvider>
    </div>
  );
};

export default function LabeledField({
  label,
  value,
  icon,
  truncate = false,
  copyable = false,
  defaultUserIdCheck = false,
}: LabeledFieldProps) {
  const isEmpty = !value;
  const isDefaultUser = defaultUserIdCheck && value === "default_user_id";
  const displayValue = isEmpty ? "-" : value;
  const isCopyable = copyable && !isEmpty && !isDefaultUser;

  const valueEl = isDefaultUser ? (
    <DefaultProxyAdminTag userId={value} />
  ) : isCopyable ? (
    <CopyableValue value={displayValue} label={label} truncate={truncate} />
  ) : (
    <span
      className={cn(
        "font-semibold",
        truncate && "block max-w-[160px] truncate",
      )}
    >
      {displayValue}
    </span>
  );

  return (
    <div>
      <div className="flex items-center gap-1 text-muted-foreground">
        {icon && <span className="text-xs">{icon}</span>}
        <span className="text-xs uppercase tracking-wider">{label}</span>
      </div>
      <div>{valueEl}</div>
    </div>
  );
}
