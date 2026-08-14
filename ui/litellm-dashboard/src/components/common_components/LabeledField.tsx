import React from "react";
import CopyButton from "@/components/shared/CopyButton";
import { cx } from "@/lib/cva.config";
import DefaultProxyAdminTag from "./DefaultProxyAdminTag";

interface LabeledFieldProps {
  label: string;
  value: string;
  icon?: React.ReactNode;
  truncate?: boolean;
  copyable?: boolean;
  defaultUserIdCheck?: boolean;
}

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
  ) : (
    <span className="inline-flex min-w-0 items-center gap-1">
      <strong className={cx("font-semibold", truncate ? "block max-w-40 truncate" : "break-words")}>
        {displayValue}
      </strong>
      {isCopyable && <CopyButton value={value} label={`Copy ${label}`} />}
    </span>
  );
  return (
    <div className="min-w-0">
      <div className="flex items-center gap-1 text-muted-foreground">
        {icon}
        <span className="text-xs tracking-wider uppercase">{label}</span>
      </div>
      <div className="min-w-0">{valueEl}</div>
    </div>
  );
}
