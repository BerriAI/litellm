import { useTranslation } from "react-i18next";
import type { ReactNode } from "react";
import { InputGroup, InputGroupAddon, InputGroupInput, InputGroupText } from "@/components/ui/input-group";

export default function CustomerBudgetAmountInput({
  id,
  label,
  value,
  onChange,
  disabled,
  placeholder,
  children,
  compact = false,
  "aria-describedby": describedBy,
}: {
  id: string;
  label: string;
  value: string;
  onChange: (value: string) => void;
  disabled?: boolean;
  placeholder?: string;
  children?: ReactNode;
  compact?: boolean;
  "aria-describedby"?: string;
}) {
  const { t } = useTranslation();
  return (
    <InputGroup
      className={`has-[[role=alert]]:h-auto has-[[role=alert]]:flex-wrap ${compact ? "h-8 w-auto shrink-0 shadow-none" : ""}`}
    >
      <InputGroupAddon>
        <InputGroupText className={compact ? "text-xs" : undefined}>$</InputGroupText>
      </InputGroupAddon>
      <InputGroupInput
        id={id}
        aria-label={label}
        aria-describedby={describedBy}
        type="number"
        min={0}
        step="any"
        disabled={disabled}
        value={value}
        onChange={(event) => onChange(event.target.value)}
        placeholder={placeholder ?? t("customerPortal.createDialog.noLimit")}
        className={
          compact
            ? "h-8 flex-none appearance-textfield px-1.5 text-xs md:text-xs [&::-webkit-inner-spin-button]:appearance-none [&::-webkit-outer-spin-button]:appearance-none"
            : undefined
        }
        style={compact ? { width: `calc(${Math.max(value.length, value ? 3 : 7)}ch + 0.75rem)` } : undefined}
      />
      {children}
    </InputGroup>
  );
}
