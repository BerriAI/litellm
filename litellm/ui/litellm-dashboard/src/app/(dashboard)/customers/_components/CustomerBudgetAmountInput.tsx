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
}: {
  id: string;
  label: string;
  value: string;
  onChange: (value: string) => void;
  disabled?: boolean;
  placeholder?: string;
  children?: ReactNode;
}) {
  const { t } = useTranslation();
  return (
    <InputGroup>
      <InputGroupAddon>
        <InputGroupText>$</InputGroupText>
      </InputGroupAddon>
      <InputGroupInput
        id={id}
        aria-label={label}
        type="number"
        min={0}
        step="any"
        disabled={disabled}
        value={value}
        onChange={(event) => onChange(event.target.value)}
        placeholder={placeholder ?? t("customerPortal.createDialog.noLimit")}
      />
      {children}
    </InputGroup>
  );
}
