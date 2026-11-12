import { useState } from "react";
import { useTranslation } from "react-i18next";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { formatRateLimit, parseRateLimit } from "./customerForms";

interface Props {
  id: string;
  label: string;
  ariaLabel?: string;
  value: string;
  onChange: (value: string) => void;
}

export default function RateLimitInput({ id, label, ariaLabel, value, onChange }: Props) {
  const { t } = useTranslation();
  const [focused, setFocused] = useState(false);
  const parsed = parseRateLimit(value);
  const invalid = !parsed.valid;
  const showExpanded = focused && parsed.valid && /[kmb]$/i.test(value.trim());
  const expandedDescription = showExpanded ? `${id}-expanded` : undefined;

  return (
    <div className="min-w-0 space-y-2">
      <Label htmlFor={id} className="h-5 text-xs leading-5">
        {label}
      </Label>
      <Input
        id={id}
        aria-label={ariaLabel}
        aria-invalid={invalid}
        aria-describedby={invalid ? `${id}-error` : expandedDescription}
        inputMode="text"
        maxLength={48}
        className="h-10 tabular-nums"
        value={focused ? value : formatRateLimit(value)}
        onFocus={() => {
          onChange(formatRateLimit(value));
          setFocused(true);
        }}
        onChange={(event) => onChange(event.target.value)}
        onBlur={() => {
          onChange(formatRateLimit(value));
          setFocused(false);
        }}
        placeholder={t("customerPortal.rateLimits.noLimit")}
      />
      {invalid && (
        <p id={`${id}-error`} className="text-xs text-destructive">
          {t("customerPortal.rateLimits.invalidNumber")}
        </p>
      )}
      {showExpanded && (
        <p
          id={`${id}-expanded`}
          role="status"
          aria-live="polite"
          className="text-xs tabular-nums text-muted-foreground"
        >
          = {formatRateLimit(value)}
        </p>
      )}
    </div>
  );
}
