import { forwardRef, useState, type ChangeEvent, type ComponentProps } from "react";
import { Input } from "@/components/ui/input";
import { useCurrency } from "@/app/(dashboard)/hooks/uiSettings/useCurrency";
import { fromUsd, toUsd } from "@/lib/money";

type MoneyInputProps = Omit<ComponentProps<typeof Input>, "type" | "value" | "onChange"> & {
  value: number | null | undefined;
  onChange: (value: number | null) => void;
};

export const MoneyInput = forwardRef<HTMLInputElement, MoneyInputProps>(function MoneyInput(
  { value, onChange, ...props },
  ref,
) {
  const currency = useCurrency();
  const [edit, setEdit] = useState<{
    text: string;
    value: number | null | undefined;
    code: string;
    rate: number;
  } | null>(null);
  const symbol = new Intl.NumberFormat("en-US", { style: "currency", currency: currency.code })
    .formatToParts(0)
    .find((part) => part.type === "currency")?.value;
  const display = value == null ? "" : String(Number(fromUsd(value, currency).toPrecision(15)));
  const sameCurrency = edit?.code === currency.code && edit?.rate === currency.usd_rate;
  const text = edit && edit.value === value && sameCurrency ? edit.text : display;

  function handleChange(event: ChangeEvent<HTMLInputElement>) {
    const next = event.target.value;
    const converted = toUsd(Number(next), currency);
    const valid = /^-?(?:\d+\.?\d*|\.\d+)$/.test(next) && Number.isFinite(converted);
    const retained = valid ? converted : value;
    const emitted = next === "" ? null : retained;
    const draft = { text: next, value: emitted, code: currency.code, rate: currency.usd_rate };
    setEdit(draft);
    if (next === "") onChange(null);
    else if (valid && emitted != null) onChange(emitted);
  }

  return (
    <div className="flex items-center gap-2">
      <span aria-hidden="true">{symbol}</span>
      <Input
        {...props}
        ref={ref}
        type="text"
        inputMode="decimal"
        value={text}
        onChange={handleChange}
        onBlur={(event) => {
          setEdit(null);
          props.onBlur?.(event);
        }}
      />
    </div>
  );
});
