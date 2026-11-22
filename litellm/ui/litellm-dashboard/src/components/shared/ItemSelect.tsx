"use client";

import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";

interface ItemSelectProps<Value extends string> {
  id: string;
  value: Value;
  items: readonly { value: Value; label: string }[];
  onValueChange: (value: Value) => void;
  className?: string;
  disabled?: boolean;
}

export function ItemSelect<Value extends string>({
  id,
  value,
  items,
  onValueChange,
  className,
  disabled,
}: ItemSelectProps<Value>) {
  return (
    <Select<Value>
      value={value}
      items={items}
      onValueChange={(next) => next !== null && onValueChange(next)}
      disabled={disabled}
    >
      <SelectTrigger id={id} className={className}>
        <SelectValue />
      </SelectTrigger>
      <SelectContent>
        {items.map((item) => (
          <SelectItem key={item.value} value={item.value}>
            {item.label}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  );
}
