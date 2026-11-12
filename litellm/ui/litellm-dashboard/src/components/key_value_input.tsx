import React from "react";
import { Minus, Plus } from "lucide-react";
import { useTranslation } from "react-i18next";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";

export type KeyValuePair = readonly [string, string];

interface KeyValueInputProps {
  value?: readonly KeyValuePair[];
  onChange?: (value: readonly KeyValuePair[]) => void;
}

const KeyValueInput: React.FC<KeyValueInputProps> = ({ value = [], onChange }) => {
  const { t } = useTranslation();
  const handleAdd = () => onChange?.([...value, ["", ""]]);

  const handleRemove = (index: number) => onChange?.(value.filter((_, i) => i !== index));

  const handleChange = (index: number, pair: KeyValuePair) =>
    onChange?.(value.map((existing, i) => (i === index ? pair : existing)));

  return (
    <div className="space-y-2">
      {value.map(([key, val], index) => (
        <div key={index} className="flex items-center gap-2">
          <Input
            placeholder={t("keyValueInput.headerName")}
            value={key}
            onChange={(e) => handleChange(index, [e.target.value, val])}
          />
          <Input
            placeholder={t("keyValueInput.headerValue")}
            value={val}
            onChange={(e) => handleChange(index, [key, e.target.value])}
          />
          <Button
            type="button"
            variant="ghost"
            size="icon-sm"
            onClick={() => handleRemove(index)}
            aria-label={t("keyValueInput.removeHeader", { position: index + 1 })}
          >
            <Minus />
          </Button>
        </div>
      ))}
      <Button type="button" variant="outline" onClick={handleAdd}>
        <Plus />
        {t("keyValueInput.addHeader")}
      </Button>
    </div>
  );
};

export default KeyValueInput;
