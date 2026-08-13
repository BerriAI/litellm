import React, { useState } from "react";
import { Minus, Plus } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";

interface KeyValueInputProps {
  value?: Record<string, string>;
  onChange?: (value: Record<string, string>) => void;
}

const KeyValueInput: React.FC<KeyValueInputProps> = ({ value = {}, onChange }) => {
  const [pairs, setPairs] = useState<[string, string][]>(Object.entries(value));

  const handleAdd = () => {
    setPairs([...pairs, ["", ""]]);
  };

  const handleRemove = (index: number) => {
    const newPairs = pairs.filter((_, i) => i !== index);
    setPairs(newPairs);
    onChange?.(Object.fromEntries(newPairs));
  };

  const handleChange = (index: number, key: string, val: string) => {
    const newPairs = [...pairs];
    newPairs[index] = [key, val];
    setPairs(newPairs);
    onChange?.(Object.fromEntries(newPairs));
  };

  return (
    <div className="space-y-2">
      {pairs.map(([key, val], index) => (
        <div key={index} className="flex items-center gap-2">
          <Input placeholder="Header Name" value={key} onChange={(e) => handleChange(index, e.target.value, val)} />
          <Input placeholder="Header Value" value={val} onChange={(e) => handleChange(index, key, e.target.value)} />
          <Button
            type="button"
            variant="ghost"
            size="icon-sm"
            onClick={() => handleRemove(index)}
            aria-label={`Remove header ${index + 1}`}
          >
            <Minus />
          </Button>
        </div>
      ))}
      <Button type="button" variant="outline" onClick={handleAdd}>
        <Plus />
        Add Header
      </Button>
    </div>
  );
};

export default KeyValueInput;
