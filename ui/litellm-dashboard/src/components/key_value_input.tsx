import React, { useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { MinusCircle, Plus } from "lucide-react";

interface KeyValueInputProps {
  value?: Record<string, string>;
  onChange?: (value: Record<string, string>) => void;
}

const KeyValueInput: React.FC<KeyValueInputProps> = ({
  value = {},
  onChange,
}) => {
  const [pairs, setPairs] = useState<[string, string][]>(
    Object.entries(value),
  );

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
    <div>
      {pairs.map(([key, val], index) => (
        <div key={index} className="flex items-center gap-2 mb-2">
          <Input
            placeholder="Header Name"
            value={key}
            onChange={(e) => handleChange(index, e.target.value, val)}
          />
          <Input
            placeholder="Header Value"
            value={val}
            onChange={(e) => handleChange(index, key, e.target.value)}
          />
          <Button
            type="button"
            variant="ghost"
            size="icon"
            onClick={() => handleRemove(index)}
            aria-label="Remove header"
            className="h-8 w-8 shrink-0"
          >
            <MinusCircle className="h-4 w-4" />
          </Button>
        </div>
      ))}
      <Button
        type="button"
        variant="outline"
        onClick={handleAdd}
        className="border-dashed"
      >
        <Plus className="h-4 w-4" />
        Add Header
      </Button>
    </div>
  );
};

export default KeyValueInput;
