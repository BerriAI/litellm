import React, { useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { MinusCircle, Plus } from "lucide-react";

interface QueryParamInputProps {
  value?: Record<string, string>;
  onChange?: (value: Record<string, string>) => void;
}

const QueryParamInput: React.FC<QueryParamInputProps> = ({
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
    <div className="space-y-2">
      {pairs.map(([key, val], index) => (
        <div key={index} className="flex items-center gap-2">
          <Input
            placeholder="Parameter Name (e.g., version)"
            value={key}
            onChange={(e) => handleChange(index, e.target.value, val)}
          />
          <Input
            placeholder="Parameter Value (e.g., v1)"
            value={val}
            onChange={(e) => handleChange(index, key, e.target.value)}
          />
          <button
            type="button"
            onClick={() => handleRemove(index)}
            className="shrink-0 text-muted-foreground hover:text-destructive"
            aria-label={`Remove query parameter ${index + 1}`}
          >
            <MinusCircle className="h-4 w-4" />
          </button>
        </div>
      ))}
      <Button
        type="button"
        variant="outline"
        onClick={handleAdd}
        className="border-dashed"
      >
        <Plus className="h-4 w-4" />
        Add Query Parameter
      </Button>
    </div>
  );
};

export default QueryParamInput;
