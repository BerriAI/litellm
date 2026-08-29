import React from "react";
import { Input } from "@/components/ui/input";

interface VariableInputProps {
  extractedVariables: string[];
  variables: Record<string, string>;
  onVariableChange: (varName: string, value: string) => void;
}

const VariableInput: React.FC<VariableInputProps> = ({ extractedVariables, variables, onVariableChange }) => {
  if (extractedVariables.length === 0) {
    return null;
  }

  return (
    <div className="p-4 border-b border-border bg-accent">
      <h3 className="text-sm font-semibold text-foreground mb-3">Fill in template variables to start testing</h3>
      <div className="space-y-2">
        {extractedVariables.map((varName) => (
          <div key={varName}>
            <label className="block text-xs text-muted-foreground mb-1 font-medium">
              {"{{"}
              {varName}
              {"}}"}
            </label>
            <Input
              value={variables[varName] || ""}
              onChange={(e) => onVariableChange(varName, e.target.value)}
              placeholder={`Enter value for ${varName}`}
            />
          </div>
        ))}
      </div>
    </div>
  );
};

export default VariableInput;
