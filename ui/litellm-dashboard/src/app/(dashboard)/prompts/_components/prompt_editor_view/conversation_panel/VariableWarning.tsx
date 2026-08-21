import React from "react";

interface VariableWarningProps {
  extractedVariables: string[];
  variables: Record<string, string>;
}

const VariableWarning: React.FC<VariableWarningProps> = ({ extractedVariables, variables }) => {
  const missingVariables = extractedVariables.filter(
    (varName) => !variables[varName] || variables[varName].trim() === "",
  );

  if (missingVariables.length === 0) {
    return null;
  }

  return (
    <div className="mb-3 p-3 bg-warning/10 border border-warning/20 rounded-lg">
      <div className="flex items-start gap-2">
        <span className="text-warning text-sm">⚠️</span>
        <div className="flex-1">
          <p className="text-sm text-warning font-medium mb-1">Please fill in all template variables above</p>
          <p className="text-xs text-warning">
            Missing: {missingVariables.map((varName) => `{{${varName}}}`).join(", ")}
          </p>
        </div>
      </div>
    </div>
  );
};

export default VariableWarning;
