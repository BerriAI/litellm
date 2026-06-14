import React from "react";
import { Input } from "antd";
import { useTranslation } from "react-i18next";

interface VariableInputProps {
  extractedVariables: string[];
  variables: Record<string, string>;
  onVariableChange: (varName: string, value: string) => void;
}

const VariableInput: React.FC<VariableInputProps> = ({ extractedVariables, variables, onVariableChange }) => {
  const { t } = useTranslation();
  if (extractedVariables.length === 0) {
    return null;
  }

  return (
    <div className="p-4 border-b border-gray-200 bg-blue-50">
      <h3 className="text-sm font-semibold text-gray-700 mb-3">{t("promptsPage.variableInput.title")}</h3>
      <div className="space-y-2">
        {extractedVariables.map((varName) => (
          <div key={varName}>
            <label className="block text-xs text-gray-600 mb-1 font-medium">
              {"{{"}
              {varName}
              {"}}"}
            </label>
            <Input
              value={variables[varName] || ""}
              onChange={(e) => onVariableChange(varName, e.target.value)}
              placeholder={t("promptsPage.variableInput.valuePlaceholder", { varName })}
              size="small"
            />
          </div>
        ))}
      </div>
    </div>
  );
};

export default VariableInput;
