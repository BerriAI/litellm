import React from "react";
import { useTranslation } from "react-i18next";

interface VariableWarningProps {
  extractedVariables: string[];
  variables: Record<string, string>;
}

const VariableWarning: React.FC<VariableWarningProps> = ({ extractedVariables, variables }) => {
  const { t } = useTranslation();
  const missingVariables = extractedVariables.filter(
    (varName) => !variables[varName] || variables[varName].trim() === "",
  );

  if (missingVariables.length === 0) {
    return null;
  }

  return (
    <div className="mb-3 p-3 bg-yellow-50 border border-yellow-200 rounded-lg">
      <div className="flex items-start gap-2">
        <span className="text-yellow-600 text-sm">⚠️</span>
        <div className="flex-1">
          <p className="text-sm text-yellow-800 font-medium mb-1">{t("promptsPage.variableWarning.title")}</p>
          <p className="text-xs text-yellow-700">
            {t("promptsPage.variableWarning.missing", {
              list: missingVariables.map((varName) => `{{${varName}}}`).join(", "),
            })}
          </p>
        </div>
      </div>
    </div>
  );
};

export default VariableWarning;
