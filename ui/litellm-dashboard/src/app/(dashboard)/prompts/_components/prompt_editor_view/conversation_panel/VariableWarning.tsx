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
    <div className="mb-3 p-3 bg-warning/10 border border-warning/20 rounded-lg">
      <div className="flex items-start gap-2">
        <span className="text-warning text-sm">⚠️</span>
        <div className="flex-1">
          <p className="text-sm text-warning font-medium mb-1">{t("prompts.conversation.warning.title")}</p>
          <p className="text-xs text-warning">
            {t("prompts.conversation.warning.missing", {
              variables: missingVariables.map((varName) => `{{${varName}}}`).join(", "),
            })}
          </p>
        </div>
      </div>
    </div>
  );
};

export default VariableWarning;
