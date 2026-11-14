import React from "react";
import { useTranslation } from "react-i18next";
import { Card, CardContent } from "@/components/ui/card";
import VariableTextArea from "../variable_textarea";

interface DeveloperMessageCardProps {
  value: string;
  onChange: (value: string) => void;
}

const DeveloperMessageCard: React.FC<DeveloperMessageCardProps> = ({ value, onChange }) => {
  const { t } = useTranslation();
  return (
    <Card>
      <CardContent className="p-3">
        <p className="mb-2 text-sm font-medium text-foreground">{t("prompts.developerMessage.title")}</p>
        <p className="mb-2 text-xs text-muted-foreground">{t("prompts.developerMessage.description")}</p>
        <VariableTextArea
          value={value}
          onChange={onChange}
          rows={3}
          placeholder={t("prompts.developerMessage.placeholder")}
        />
      </CardContent>
    </Card>
  );
};

export default DeveloperMessageCard;
