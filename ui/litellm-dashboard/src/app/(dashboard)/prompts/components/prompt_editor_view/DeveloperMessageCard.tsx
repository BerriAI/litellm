import React from "react";
import { useTranslation } from "react-i18next";
import { Card, Text } from "@tremor/react";
import VariableTextArea from "../variable_textarea";

interface DeveloperMessageCardProps {
  value: string;
  onChange: (value: string) => void;
}

const DeveloperMessageCard: React.FC<DeveloperMessageCardProps> = ({ value, onChange }) => {
  const { t } = useTranslation();
  return (
    <Card className="p-3">
      <Text className="block mb-2 text-sm font-medium">{t("promptsPage.developerMessageCard.title")}</Text>
      <Text className="text-gray-500 text-xs mb-2">{t("promptsPage.developerMessageCard.subtitle")}</Text>
      <VariableTextArea
        value={value}
        onChange={onChange}
        rows={3}
        placeholder={t("promptsPage.developerMessageCard.placeholder")}
      />
    </Card>
  );
};

export default DeveloperMessageCard;
