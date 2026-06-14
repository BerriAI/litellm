import React from "react";
import { Select, SelectItem } from "@tremor/react";
import { useTranslation } from "react-i18next";

interface RedisTypeSelectorProps {
  redisType: string;
  redisTypeDescriptions: { [key: string]: string };
  onTypeChange: (type: string) => void;
}

const RedisTypeSelector: React.FC<RedisTypeSelectorProps> = ({ redisType, redisTypeDescriptions, onTypeChange }) => {
  const { t } = useTranslation();
  return (
    <div className="space-y-2">
      <label className="text-sm font-medium text-gray-700">{t("cacheSettings.redisTypeSelector.redisType")}</label>
      <Select value={redisType} onValueChange={onTypeChange}>
        <SelectItem value="node">{t("cacheSettings.redisTypeSelector.nodeType")}</SelectItem>
        <SelectItem value="cluster">{t("cacheSettings.redisTypeSelector.clusterType")}</SelectItem>
        <SelectItem value="sentinel">{t("cacheSettings.redisTypeSelector.sentinelType")}</SelectItem>
        <SelectItem value="semantic">{t("cacheSettings.redisTypeSelector.semanticType")}</SelectItem>
      </Select>
      <p className="text-xs text-gray-500">
        {redisTypeDescriptions[redisType] || t("cacheSettings.redisTypeSelector.selectTypeDescription")}
      </p>
    </div>
  );
};

export default RedisTypeSelector;
