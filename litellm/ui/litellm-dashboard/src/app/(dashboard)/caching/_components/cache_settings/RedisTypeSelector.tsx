import type { ParseKeys } from "i18next";
import React from "react";
import { useTranslation } from "react-i18next";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";

interface RedisTypeSelectorProps {
  redisType: string;
  redisTypeDescriptions: Readonly<Record<string, ParseKeys>>;
  onTypeChange: (type: string) => void;
}

const REDIS_TYPE_LABELS: Readonly<Record<string, ParseKeys>> = {
  node: "caching.settings.redisType.options.node",
  cluster: "caching.settings.redisType.options.cluster",
  sentinel: "caching.settings.redisType.options.sentinel",
  semantic: "caching.settings.redisType.options.semantic",
};

const RedisTypeSelector: React.FC<RedisTypeSelectorProps> = ({ redisType, redisTypeDescriptions, onTypeChange }) => {
  const { t } = useTranslation();
  return (
    <div className="space-y-2">
      <label className="text-sm font-medium">{t("caching.settings.redisType.label")}</label>
      <Select value={redisType} onValueChange={(value) => value !== null && onTypeChange(value)}>
        <SelectTrigger className="w-full">
          <SelectValue>
            {Object.hasOwn(REDIS_TYPE_LABELS, redisType) ? t(REDIS_TYPE_LABELS[redisType]) : redisType}
          </SelectValue>
        </SelectTrigger>
        <SelectContent>
          {Object.entries(REDIS_TYPE_LABELS).map(([value, label]) => (
            <SelectItem key={value} value={value}>
              {t(label)}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
      <p className="text-xs text-muted-foreground">
        {t(
          (Object.hasOwn(redisTypeDescriptions, redisType) && redisTypeDescriptions[redisType]) ||
            "caching.settings.redisType.fallbackDescription",
        )}
      </p>
    </div>
  );
};

export default RedisTypeSelector;
