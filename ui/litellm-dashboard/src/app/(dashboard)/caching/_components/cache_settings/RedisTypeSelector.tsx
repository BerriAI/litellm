import { useLanguage } from "@/contexts/LanguageContext";
import React from "react";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";

interface RedisTypeSelectorProps {
  redisType: string;
  redisTypeDescriptions: Readonly<Record<string, string>>;
  onTypeChange: (type: string) => void;
}

const REDIS_TYPE_LABELS: Readonly<Record<string, string>> = {
  node: "Node (Single Instance)",
  cluster: "Cluster",
  sentinel: "Sentinel",
  semantic: "Semantic",
};

const RedisTypeSelector: React.FC<RedisTypeSelectorProps> = ({ redisType, redisTypeDescriptions, onTypeChange }) => {
  const { t } = useLanguage();
  return (
    <div className="space-y-2">
      <label className="text-sm font-medium">{t("Redis Type")}</label>
      <Select value={redisType} onValueChange={(value) => value !== null && onTypeChange(value)}>
        <SelectTrigger className="w-full">
          <SelectValue>{REDIS_TYPE_LABELS[redisType] ? t(REDIS_TYPE_LABELS[redisType]) : redisType}</SelectValue>
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
        {t(redisTypeDescriptions[redisType] || t("Select the type of Redis deployment you're using"))}
      </p>
    </div>
  );
};

export default RedisTypeSelector;
