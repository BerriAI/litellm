import React from "react";
import { Select } from "antd";
import {
  CACHE_MODES,
  CACHE_MODE_DESCRIPTIONS,
  CACHE_MODE_LABELS,
  CacheMode,
  REDIS_DEPLOYMENT_DESCRIPTIONS,
  REDIS_DEPLOYMENT_LABELS,
  REDIS_DEPLOYMENT_TYPES,
  RedisDeploymentType,
} from "./cacheSettingsFields";

interface CacheTypeSelectorProps {
  cacheMode: CacheMode;
  deploymentType: RedisDeploymentType;
  onCacheModeChange: (mode: CacheMode) => void;
  onDeploymentTypeChange: (type: RedisDeploymentType) => void;
}

const cacheModeOptions = CACHE_MODES.map((mode) => ({ value: mode, label: CACHE_MODE_LABELS[mode] }));
const deploymentTypeOptions = REDIS_DEPLOYMENT_TYPES.map((type) => ({
  value: type,
  label: REDIS_DEPLOYMENT_LABELS[type],
}));

const CacheTypeSelector: React.FC<CacheTypeSelectorProps> = ({
  cacheMode,
  deploymentType,
  onCacheModeChange,
  onDeploymentTypeChange,
}) => {
  return (
    <div className="space-y-6">
      <div className="space-y-2">
        <label className="text-sm font-medium text-gray-700">Cache Type</label>
        <Select<CacheMode>
          value={cacheMode}
          onChange={(value) => onCacheModeChange(value)}
          options={cacheModeOptions}
          style={{ width: "100%" }}
        />
        <p className="text-xs text-gray-500">{CACHE_MODE_DESCRIPTIONS[cacheMode]}</p>
      </div>

      {cacheMode === "standard" && (
        <div className="space-y-2">
          <label className="text-sm font-medium text-gray-700">Redis Deployment Type</label>
          <Select<RedisDeploymentType>
            value={deploymentType}
            onChange={(value) => onDeploymentTypeChange(value)}
            options={deploymentTypeOptions}
            style={{ width: "100%" }}
          />
          <p className="text-xs text-gray-500">{REDIS_DEPLOYMENT_DESCRIPTIONS[deploymentType]}</p>
        </div>
      )}
    </div>
  );
};

export default CacheTypeSelector;
