import React from "react";
import { Select } from "antd";
import {
  COORDINATION_REDIS_TYPES,
  COORDINATION_REDIS_TYPE_DESCRIPTIONS,
  COORDINATION_REDIS_TYPE_LABELS,
  CoordinationRedisType,
} from "./coordinationRedisFields";

interface CoordinationRedisTypeSelectorProps {
  redisType: CoordinationRedisType;
  onTypeChange: (type: CoordinationRedisType) => void;
}

const OPTIONS = COORDINATION_REDIS_TYPES.map((type) => ({ value: type, label: COORDINATION_REDIS_TYPE_LABELS[type] }));

const CoordinationRedisTypeSelector: React.FC<CoordinationRedisTypeSelectorProps> = ({ redisType, onTypeChange }) => (
  <div className="space-y-2">
    <label htmlFor="coordination-redis-type" className="text-sm font-medium text-gray-700">
      Redis Type
    </label>
    <Select
      id="coordination-redis-type"
      value={redisType}
      onChange={onTypeChange}
      options={OPTIONS}
      style={{ width: "100%" }}
    />
    <p className="text-xs text-gray-500">{COORDINATION_REDIS_TYPE_DESCRIPTIONS[redisType]}</p>
  </div>
);

export default CoordinationRedisTypeSelector;
