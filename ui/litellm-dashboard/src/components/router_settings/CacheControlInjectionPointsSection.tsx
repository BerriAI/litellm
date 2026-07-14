import React from "react";
import { Switch } from "antd";
import CacheControlInjectionPointsEditor, {
  CacheControlInjectionPoint,
} from "../shared/cache_control_injection_points_editor";

interface CacheControlInjectionPointsSectionProps {
  value: Record<string, unknown>;
  onChange: (value: Record<string, unknown>) => void;
}

const CACHE_CONTROL_ROLES = ["user", "system", "assistant"] as const;
type CacheControlRole = (typeof CACHE_CONTROL_ROLES)[number];

const isCacheControlRole = (value: unknown): value is CacheControlRole =>
  CACHE_CONTROL_ROLES.some((role) => role === value);

const parseInjectionPoint = (value: unknown): CacheControlInjectionPoint | undefined => {
  if (typeof value !== "object" || value === null) {
    return undefined;
  }
  if (!("location" in value) || value.location !== "message") {
    return undefined;
  }

  const role = "role" in value ? value.role : undefined;
  const index = "index" in value ? value.index : undefined;
  const hasRole = role !== undefined && role !== null;
  if (hasRole && !isCacheControlRole(role)) {
    return undefined;
  }
  const hasIndex = index !== undefined && index !== null;
  if (hasIndex && typeof index !== "number") {
    return undefined;
  }
  if (typeof index === "number" && !Number.isInteger(index)) {
    return undefined;
  }

  return {
    location: "message",
    ...(isCacheControlRole(role) ? { role } : {}),
    ...(typeof index === "number" ? { index } : {}),
  };
};

const parseInjectionPoints = (value: unknown): CacheControlInjectionPoint[] | undefined => {
  if (value === undefined) {
    return [];
  }
  if (!Array.isArray(value)) {
    return undefined;
  }

  const points = value.map(parseInjectionPoint);
  return points.every((point) => point !== undefined)
    ? points.filter((point): point is CacheControlInjectionPoint => point !== undefined)
    : undefined;
};

const CacheControlInjectionPointsSection: React.FC<CacheControlInjectionPointsSectionProps> = ({ value, onChange }) => {
  const injectionPoints = React.useMemo(
    () => parseInjectionPoints(value.cache_control_injection_points),
    [value.cache_control_injection_points],
  );
  const enabled = injectionPoints !== undefined && injectionPoints.length > 0;

  const handlePointsChange = (points: CacheControlInjectionPoint[]) => {
    onChange({ ...value, cache_control_injection_points: points });
  };

  const handleToggle = (checked: boolean) => {
    if (checked) {
      handlePointsChange([{ location: "message" }]);
      return;
    }

    const nextValue = Object.fromEntries(
      Object.entries(value).filter(([key]) => key !== "cache_control_injection_points"),
    );
    onChange(nextValue);
  };

  return (
    <div className="space-y-4 max-w-3xl">
      <div className="flex items-center justify-between gap-4">
        <div>
          <span className="text-xs font-medium text-gray-700 uppercase tracking-wide">
            Cache Control Injection Points
          </span>
          <p className="text-xs text-gray-500 mt-0.5">
            Choose message locations where LiteLLM should inject cache-control markers.{" "}
            <a
              href="https://docs.litellm.ai/docs/tutorials/claude_code_prompt_cache_routing"
              target="_blank"
              rel="noopener noreferrer"
              className="text-blue-600 hover:text-blue-800 underline"
            >
              Learn more
            </a>
          </p>
        </div>
        <Switch
          checked={enabled}
          disabled={injectionPoints === undefined}
          onChange={handleToggle}
          aria-label="Cache Control Injection Points"
        />
      </div>

      {injectionPoints === undefined && (
        <p className="text-xs text-amber-700">
          The configured injection points use values this editor does not support and will be preserved unchanged.
        </p>
      )}

      {enabled && <CacheControlInjectionPointsEditor value={injectionPoints} onChange={handlePointsChange} />}
    </div>
  );
};

export default CacheControlInjectionPointsSection;
