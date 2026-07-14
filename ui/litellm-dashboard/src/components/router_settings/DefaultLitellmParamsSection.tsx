import React from "react";
import { Input, Switch } from "antd";
import CacheControlInjectionPointsEditor, {
  CacheControlInjectionPoint,
} from "../shared/cache_control_injection_points_editor";
import NotificationsManager from "../molecules/notifications_manager";

interface DefaultLitellmParamsSectionProps {
  value: Record<string, unknown>;
  onChange: (value: Record<string, unknown>) => void;
}

type ParsedDefaultParams = { status: "valid"; value: Record<string, unknown> } | { status: "invalid"; message: string };

const CACHE_CONTROL_ROLES = ["user", "system", "assistant"] as const;
type CacheControlRole = (typeof CACHE_CONTROL_ROLES)[number];

const parseDefaultParams = (text: string): ParsedDefaultParams => {
  try {
    const parsed: unknown = JSON.parse(text || "{}");
    if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) {
      return { status: "invalid", message: "Expected a JSON object" };
    }
    return { status: "valid", value: parsed as Record<string, unknown> };
  } catch (error) {
    return { status: "invalid", message: error instanceof Error ? error.message : "Invalid JSON" };
  }
};

const isCacheControlRole = (value: unknown): value is CacheControlRole =>
  CACHE_CONTROL_ROLES.some((validRole) => validRole === value);

const parseCacheControlInjectionPoint = (value: unknown): CacheControlInjectionPoint | undefined => {
  if (typeof value !== "object" || value === null) {
    return undefined;
  }
  if (!("location" in value) || value.location !== "message") {
    return undefined;
  }
  const role = "role" in value ? value.role : undefined;
  const index = "index" in value ? value.index : undefined;
  const hasInvalidRole = role !== undefined && role !== null && !isCacheControlRole(role);
  if (hasInvalidRole) {
    return undefined;
  }
  const hasIndex = index !== undefined && index !== null;
  const hasInvalidIndex = hasIndex && (typeof index !== "number" || !Number.isInteger(index));
  if (hasInvalidIndex) {
    return undefined;
  }
  return {
    location: "message",
    ...(isCacheControlRole(role) ? { role } : {}),
    ...(typeof index === "number" ? { index } : {}),
  };
};

const parseCacheControlInjectionPoints = (value: unknown): CacheControlInjectionPoint[] | undefined => {
  if (!Array.isArray(value)) {
    return undefined;
  }
  const points = value.map(parseCacheControlInjectionPoint);
  return points.every((point) => point !== undefined)
    ? points.filter((point): point is CacheControlInjectionPoint => point !== undefined)
    : undefined;
};

const DefaultLitellmParamsSection: React.FC<DefaultLitellmParamsSectionProps> = ({ value, onChange }) => {
  const cacheControlInjectionPoints = React.useMemo(
    () => parseCacheControlInjectionPoints(value.cache_control_injection_points),
    [value.cache_control_injection_points],
  );
  const otherParams = React.useMemo(
    () =>
      Object.fromEntries(
        Object.entries(value).filter(
          ([key]) => key !== "cache_control_injection_points" || cacheControlInjectionPoints === undefined,
        ),
      ),
    [value, cacheControlInjectionPoints],
  );

  const [otherParamsText, setOtherParamsText] = React.useState(() => JSON.stringify(otherParams, null, 2));
  const [hasInvalidJson, setHasInvalidJson] = React.useState(false);
  const showCacheControl = (cacheControlInjectionPoints?.length ?? 0) > 0;

  const handleOtherParamsBlur = () => {
    const parsed = parseDefaultParams(otherParamsText);
    if (parsed.status === "invalid") {
      setHasInvalidJson(true);
      NotificationsManager.warning(`Default LiteLLM Params is not valid JSON: ${parsed.message}`);
      return;
    }
    setHasInvalidJson(false);
    onChange(
      cacheControlInjectionPoints
        ? { ...parsed.value, cache_control_injection_points: cacheControlInjectionPoints }
        : parsed.value,
    );
  };

  const handleCacheControlPointsChange = (points: CacheControlInjectionPoint[]) => {
    const parsed = parseDefaultParams(otherParamsText);
    const base = parsed.status === "valid" ? parsed.value : otherParams;
    onChange(points.length > 0 ? { ...base, cache_control_injection_points: points } : base);
  };

  const handleCacheControlToggle = (checked: boolean) => {
    handleCacheControlPointsChange(checked ? [{ location: "message" }] : []);
  };

  return (
    <div className="space-y-6">
      <div className="max-w-3xl space-y-2">
        <span className="text-xs font-medium text-gray-700 uppercase tracking-wide">Default LiteLLM Params</span>
        <p className="text-xs text-gray-500 mt-0.5 mb-2">
          Default parameters for Router.chat.completion.create. Set cache control injection points to enable prompt
          caching for every model on this proxy.{" "}
          <a
            href="https://docs.litellm.ai/docs/tutorials/claude_code_prompt_cache_routing"
            target="_blank"
            rel="noopener noreferrer"
            className="text-blue-600 hover:text-blue-800 underline"
          >
            Learn more
          </a>
        </p>
        <Input.TextArea
          value={otherParamsText}
          onChange={(e) => {
            setOtherParamsText(e.target.value);
            setHasInvalidJson(false);
          }}
          onBlur={handleOtherParamsBlur}
          status={hasInvalidJson ? "error" : undefined}
          autoSize={{ minRows: 2 }}
          className="font-mono text-sm w-full"
        />
      </div>

      <div className="max-w-3xl">
        <div className="flex items-center justify-between mb-2">
          <span className="text-xs font-medium text-gray-700 uppercase tracking-wide">
            Cache Control Injection Points
          </span>
          <Switch
            checked={showCacheControl}
            onChange={handleCacheControlToggle}
            className="bg-gray-600"
            aria-label="Cache Control Injection Points"
          />
        </div>
        {showCacheControl && (
          <div className="ml-6 pl-4 border-l-2 border-gray-200">
            <CacheControlInjectionPointsEditor
              value={cacheControlInjectionPoints || [{ location: "message" }]}
              onChange={handleCacheControlPointsChange}
            />
          </div>
        )}
      </div>
    </div>
  );
};

export default DefaultLitellmParamsSection;
