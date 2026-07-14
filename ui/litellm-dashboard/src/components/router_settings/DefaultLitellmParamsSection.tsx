import React from "react";
import { Input, Switch } from "antd";
import CacheControlInjectionPointsEditor, {
  CacheControlInjectionPoint,
} from "../shared/cache_control_injection_points_editor";

interface DefaultLitellmParamsSectionProps {
  value: { [key: string]: any };
  routerFieldsMetadata: { [key: string]: any };
  onChange: (value: { [key: string]: any }) => void;
}

const DefaultLitellmParamsSection: React.FC<DefaultLitellmParamsSectionProps> = ({
  value,
  routerFieldsMetadata,
  onChange,
}) => {
  const meta = routerFieldsMetadata["default_litellm_params"];
  const { cache_control_injection_points, ...otherParams } = value || {};

  const [otherParamsText, setOtherParamsText] = React.useState(() => JSON.stringify(otherParams, null, 2));
  const [showCacheControl, setShowCacheControl] = React.useState((cache_control_injection_points?.length ?? 0) > 0);

  const parseOtherParams = (): { [key: string]: any } => {
    try {
      return JSON.parse(otherParamsText || "{}");
    } catch {
      return otherParams;
    }
  };

  const handleOtherParamsBlur = () => {
    try {
      const parsed = JSON.parse(otherParamsText || "{}");
      onChange({ ...parsed, cache_control_injection_points });
    } catch (error) {
      console.error("Error parsing default_litellm_params JSON:", error);
    }
  };

  const handleCacheControlPointsChange = (points: CacheControlInjectionPoint[]) => {
    const base = parseOtherParams();
    onChange(points.length > 0 ? { ...base, cache_control_injection_points: points } : base);
  };

  const handleCacheControlToggle = (checked: boolean) => {
    setShowCacheControl(checked);
    handleCacheControlPointsChange(checked ? [{ location: "message" }] : []);
  };

  return (
    <div className="space-y-6">
      <div className="max-w-3xl space-y-2">
        <span className="text-xs font-medium text-gray-700 uppercase tracking-wide">
          {meta?.ui_field_name || "default_litellm_params"}
        </span>
        <p className="text-xs text-gray-500 mt-0.5 mb-2">{meta?.field_description || ""}</p>
        <Input.TextArea
          value={otherParamsText}
          onChange={(e) => setOtherParamsText(e.target.value)}
          onBlur={handleOtherParamsBlur}
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
              value={cache_control_injection_points || [{ location: "message" }]}
              onChange={handleCacheControlPointsChange}
            />
          </div>
        )}
      </div>
    </div>
  );
};

export default DefaultLitellmParamsSection;
