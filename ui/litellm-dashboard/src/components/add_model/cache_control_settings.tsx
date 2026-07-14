import React from "react";
import { Form, Switch, Typography } from "antd";
import CacheControlInjectionPointsEditor, {
  CacheControlInjectionPoint,
} from "../shared/cache_control_injection_points_editor";

const { Text } = Typography;

interface CacheControlSettingsProps {
  form: any; // Form instance from parent
  showCacheControl: boolean;
  onCacheControlChange: (checked: boolean) => void;
}

const CacheControlSettings: React.FC<CacheControlSettingsProps> = ({
  form,
  showCacheControl,
  onCacheControlChange,
}) => {
  const updateCacheControlPoints = (injectionPoints: CacheControlInjectionPoint[]) => {
    form.setFieldValue("cache_control_injection_points", injectionPoints);

    const currentParams = form.getFieldValue("litellm_extra_params");
    try {
      const paramsObj = currentParams ? JSON.parse(currentParams) : {};
      if (injectionPoints.length > 0) {
        paramsObj.cache_control_injection_points = injectionPoints;
      } else {
        delete paramsObj.cache_control_injection_points;
      }
      form.setFieldValue(
        "litellm_extra_params",
        Object.keys(paramsObj).length > 0 ? JSON.stringify(paramsObj, null, 2) : "",
      );
    } catch (error) {
      console.error("Error updating cache control points:", error);
    }
  };

  const cacheControlInjectionPoints = Form.useWatch("cache_control_injection_points", form) || [
    { location: "message" as const },
  ];

  return (
    <>
      <Form.Item
        label="Cache Control Injection Points"
        name="cache_control"
        valuePropName="checked"
        className="mb-4"
        tooltip="Tell litellm where to inject cache control checkpoints. You can specify either by role (to apply to all messages of that role) or by specific message index."
      >
        <Switch onChange={onCacheControlChange} className="bg-gray-600" />
      </Form.Item>

      {showCacheControl && (
        <div className="ml-6 pl-4 border-l-2 border-gray-200">
          <Text className="text-sm text-gray-500 block mb-4">
            Providers like Anthropic, Bedrock API require users to specify where to inject cache control checkpoints,
            litellm can automatically add them for you as a cost saving feature.
          </Text>

          <CacheControlInjectionPointsEditor value={cacheControlInjectionPoints} onChange={updateCacheControlPoints} />
        </div>
      )}
    </>
  );
};

export default CacheControlSettings;
