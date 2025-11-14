import React from "react";
import { Form, Switch, Select, Typography } from "antd";
import { PlusOutlined, MinusCircleOutlined } from "@ant-design/icons";
import NumericalInput from "../shared/numerical_input";

const { Text } = Typography;

interface CacheControlInjectionPoint {
  location: "message";
  role?: "user" | "system" | "assistant";
  index?: number;
}

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
    const currentParams = form.getFieldValue("litellm_extra_params");
    try {
      let paramsObj = currentParams ? JSON.parse(currentParams) : {};
      if (injectionPoints.length > 0) {
        paramsObj.cache_control_injection_points = injectionPoints;
      } else {
        delete paramsObj.cache_control_injection_points;
      }
      if (Object.keys(paramsObj).length > 0) {
        form.setFieldValue("litellm_extra_params", JSON.stringify(paramsObj, null, 2));
      } else {
        form.setFieldValue("litellm_extra_params", "");
      }
    } catch (error) {
      console.error("Error updating cache control points:", error);
    }
  };

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

          <Form.List name="cache_control_injection_points" initialValue={[{ location: "message" }]}>
            {(fields, { add, remove }) => (
              <>
                {fields.map((field, index) => (
                  <div key={field.key} className="flex items-center mb-4 gap-4">
                    <Form.Item
                      {...field}
                      label="Type"
                      name={[field.name, "location"]}
                      initialValue="message"
                      className="mb-0"
                      style={{ width: "180px" }}
                    >
                      <Select disabled options={[{ value: "message", label: "Message" }]} />
                    </Form.Item>

                    <Form.Item
                      {...field}
                      label="Role"
                      name={[field.name, "role"]}
                      className="mb-0"
                      style={{ width: "180px" }}
                      tooltip="LiteLLM will mark all messages of this role as cacheable"
                    >
                      <Select
                        placeholder="Select a role"
                        allowClear
                        options={[
                          { value: "user", label: "User" },
                          { value: "system", label: "System" },
                          { value: "assistant", label: "Assistant" },
                        ]}
                        onChange={() => {
                          const values = form.getFieldValue("cache_control_points");
                          updateCacheControlPoints(values);
                        }}
                      />
                    </Form.Item>

                    <Form.Item
                      {...field}
                      label="Index"
                      name={[field.name, "index"]}
                      className="mb-0"
                      style={{ width: "180px" }}
                      tooltip="(Optional) If set litellm will mark the message at this index as cacheable"
                    >
                      <NumericalInput
                        type="number"
                        placeholder="Optional"
                        step={1}
                        onChange={() => {
                          const values = form.getFieldValue("cache_control_points");
                          updateCacheControlPoints(values);
                        }}
                      />
                    </Form.Item>

                    {fields.length > 1 && (
                      <MinusCircleOutlined
                        className="text-red-500 cursor-pointer text-lg ml-12"
                        onClick={() => {
                          remove(field.name);
                          setTimeout(() => {
                            const values = form.getFieldValue("cache_control_points");
                            updateCacheControlPoints(values);
                          }, 0);
                        }}
                      />
                    )}
                  </div>
                ))}

                <Form.Item>
                  <button
                    type="button"
                    className="flex items-center justify-center w-full border border-dashed border-gray-300 py-2 px-4 text-gray-600 hover:text-blue-600 hover:border-blue-300 transition-all rounded"
                    onClick={() => add()}
                  >
                    <PlusOutlined className="mr-2" />
                    Add Injection Point
                  </button>
                </Form.Item>
              </>
            )}
          </Form.List>
        </div>
      )}
    </>
  );
};

export default CacheControlSettings;
