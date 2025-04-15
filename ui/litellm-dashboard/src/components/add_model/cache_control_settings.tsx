import React from "react";
import { Form, Switch, Select, Input, Button } from "antd";
import { MinusCircleOutlined, PlusOutlined } from '@ant-design/icons';

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
    const currentParams = form.getFieldValue('litellm_extra_params');
    try {
      let paramsObj = currentParams ? JSON.parse(currentParams) : {};
      if (injectionPoints.length > 0) {
        paramsObj.cache_control_injection_points = injectionPoints;
      } else {
        delete paramsObj.cache_control_injection_points;
      }
      if (Object.keys(paramsObj).length > 0) {
        form.setFieldValue('litellm_extra_params', JSON.stringify(paramsObj, null, 2));
      } else {
        form.setFieldValue('litellm_extra_params', '');
      }
    } catch (error) {
      console.error('Error updating cache control points:', error);
    }
  };

  return (
    <>
      <Form.Item
        label="Cache Control Injection Points"
        name="cache_control"
        valuePropName="checked"
        className="mb-4"
        tooltip="Tell litellm where to inject cache control checkpoints. This is useful for reducing costs by caching responses."
      >
        <Switch onChange={onCacheControlChange} className="bg-gray-600" />
      </Form.Item>

      {showCacheControl && (
        <div className="ml-6 pl-4 border-l-2 border-gray-200">
          <Form.List
            name="cache_control_points"
            initialValue={[{ location: "message" }]}
          >
            {(fields, { add, remove }) => (
              <>
                {fields.map((field, index) => (
                  <div key={field.key} style={{ display: 'flex', marginBottom: 8, gap: 8, alignItems: 'baseline' }}>
                    <Form.Item
                      {...field}
                      name={[field.name, 'location']}
                      initialValue="message"
                      style={{ marginBottom: 0, width: '120px' }}
                    >
                      <Select disabled options={[{ value: 'message', label: 'Message' }]} />
                    </Form.Item>
                    <Form.Item
                      {...field}
                      name={[field.name, 'role']}
                      style={{ marginBottom: 0, width: '120px' }}
                    >
                      <Select
                        placeholder="Select role"
                        allowClear
                        options={[
                          { value: 'user', label: 'User' },
                          { value: 'system', label: 'System' },
                          { value: 'assistant', label: 'Assistant' },
                        ]}
                        onChange={() => {
                          const values = form.getFieldValue('cache_control_points');
                          updateCacheControlPoints(values);
                        }}
                      />
                    </Form.Item>
                    <Form.Item
                      {...field}
                      name={[field.name, 'index']}
                      style={{ marginBottom: 0, width: '120px' }}
                    >
                      <Input
                        type="number"
                        placeholder="Index (optional)"
                        onChange={() => {
                          const values = form.getFieldValue('cache_control_points');
                          updateCacheControlPoints(values);
                        }}
                      />
                    </Form.Item>
                    {fields.length > 1 && (
                      <MinusCircleOutlined onClick={() => {
                        remove(field.name);
                        setTimeout(() => {
                          const values = form.getFieldValue('cache_control_points');
                          updateCacheControlPoints(values);
                        }, 0);
                      }} />
                    )}
                  </div>
                ))}
                <Form.Item style={{ marginTop: 8 }}>
                  <Button
                    type="dashed"
                    onClick={() => add()}
                    icon={<PlusOutlined />}
                  >
                    Add Injection Point
                  </Button>
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