/* eslint-disable @next/next/no-img-element */
/* eslint-disable react/no-unescaped-entities */
import React, { useState } from 'react';
import { Form, Select, Space, Tooltip, Divider } from 'antd';
import { InfoCircleOutlined } from '@ant-design/icons';
import { Button, Card, TextInput } from '@tremor/react';
import { PlusIcon, TrashIcon, CogIcon } from '@heroicons/react/outline';
import { callbackInfo, Callbacks, callback_map } from '../callback_info_helpers';

const { Option } = Select;

interface LoggingConfig {
  callback_name: string;
  callback_type: string;
  callback_vars: Record<string, string>;
}

interface LoggingSettingsProps {
  value?: LoggingConfig[];
  onChange?: (value: LoggingConfig[]) => void;
}

const LoggingSettings: React.FC<LoggingSettingsProps> = ({ value = [], onChange }) => {
  // Get callbacks that support team and key logging
  const supportedCallbacks = Object.entries(callbackInfo)
    .filter(([_, info]) => info.supports_key_team_logging)
    .map(([name, _]) => name);

  const handleChange = (newValue: LoggingConfig[]) => {
    onChange?.(newValue);
  };

  const addLoggingConfig = () => {
    const newConfig: LoggingConfig = {
      callback_name: '',
      callback_type: 'success',
      callback_vars: {}
    };
    handleChange([...value, newConfig]);
  };

  const removeLoggingConfig = (index: number) => {
    const newValue = value.filter((_, i) => i !== index);
    handleChange(newValue);
  };

  const updateLoggingConfig = (index: number, field: keyof LoggingConfig, newValue: any) => {
    const updatedConfigs = [...value];
    if (field === 'callback_name') {
      // Convert display name to callback value and reset callback_vars when callback changes
      const callbackValue = callback_map[newValue] || newValue;
      updatedConfigs[index] = {
        ...updatedConfigs[index],
        [field]: callbackValue,
        callback_vars: {}
      };
    } else {
      updatedConfigs[index] = {
        ...updatedConfigs[index],
        [field]: newValue
      };
    }
    handleChange(updatedConfigs);
  };

  const updateCallbackVar = (configIndex: number, varName: string, varValue: string) => {
    const updatedConfigs = [...value];
    updatedConfigs[configIndex] = {
      ...updatedConfigs[configIndex],
      callback_vars: {
        ...updatedConfigs[configIndex].callback_vars,
        [varName]: varValue
      }
    };
    handleChange(updatedConfigs);
  };

  const renderDynamicParams = (config: LoggingConfig, configIndex: number) => {
    if (!config.callback_name) return null;

    // Find the display name for the callback
    const callbackDisplayName = Object.entries(callback_map).find(
      ([_, value]) => value === config.callback_name
    )?.[0];

    if (!callbackDisplayName) return null;

    const dynamicParams = callbackInfo[callbackDisplayName]?.dynamic_params || {};

    if (Object.keys(dynamicParams).length === 0) return null;

    return (
      <div className="mt-6 pt-4 border-t border-gray-100">
        <div className="flex items-center space-x-2 mb-4">
          <div className="w-3 h-3 bg-blue-100 rounded-full flex items-center justify-center">
            <div className="w-1.5 h-1.5 bg-blue-500 rounded-full"></div>
          </div>
          <span className="text-sm font-medium text-gray-700">Integration Parameters</span>
        </div>
        <div className="grid grid-cols-1 gap-4">
          {Object.entries(dynamicParams).map(([paramName, paramType]) => (
            <div key={paramName} className="space-y-2">
              <label className="text-sm font-medium text-gray-700 capitalize flex items-center space-x-1">
                <span>{paramName.replace(/_/g, ' ')}</span>
                <Tooltip title={`Environment variable reference recommended: os.environ/${paramName.toUpperCase()}`}>
                  <InfoCircleOutlined className="text-gray-400 cursor-help text-xs" />
                </Tooltip>
                {paramType === 'password' && (
                  <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-yellow-100 text-yellow-800">
                    Sensitive
                  </span>
                )}
              </label>
              <TextInput
                type={paramType === 'password' ? 'password' : 'text'}
                placeholder={`os.environ/${paramName.toUpperCase()}`}
                value={config.callback_vars[paramName] || ''}
                onChange={(e) => updateCallbackVar(configIndex, paramName, e.target.value)}
              />
            </div>
          ))}
        </div>
      </div>
    );
  };

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <div className="flex items-center space-x-2">
          <CogIcon className="w-5 h-5 text-blue-500" />
          <span className="text-base font-semibold text-gray-800">Logging Integrations</span>
          <Tooltip title="Configure callback logging integrations for this team.">
            <InfoCircleOutlined className="text-gray-400 cursor-help" />
          </Tooltip>
        </div>
        <Button
          variant="secondary"
          onClick={addLoggingConfig}
          icon={PlusIcon}
          size="sm"
          className="hover:border-blue-400 hover:text-blue-500"
          type="button"
        >
          Add Integration
        </Button>
      </div>

      <div className="space-y-4">
        {value.map((config, index) => {
          const callbackDisplayName = config.callback_name 
            ? Object.entries(callback_map).find(([_, value]) => value === config.callback_name)?.[0] 
            : undefined;
          const logoUrl = callbackDisplayName ? callbackInfo[callbackDisplayName]?.logo : null;

          return (
            <Card
              key={index}
              className="border border-gray-200 shadow-sm hover:shadow-md transition-shadow duration-200"
              decoration="top"
              decorationColor="blue"
            >
              <div className="flex justify-between items-start mb-4">
                <div className="flex items-center space-x-2">
                  {logoUrl && (
                    <img src={logoUrl} alt={callbackDisplayName} className="w-5 h-5 object-contain" />
                  )}
                  <span className="text-sm font-medium">
                    {callbackDisplayName || 'New Integration'} Configuration
                  </span>
                </div>
                <Button
                  variant="light"
                  onClick={() => removeLoggingConfig(index)}
                  icon={TrashIcon}
                  size="xs"
                  color="red"
                  className="hover:bg-red-50"
                  type="button"
                >
                  Remove
                </Button>
              </div>
              <div className="space-y-4">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <label className="text-sm font-medium text-gray-700">Integration Type</label>
                    <Select
                      value={callbackDisplayName}
                      placeholder="Select integration"
                      onChange={(value) => updateLoggingConfig(index, 'callback_name', value)}
                      className="w-full"
                      optionLabelProp="label"
                    >
                      {supportedCallbacks.map((callbackName) => {
                        const logo = callbackInfo[callbackName]?.logo;
                        return (
                          <Option key={callbackName} value={callbackName} label={callbackName}>
                            <div className="flex items-center space-x-2">
                              {logo && (
                                <img 
                                  src={logo} 
                                  alt={callbackName} 
                                  className="w-4 h-4 object-contain" 
                                />
                              )}
                              <span>{callbackName}</span>
                            </div>
                          </Option>
                        );
                      })}
                    </Select>
                  </div>

                  <div className="space-y-2">
                    <label className="text-sm font-medium text-gray-700">Event Type</label>
                    <Select
                      value={config.callback_type}
                      onChange={(value) => updateLoggingConfig(index, 'callback_type', value)}
                      className="w-full"
                    >
                      <Option value="success">
                        <div className="flex items-center space-x-2">
                          <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                          <span>Success Only</span>
                        </div>
                      </Option>
                      <Option value="failure">
                        <div className="flex items-center space-x-2">
                          <div className="w-2 h-2 bg-red-500 rounded-full"></div>
                          <span>Failure Only</span>
                        </div>
                      </Option>
                      <Option value="success_and_failure">
                        <div className="flex items-center space-x-2">
                          <div className="w-2 h-2 bg-blue-500 rounded-full"></div>
                          <span>Success & Failure</span>
                        </div>
                      </Option>
                    </Select>
                  </div>
                </div>

                {renderDynamicParams(config, index)}
              </div>
            </Card>
          );
        })}
      </div>

      {value.length === 0 && (
        <div className="text-center py-12 text-gray-500 border-2 border-dashed border-gray-200 rounded-lg bg-gray-50/50">
          <CogIcon className="w-12 h-12 text-gray-300 mb-3 mx-auto" />
          <div className="text-base font-medium mb-1">No logging integrations configured</div>
          <div className="text-sm text-gray-400">Click "Add Integration" to configure logging for this team</div>
        </div>
      )}
    </div>
  );
};

export default LoggingSettings; 