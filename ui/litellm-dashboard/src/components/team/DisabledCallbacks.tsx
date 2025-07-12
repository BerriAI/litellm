import React, { useState, useEffect } from 'react';
import { Select, Tooltip } from 'antd';
import { InfoCircleOutlined } from '@ant-design/icons';
import { Card } from '@tremor/react';
import { getCallbacksListCall } from '../networking';
import { callbackInfo, callback_map } from '../callback_info_helpers';

const { Option } = Select;

interface DisabledCallbacksProps {
  value?: string[];
  onChange?: (disabled: string[]) => void;
  accessToken?: string;
}

interface CallbacksList {
  success: string[];
  failure: string[];
  success_and_failure: string[];
}

const DisabledCallbacks: React.FC<DisabledCallbacksProps> = ({
  value = [],
  onChange,
  accessToken
}) => {
  const [availableCallbacks, setAvailableCallbacks] = useState<CallbacksList>({
    success: [],
    failure: [],
    success_and_failure: []
  });

  // Fetch available callbacks from the API
  useEffect(() => {
    const fetchCallbacks = async () => {
      if (!accessToken) return;
      
      try {
        const callbacksList = await getCallbacksListCall(accessToken);
        setAvailableCallbacks(callbacksList);
      } catch (error) {
        console.error('Failed to fetch callbacks list:', error);
      }
    };

    fetchCallbacks();
  }, [accessToken]);

  // Get all available callbacks for the disabled callbacks selector
  const getAllAvailableCallbacks = () => {
    const { success, failure, success_and_failure } = availableCallbacks;
    return [...success, ...failure, ...success_and_failure];
  };

  // Find callback info (logo, display name) for a given callback name
  const getCallbackInfo = (callbackName: string) => {
    // Try to find a matching callback in callback_map (reverse lookup)
    const displayName = Object.entries(callback_map).find(
      ([_, value]) => value === callbackName
    )?.[0];

    if (displayName && callbackInfo[displayName]) {
      return {
        logo: callbackInfo[displayName].logo,
        displayName
      };
    }

    // If no match found, return null
    return null;
  };

  const getCallbackType = (callbackName: string) => {
    const { success, failure, success_and_failure } = availableCallbacks;
    if (success.includes(callbackName)) return 'Success';
    if (failure.includes(callbackName)) return 'Failure';
    if (success_and_failure.includes(callbackName)) return 'Success & Failure';
    return '';
  };

  const handleChange = (disabled: string[]) => {
    onChange?.(disabled);
  };

  return (
    <Card className="border border-gray-200 shadow-sm">
      <div className="space-y-4">
        <div className="flex items-center space-x-2">
          <div className="w-3 h-3 bg-red-100 rounded-full flex items-center justify-center">
            <div className="w-1.5 h-1.5 bg-red-500 rounded-full"></div>
          </div>
          <span className="text-sm font-medium text-gray-700">Disabled Callbacks</span>
          <Tooltip title="Select callbacks to disable for this key. These callbacks will not be executed.">
            <InfoCircleOutlined className="text-gray-400 cursor-help text-xs" />
          </Tooltip>
        </div>
        <div className="space-y-2">
          <label className="text-sm font-medium text-gray-700">
            Callbacks to Disable
          </label>
          <Select
            mode="multiple"
            placeholder="Select callbacks to disable"
            value={value}
            onChange={handleChange}
            className="w-full"
            showSearch
            filterOption={(input, option) =>
              String(option?.label ?? '').toLowerCase().includes(input.toLowerCase())
            }
          >
            {getAllAvailableCallbacks().map((callbackName) => {
              const callbackInfo = getCallbackInfo(callbackName);
              const callbackType = getCallbackType(callbackName);
              
              return (
                <Option key={callbackName} value={callbackName} label={callbackName}>
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-2">
                      {callbackInfo?.logo && (
                        <img 
                          src={callbackInfo.logo} 
                          alt={callbackInfo.displayName} 
                          className="w-4 h-4 object-contain" 
                        />
                      )}
                      <span>{callbackInfo?.displayName || callbackName} </span>
                    </div>
                    <span className="text-xs text-gray-500">
                      {callbackType}
                    </span>
                  </div>
                </Option>
              );
            })}
          </Select>
          {value.length > 0 && (
            <div className="text-xs text-gray-500 mt-2">
              {value.length} callback(s) will be disabled for this key
            </div>
          )}
        </div>
      </div>
    </Card>
  );
};

export default DisabledCallbacks; 