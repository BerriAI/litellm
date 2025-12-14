import React from "react";
import { Card, Title, Subtitle, Text } from "@tremor/react";
import { Form, Switch } from "antd";

export interface PassThroughSecuritySectionProps {
  premiumUser: boolean;
  authEnabled: boolean;
  onAuthChange: (checked: boolean) => void;
}

/**
 * Reusable Security section for pass-through endpoints
 * Shows authentication toggle for premium users or upgrade message for free users
 */
const PassThroughSecuritySection: React.FC<PassThroughSecuritySectionProps> = ({
  premiumUser,
  authEnabled,
  onAuthChange,
}) => {
  return (
    <Card className="p-6">
      <Title className="text-lg font-semibold text-gray-900 mb-2">Security</Title>
      <Subtitle className="text-gray-600 mb-4">
        When enabled, requests to this endpoint will require a valid LiteLLM Virtual Key
      </Subtitle>
      {premiumUser ? (
        <Form.Item name="auth" valuePropName="checked" className="mb-0">
          <Switch
            checked={authEnabled}
            onChange={(checked) => {
              onAuthChange(checked);
            }}
          />
        </Form.Item>
      ) : (
        <div>
          <div className="flex items-center mb-3">
            <Switch disabled checked={false} style={{ outline: "2px solid #d1d5db", outlineOffset: "2px" }} />
            <span className="ml-2 text-sm text-gray-400">Authentication (Premium)</span>
          </div>
          <div className="p-3 bg-yellow-50 border border-yellow-200 rounded-lg">
            <Text className="text-sm text-yellow-800">
              Setting authentication for pass-through endpoints is a LiteLLM Enterprise feature. Get a trial key{" "}
              <a href="https://www.litellm.ai/#pricing" target="_blank" rel="noopener noreferrer" className="underline">
                here
              </a>
              .
            </Text>
          </div>
        </div>
      )}
    </Card>
  );
};

export default PassThroughSecuritySection;
