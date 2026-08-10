import React from "react";
import { Card, Title, Subtitle, Text } from "@tremor/react";
import { Form, Switch } from "antd";
import { useTranslation } from "react-i18next";

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
  const { t } = useTranslation("gateway");
  return (
    <Card className="p-6">
      <Title className="text-lg font-semibold text-gray-900 mb-2">{t("models.passThrough.security.title")}</Title>
      <Subtitle className="text-gray-600 mb-4">{t("models.passThrough.security.description")}</Subtitle>
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
            <span className="ml-2 text-sm text-gray-400">{t("models.passThrough.security.premium")}</span>
          </div>
          <div className="p-3 bg-yellow-50 border border-yellow-200 rounded-lg">
            <Text className="text-sm text-yellow-800">
              {t("models.passThrough.security.enterprisePrefix")}{" "}
              <a href="https://www.litellm.ai/#pricing" target="_blank" rel="noopener noreferrer" className="underline">
                {t("models.passThrough.security.here")}
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
