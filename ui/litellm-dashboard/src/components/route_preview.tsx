import React from "react";
import { Card, Typography } from "antd";
import { RightOutlined, InfoCircleOutlined } from "@ant-design/icons";
import { Trans, useTranslation } from "react-i18next";

const { Title, Text } = Typography;
import { getProxyBaseUrl } from "./networking";

interface RoutePreviewProps {
  pathValue: string;
  targetValue: string;
  includeSubpath: boolean;
}

const RoutePreview: React.FC<RoutePreviewProps> = ({ pathValue, targetValue, includeSubpath }) => {
  const { t } = useTranslation();
  const proxyBaseUrl = getProxyBaseUrl();

  const getLiteLLMProxyUrl = () => {
    return pathValue ? `${proxyBaseUrl}${pathValue}` : "";
  };

  // Only show if both path and target are provided
  if (!pathValue || !targetValue) {
    return null;
  }

  return (
    <Card className="p-5">
      <Title level={5} className="text-lg font-semibold text-gray-900 mb-2">
        {t("routePreview.title")}
      </Title>
      <Text type="secondary" className="text-gray-600 mb-5" style={{ display: "block" }}>
        {t("routePreview.subtitle")}
      </Text>

      <div className="space-y-5">
        {/* Basic routing */}
        <div>
          <div className="text-base font-semibold text-gray-900 mb-3">{t("routePreview.basicRouting")}</div>
          <div className="flex items-center gap-4">
            {/* Your endpoint */}
            <div className="flex-1 bg-gray-50 border border-gray-200 rounded-lg p-3">
              <div className="text-sm text-gray-600 mb-2">{t("routePreview.yourEndpoint")}</div>
              <code className="font-mono text-sm text-gray-900">{getLiteLLMProxyUrl()}</code>
            </div>

            {/* Arrow */}
            <div className="text-gray-400">
              <RightOutlined className="text-lg" />
            </div>

            {/* Forwards to */}
            <div className="flex-1 bg-gray-50 border border-gray-200 rounded-lg p-3">
              <div className="text-sm text-gray-600 mb-2">{t("routePreview.forwardsTo")}</div>
              <code className="font-mono text-sm text-gray-900">{targetValue}</code>
            </div>
          </div>
        </div>

        {includeSubpath && (
          <>
            {/* With subpaths */}
            <div>
              <div className="text-base font-semibold text-gray-900 mb-3">{t("routePreview.withSubpaths")}</div>
              <div className="flex items-center gap-4">
                {/* Your endpoint + subpath */}
                <div className="flex-1 bg-gray-50 border border-gray-200 rounded-lg p-3">
                  <div className="text-sm text-gray-600 mb-2">{t("routePreview.yourEndpointSubpath")}</div>
                  <code className="font-mono text-sm text-gray-900">
                    {pathValue && `${proxyBaseUrl}${pathValue}`}
                    <span className="text-blue-600">/v1/text-to-image/base/model</span>
                  </code>
                </div>

                {/* Arrow */}
                <div className="text-gray-400">
                  <RightOutlined className="text-lg" />
                </div>

                {/* Forwards to with subpath */}
                <div className="flex-1 bg-gray-50 border border-gray-200 rounded-lg p-3">
                  <div className="text-sm text-gray-600 mb-2">{t("routePreview.forwardsTo")}</div>
                  <code className="font-mono text-sm text-gray-900">
                    {targetValue}
                    <span className="text-blue-600">/v1/text-to-image/base/model</span>
                  </code>
                </div>
              </div>

              {/* Note */}
              <div className="mt-3 text-sm text-gray-600">{t("routePreview.subpathNote", { pathValue })}</div>
            </div>
          </>
        )}

        {!includeSubpath && (
          <div className="mt-4 p-3 bg-blue-50 rounded-md border border-blue-200">
            <div className="flex items-start">
              <InfoCircleOutlined className="text-blue-500 mt-0.5 mr-2 flex-shrink-0" />
              <div className="text-sm text-blue-700">
                <Trans
                  i18nKey="routePreview.subpathHint"
                  components={{
                    bold: <span className="font-medium" />,
                    code: <code className="bg-blue-100 px-1 py-0.5 rounded font-mono text-xs" />,
                  }}
                />
              </div>
            </div>
          </div>
        )}
      </div>
    </Card>
  );
};

export default RoutePreview;
