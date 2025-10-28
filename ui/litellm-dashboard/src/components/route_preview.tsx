import React from "react";
import { Card, Title, Subtitle } from "@tremor/react";
import { RightOutlined, InfoCircleOutlined } from "@ant-design/icons";
import { getProxyBaseUrl } from "./networking";

interface RoutePreviewProps {
  pathValue: string;
  targetValue: string;
  includeSubpath: boolean;
}

const RoutePreview: React.FC<RoutePreviewProps> = ({ pathValue, targetValue, includeSubpath }) => {
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
      <Title className="text-lg font-semibold text-gray-900 mb-2">Route Preview</Title>
      <Subtitle className="text-gray-600 mb-5">How your requests will be routed</Subtitle>

      <div className="space-y-5">
        {/* Basic routing */}
        <div>
          <div className="text-base font-semibold text-gray-900 mb-3">Basic routing:</div>
          <div className="flex items-center gap-4">
            {/* Your endpoint */}
            <div className="flex-1 bg-gray-50 border border-gray-200 rounded-lg p-3">
              <div className="text-sm text-gray-600 mb-2">Your endpoint</div>
              <code className="font-mono text-sm text-gray-900">{getLiteLLMProxyUrl()}</code>
            </div>

            {/* Arrow */}
            <div className="text-gray-400">
              <RightOutlined className="text-lg" />
            </div>

            {/* Forwards to */}
            <div className="flex-1 bg-gray-50 border border-gray-200 rounded-lg p-3">
              <div className="text-sm text-gray-600 mb-2">Forwards to</div>
              <code className="font-mono text-sm text-gray-900">{targetValue}</code>
            </div>
          </div>
        </div>

        {includeSubpath && (
          <>
            {/* With subpaths */}
            <div>
              <div className="text-base font-semibold text-gray-900 mb-3">With subpaths:</div>
              <div className="flex items-center gap-4">
                {/* Your endpoint + subpath */}
                <div className="flex-1 bg-gray-50 border border-gray-200 rounded-lg p-3">
                  <div className="text-sm text-gray-600 mb-2">Your endpoint + subpath</div>
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
                  <div className="text-sm text-gray-600 mb-2">Forwards to</div>
                  <code className="font-mono text-sm text-gray-900">
                    {targetValue}
                    <span className="text-blue-600">/v1/text-to-image/base/model</span>
                  </code>
                </div>
              </div>

              {/* Note */}
              <div className="mt-3 text-sm text-gray-600">
                Any path after {pathValue} will be appended to the target URL
              </div>
            </div>
          </>
        )}

        {!includeSubpath && (
          <div className="mt-4 p-3 bg-blue-50 rounded-md border border-blue-200">
            <div className="flex items-start">
              <InfoCircleOutlined className="text-blue-500 mt-0.5 mr-2 flex-shrink-0" />
              <div className="text-sm text-blue-700">
                <span className="font-medium">Not seeing the routing you wanted?</span> Try enabling - Include Subpaths
                - above - this allows subroutes like{" "}
                <code className="bg-blue-100 px-1 py-0.5 rounded font-mono text-xs">/api/v1/models</code> to be
                forwarded automatically.
              </div>
            </div>
          </div>
        )}
      </div>
    </Card>
  );
};

export default RoutePreview;
