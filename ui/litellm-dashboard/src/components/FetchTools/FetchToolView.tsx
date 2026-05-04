import React from "react";
import { Button, Tag } from "antd";
import { AvailableFetchProvider, FetchTool } from "./types";

interface FetchToolViewProps {
  fetchTool: FetchTool;
  onBack: () => void;
  isEditing: boolean;
  accessToken: string;
  availableProviders: AvailableFetchProvider[];
}

const FetchToolView: React.FC<FetchToolViewProps> = ({ fetchTool, onBack, availableProviders }) => {
  const providerInfo = availableProviders.find((p) => p.provider_name === fetchTool.litellm_params?.fetch_provider);

  return (
    <div className="w-full h-full p-6">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h2 className="text-2xl font-bold">{fetchTool.fetch_tool_name}</h2>
          <p className="text-gray-500 mt-1">
            {providerInfo?.ui_friendly_name || fetchTool.litellm_params?.fetch_provider}
          </p>
        </div>
        <Button onClick={onBack}>Back to List</Button>
      </div>

      <div className="space-y-4">
        <div className="bg-gray-50 p-4 rounded">
          <h3 className="font-semibold mb-2">Configuration</h3>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <span className="text-gray-500">Provider:</span>
              <span className="ml-2">{fetchTool.litellm_params?.fetch_provider}</span>
            </div>
            {fetchTool.litellm_params?.api_base && (
              <div>
                <span className="text-gray-500">API Base:</span>
                <span className="ml-2">{fetchTool.litellm_params.api_base}</span>
              </div>
            )}
            {fetchTool.litellm_params?.timeout && (
              <div>
                <span className="text-gray-500">Timeout:</span>
                <span className="ml-2">{fetchTool.litellm_params.timeout}s</span>
              </div>
            )}
            {fetchTool.litellm_params?.max_retries && (
              <div>
                <span className="text-gray-500">Max Retries:</span>
                <span className="ml-2">{fetchTool.litellm_params.max_retries}</span>
              </div>
            )}
          </div>
        </div>

        {fetchTool.fetch_tool_info?.description && (
          <div className="bg-gray-50 p-4 rounded">
            <h3 className="font-semibold mb-2">Description</h3>
            <p>{fetchTool.fetch_tool_info.description}</p>
          </div>
        )}

        <div className="flex gap-2">
          <Tag color={fetchTool.is_from_config ? "default" : "blue"}>
            {fetchTool.is_from_config ? "Config" : "Database"}
          </Tag>
          {fetchTool.created_at && (
            <Tag color="green">Created: {new Date(fetchTool.created_at).toLocaleDateString()}</Tag>
          )}
        </div>
      </div>
    </div>
  );
};

export default FetchToolView;
