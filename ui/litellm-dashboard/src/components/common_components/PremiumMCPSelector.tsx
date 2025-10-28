import React from "react";
import { Text } from "@tremor/react";
import MCPServerSelector from "../mcp_server_management/MCPServerSelector";

interface PremiumMCPSelectorProps {
  onChange: (value: { servers: string[]; accessGroups: string[] }) => void;
  value: { servers: string[]; accessGroups: string[] };
  accessToken: string;
  placeholder?: string;
  premiumUser?: boolean;
}

export function PremiumMCPSelector({
  onChange,
  value,
  accessToken,
  placeholder = "Select MCP servers",
  premiumUser = false,
}: PremiumMCPSelectorProps) {
  if (!premiumUser) {
    return (
      <div>
        <div className="flex flex-wrap gap-2 mb-3">
          <div className="inline-flex items-center px-3 py-1.5 rounded-lg bg-purple-50 border border-purple-200 text-purple-800 text-sm font-medium opacity-50">
            ✨ premium-mcp-server-1
          </div>
          <div className="inline-flex items-center px-3 py-1.5 rounded-lg bg-purple-50 border border-purple-200 text-purple-800 text-sm font-medium opacity-50">
            ✨ premium-mcp-server-2
          </div>
        </div>
        <div className="p-3 bg-yellow-50 border border-yellow-200 rounded-lg">
          <Text className="text-sm text-yellow-800">
            MCP server access control is a LiteLLM Enterprise feature. Get a trial key{" "}
            <a href="https://www.litellm.ai/#pricing" target="_blank" rel="noopener noreferrer" className="underline">
              here
            </a>
            .
          </Text>
        </div>
      </div>
    );
  }

  return <MCPServerSelector onChange={onChange} value={value} accessToken={accessToken} placeholder={placeholder} />;
}

export default PremiumMCPSelector;
