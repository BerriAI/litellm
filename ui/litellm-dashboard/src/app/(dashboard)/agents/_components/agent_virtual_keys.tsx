import React from "react";
import { Button, Tooltip, Typography } from "antd";
import { KeyOutlined } from "@ant-design/icons";
import { KeyResponse } from "@/components/key_team_helpers/key_list";

const { Title, Text } = Typography;

interface AgentVirtualKeysProps {
  keys: KeyResponse[];
  isLoading: boolean;
  onKeyClick: (key: KeyResponse) => void;
}

const AgentVirtualKeys: React.FC<AgentVirtualKeysProps> = ({ keys, isLoading, onKeyClick }) => {
  return (
    <div style={{ marginTop: 24 }}>
      <Title level={4}>Virtual Keys</Title>
      {isLoading ? (
        <Text className="mt-2 block">Loading keys...</Text>
      ) : keys.length === 0 ? (
        <Text className="mt-2 block text-gray-500">No virtual key assigned to this agent.</Text>
      ) : (
        <div className="mt-3 flex flex-col gap-2">
          {keys.map((key) => (
            <div key={key.token} className="flex items-center gap-3 border border-gray-100 rounded-sm px-3 py-2">
              <KeyOutlined className="text-gray-400" />
              <span className="font-medium">{key.key_alias || "Unnamed key"}</span>
              {key.key_name && <span className="font-mono text-xs text-gray-500">{key.key_name}</span>}
              <Tooltip title={key.token}>
                <Button
                  size="small"
                  type="link"
                  className="font-mono text-blue-500 ml-auto"
                  onClick={() => onKeyClick(key)}
                >
                  {key.token?.slice(0, 12)}...
                </Button>
              </Tooltip>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default AgentVirtualKeys;
