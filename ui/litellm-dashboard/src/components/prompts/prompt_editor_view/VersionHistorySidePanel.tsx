import { Drawer, List, Skeleton, Tag, Typography } from "antd";
import React, { useEffect, useState } from "react";
import { getPromptVersions, PromptSpec } from "../../networking";

const { Text } = Typography;

interface VersionHistorySidePanelProps {
  isOpen: boolean;
  onClose: () => void;
  accessToken: string | null;
  promptId: string;
  activeVersionId?: string;
  onSelectVersion?: (version: PromptSpec) => void;
}

const VersionHistorySidePanel: React.FC<VersionHistorySidePanelProps> = ({
  isOpen,
  onClose,
  accessToken,
  promptId,
  activeVersionId,
  onSelectVersion,
}) => {
  const [versions, setVersions] = useState<PromptSpec[]>([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (isOpen && accessToken && promptId) {
      fetchVersions();
    }
  }, [isOpen, accessToken, promptId]);

  const fetchVersions = async () => {
    setLoading(true);
    try {
      // Strip .v suffix if present to get base ID for querying all versions
      const basePromptId = promptId.includes(".v") ? promptId.split(".v")[0] : promptId;
      const response = await getPromptVersions(accessToken!, basePromptId);
      setVersions(response.prompts);
    } catch (error) {
      console.error("Error fetching prompt versions:", error);
    } finally {
      setLoading(false);
    }
  };

  const getVersionNumber = (prompt: PromptSpec) => {
    // Use explicit version field if available, otherwise try to extract from litellm_params.prompt_id
    if (prompt.version) {
      return `v${prompt.version}`;
    }
    
    // Fallback: try to extract from litellm_params.prompt_id
    const versionedId = (prompt.litellm_params as any)?.prompt_id || prompt.prompt_id;
    if (versionedId.includes(".v")) {
      return `v${versionedId.split(".v")[1]}`;
    }
    if (versionedId.includes("_v")) {
      return `v${versionedId.split("_v")[1]}`;
    }
    return "v1";
  };

  const formatDate = (dateString?: string) => {
    if (!dateString) return "-";
    return new Date(dateString).toLocaleString();
  };

  return (
    <Drawer
      title="Version History"
      placement="right"
      onClose={onClose}
      open={isOpen}
      width={400}
      mask={false} // Allow interacting with the main editor while drawer is open
      maskClosable={false}
    >
      {loading ? (
        <Skeleton active paragraph={{ rows: 4 }} />
      ) : versions.length === 0 ? (
        <div className="text-center py-8 text-gray-500">No version history available.</div>
      ) : (
        <List
          dataSource={versions}
          renderItem={(item, index) => {
            // Use version field for comparison since all items have the same prompt_id
            const itemVersionNum = item.version || parseInt(getVersionNumber(item).replace('v', ''));
            
            // Extract version number from activeVersionId (may have .vX suffix)
            let activeVersionNum: number | null = null;
            if (activeVersionId) {
              if (activeVersionId.includes('.v')) {
                activeVersionNum = parseInt(activeVersionId.split('.v')[1]);
              } else if (activeVersionId.includes('_v')) {
                activeVersionNum = parseInt(activeVersionId.split('_v')[1]);
              }
            }
            
            // Default to latest (first item) if no activeVersionId
            const isSelected = activeVersionNum ? itemVersionNum === activeVersionNum : index === 0;
            
            return (
              <div
                key={`${item.prompt_id}-v${item.version || itemVersionNum}`}
                className={`mb-4 p-4 rounded-lg border cursor-pointer transition-all hover:shadow-md ${
                  isSelected ? "border-blue-500 bg-blue-50" : "border-gray-200 bg-white hover:border-blue-300"
                }`}
                onClick={() => onSelectVersion?.(item)}
              >
                <div className="flex justify-between items-start mb-2">
                  <div className="flex items-center gap-2">
                    <Tag className="m-0">
                      {getVersionNumber(item)}
                    </Tag>
                    {index === 0 && <Tag color="blue" className="m-0">Latest</Tag>}
                  </div>
                  {isSelected && (
                    <Tag color="green" className="m-0">
                      Active
                    </Tag>
                  )}
                </div>

                <div className="flex flex-col gap-1">
                  <Text className="text-sm text-gray-600 font-medium">{formatDate(item.created_at)}</Text>
                  <Text type="secondary" className="text-xs">
                    {item.prompt_info?.prompt_type === "db" ? "Saved to Database" : "Config Prompt"}
                  </Text>
                </div>
              </div>
            );
          }}
        />
      )}
    </Drawer>
  );
};

export default VersionHistorySidePanel;
