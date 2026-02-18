import {
  formatInstallCommand,
  getCategoryBadgeColor,
  getSourceLink
} from "@/components/claude_code_plugins/helpers";
import { MarketplacePluginEntry } from "@/components/claude_code_plugins/types";
import NotificationsManager from "@/components/molecules/notifications_manager";
import { ExternalLinkIcon } from "@heroicons/react/outline";
import { CopyOutlined } from "@ant-design/icons";
import { Badge, Button, Card, Text } from "@tremor/react";
import { Tooltip } from "antd";
import React from "react";

interface PluginCardProps {
  plugin: MarketplacePluginEntry;
}

const PluginCard: React.FC<PluginCardProps> = ({ plugin }) => {
  const installCommand = formatInstallCommand(plugin);
  const sourceLink = getSourceLink(plugin.source);
  const categoryBadgeColor = getCategoryBadgeColor(plugin.category);

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
    NotificationsManager.success("Install command copied!");
  };

  // Limit keywords display to first 5
  const displayKeywords = plugin.keywords?.slice(0, 5) || [];
  const remainingKeywords = (plugin.keywords?.length || 0) - 5;

  return (
    <Card
      className="hover:shadow-lg transition-shadow duration-200 h-full flex flex-col"
      decoration="top"
      decorationColor={categoryBadgeColor}
    >
      {/* Header */}
      <div className="flex items-start justify-between mb-3">
        <div className="flex-1">
          <div className="flex items-center gap-2 flex-wrap">
            <h3 className="text-lg font-semibold text-gray-900">
              {plugin.name}
            </h3>
            {plugin.version && (
              <Badge color="blue" size="xs">
                v{plugin.version}
              </Badge>
            )}
            {plugin.category && (
              <Badge color={categoryBadgeColor} size="xs">
                {plugin.category}
              </Badge>
            )}
          </div>
        </div>
        {sourceLink && (
          <Tooltip title="View source repository">
            <a
              href={sourceLink}
              target="_blank"
              rel="noopener noreferrer"
              className="text-gray-500 hover:text-blue-500"
              onClick={(e) => e.stopPropagation()}
            >
              <ExternalLinkIcon className="h-5 w-5" />
            </a>
          </Tooltip>
        )}
      </div>

      {/* Description */}
      <div className="mb-4 flex-1">
        {plugin.description ? (
          <Text className="text-sm text-gray-600 line-clamp-3">
            {plugin.description}
          </Text>
        ) : (
          <Text className="text-sm text-gray-400 italic">
            No description available
          </Text>
        )}
      </div>

      {/* Keywords */}
      {displayKeywords.length > 0 && (
        <div className="flex flex-wrap gap-1 mb-4">
          {displayKeywords.map((keyword, index) => (
            <Badge key={index} color="gray" size="xs" className="text-xs">
              {keyword}
            </Badge>
          ))}
          {remainingKeywords > 0 && (
            <Badge color="gray" size="xs" className="text-xs">
              +{remainingKeywords} more
            </Badge>
          )}
        </div>
      )}

      {/* Author */}
      {plugin.author && (
        <div className="mb-4">
          <Text className="text-xs text-gray-500">
            By {plugin.author.name}
            {plugin.author.email && ` (${plugin.author.email})`}
          </Text>
        </div>
      )}

      {/* Homepage Link */}
      {plugin.homepage && (
        <div className="mb-4">
          <a
            href={plugin.homepage}
            target="_blank"
            rel="noopener noreferrer"
            className="text-xs text-blue-500 hover:text-blue-700 flex items-center gap-1"
            onClick={(e) => e.stopPropagation()}
          >
            Visit homepage
            <ExternalLinkIcon className="h-3 w-3" />
          </a>
        </div>
      )}

      {/* Install Command */}
      <div className="mt-auto pt-4 border-t border-gray-100">
        <div className="flex items-center justify-between gap-2">
          <div className="flex-1 overflow-hidden">
            <Text className="text-xs text-gray-500 mb-1">Install command</Text>
            <Tooltip title={installCommand}>
              <code className="block text-xs bg-gray-50 px-2 py-1 rounded font-mono text-gray-700 truncate">
                {installCommand}
              </code>
            </Tooltip>
          </div>
          <Tooltip title="Copy install command">
            <Button
              size="xs"
              variant="secondary"
              icon={CopyOutlined}
              onClick={(e) => {
                e.stopPropagation();
                copyToClipboard(installCommand);
              }}
            />
          </Tooltip>
        </div>
      </div>
    </Card>
  );
};

export default PluginCard;
