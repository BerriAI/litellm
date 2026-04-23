import { CopyOutlined } from "@ant-design/icons";
import { ArrowLeftIcon, ExternalLinkIcon } from "@heroicons/react/outline";
import {
  Badge,
  Button,
  Card,
  Grid,
  Text,
  Title,
} from "@tremor/react";
import { Spin, Switch, Tooltip } from "antd";
import React, { useEffect, useState } from "react";
import NotificationsManager from "../molecules/notifications_manager";
import {
  disableClaudeCodePlugin,
  enableClaudeCodePlugin,
  getClaudeCodePluginDetails,
} from "../networking";
import {
  formatDateString,
  formatInstallCommand,
  getCategoryBadgeColor,
  getSourceDisplayText,
  getSourceLink
} from "./helpers";
import { Plugin } from "./types";

interface PluginInfoViewProps {
  pluginId: string;
  onClose: () => void;
  accessToken: string | null;
  isAdmin: boolean;
  onPluginUpdated: () => void;
}

const PluginInfoView: React.FC<PluginInfoViewProps> = ({
  pluginId,
  onClose,
  accessToken,
  isAdmin,
  onPluginUpdated,
}) => {
  const [plugin, setPlugin] = useState<Plugin | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isToggling, setIsToggling] = useState(false);

  useEffect(() => {
    fetchPluginInfo();
  }, [pluginId, accessToken]);

  const fetchPluginInfo = async () => {
    if (!accessToken) return;

    setIsLoading(true);
    try {
      // The backend expects plugin name, not ID
      // We'll need to find the plugin by ID from the list
      // For now, assume pluginId is actually the plugin name
      const data = await getClaudeCodePluginDetails(
        accessToken,
        pluginId as string
      );
      setPlugin(data.plugin);
    } catch (error) {
      console.error("Error fetching plugin info:", error);
      NotificationsManager.error("Failed to load plugin information");
    } finally {
      setIsLoading(false);
    }
  };

  const handleToggleEnabled = async () => {
    if (!accessToken || !plugin) return;

    setIsToggling(true);
    try {
      if (plugin.enabled) {
        await disableClaudeCodePlugin(accessToken, plugin.name);
        NotificationsManager.success(`Plugin "${plugin.name}" disabled`);
      } else {
        await enableClaudeCodePlugin(accessToken, plugin.name);
        NotificationsManager.success(`Plugin "${plugin.name}" enabled`);
      }
      onPluginUpdated();
      fetchPluginInfo();
    } catch (error) {
      NotificationsManager.error("Failed to toggle plugin status");
    } finally {
      setIsToggling(false);
    }
  };

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
    NotificationsManager.success("Copied to clipboard!");
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center p-8">
        <Spin size="large" />
      </div>
    );
  }

  if (!plugin) {
    return (
      <div className="p-8 text-center text-gray-500">
        <p>Plugin not found</p>
        <Button className="mt-4" onClick={onClose}>
          Go Back
        </Button>
      </div>
    );
  }

  const installCommand = formatInstallCommand(plugin);
  const sourceLink = getSourceLink(plugin.source);
  const categoryBadgeColor = getCategoryBadgeColor(plugin.category);

  return (
    <div className="space-y-4">
      {/* Header with Back Button */}
      <div className="flex items-center gap-3 mb-6">
        <ArrowLeftIcon
          className="h-5 w-5 cursor-pointer text-gray-500 hover:text-gray-700"
          onClick={onClose}
        />
        <h2 className="text-2xl font-bold">{plugin.name}</h2>
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
        <Badge color={plugin.enabled ? "green" : "gray"} size="xs">
          {plugin.enabled ? "Enabled" : "Disabled"}
        </Badge>
      </div>

      {/* Install Command */}
      <Card>
        <div className="flex items-center justify-between">
          <div className="flex-1">
            <Text className="text-gray-600 text-xs mb-2">Install Command</Text>
            <div className="font-mono bg-gray-100 px-3 py-2 rounded text-sm">
              {installCommand}
            </div>
          </div>
          <Tooltip title="Copy install command">
            <Button
              size="xs"
              variant="secondary"
              icon={CopyOutlined}
              onClick={() => copyToClipboard(installCommand)}
              className="ml-4"
            >
              Copy
            </Button>
          </Tooltip>
        </div>
      </Card>

      {/* Plugin Details */}
      <Card>
        <Title>Plugin Details</Title>
        <Grid
          className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6 mt-4"
        >
          {/* Plugin ID */}
          <div>
            <Text className="text-gray-600 text-xs">Plugin ID</Text>
            <div className="flex items-center gap-2 mt-1">
              <Text className="font-mono text-xs">{plugin.id}</Text>
              <CopyOutlined
                className="cursor-pointer text-gray-500 hover:text-blue-500 text-xs"
                onClick={() => copyToClipboard(plugin.id)}
              />
            </div>
          </div>

          {/* Name */}
          <div>
            <Text className="text-gray-600 text-xs">Name</Text>
            <Text className="font-semibold mt-1">{plugin.name}</Text>
          </div>

          {/* Version */}
          <div>
            <Text className="text-gray-600 text-xs">Version</Text>
            <Text className="font-semibold mt-1">
              {plugin.version || "N/A"}
            </Text>
          </div>

          {/* Source */}
          <div className="col-span-2">
            <Text className="text-gray-600 text-xs">Source</Text>
            <div className="flex items-center gap-2 mt-1">
              <Text className="font-semibold">
                {getSourceDisplayText(plugin.source)}
              </Text>
              {sourceLink && (
                <a
                  href={sourceLink}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-blue-500 hover:text-blue-700"
                >
                  <ExternalLinkIcon className="h-4 w-4" />
                </a>
              )}
            </div>
          </div>

          {/* Category */}
          <div>
            <Text className="text-gray-600 text-xs">Category</Text>
            <div className="mt-1">
              {plugin.category ? (
                <Badge color={categoryBadgeColor} size="xs">
                  {plugin.category}
                </Badge>
              ) : (
                <Text className="text-gray-400">Uncategorized</Text>
              )}
            </div>
          </div>

          {/* Enabled Status */}
          {isAdmin && (
            <div className="col-span-3">
              <Text className="text-gray-600 text-xs">Status</Text>
              <div className="flex items-center gap-3 mt-2">
                <Switch
                  checked={plugin.enabled}
                  loading={isToggling}
                  onChange={handleToggleEnabled}
                />
                <Text className="text-sm">
                  {plugin.enabled
                    ? "Plugin is enabled and visible in marketplace"
                    : "Plugin is disabled and hidden from marketplace"}
                </Text>
              </div>
            </div>
          )}
        </Grid>
      </Card>

      {/* Description */}
      {plugin.description && (
        <Card>
          <Title>Description</Title>
          <Text className="mt-2">{plugin.description}</Text>
        </Card>
      )}

      {/* Keywords */}
      {plugin.keywords && plugin.keywords.length > 0 && (
        <Card>
          <Title>Keywords</Title>
          <div className="flex flex-wrap gap-2 mt-2">
            {plugin.keywords.map((keyword, index) => (
              <Badge key={index} color="gray" size="xs">
                {keyword}
              </Badge>
            ))}
          </div>
        </Card>
      )}

      {/* Author Information */}
      {plugin.author && (
        <Card>
          <Title>Author Information</Title>
          <Grid className="grid grid-cols-1 sm:grid-cols-2 gap-4 mt-4">
            {plugin.author.name && (
              <div>
                <Text className="text-gray-600 text-xs">Name</Text>
                <Text className="font-semibold mt-1">
                  {plugin.author.name}
                </Text>
              </div>
            )}
            {plugin.author.email && (
              <div>
                <Text className="text-gray-600 text-xs">Email</Text>
                <Text className="font-semibold mt-1">
                  <a
                    href={`mailto:${plugin.author.email}`}
                    className="text-blue-500 hover:text-blue-700"
                  >
                    {plugin.author.email}
                  </a>
                </Text>
              </div>
            )}
          </Grid>
        </Card>
      )}

      {/* Additional Links */}
      {plugin.homepage && (
        <Card>
          <Title>Homepage</Title>
          <a
            href={plugin.homepage}
            target="_blank"
            rel="noopener noreferrer"
            className="text-blue-500 hover:text-blue-700 flex items-center gap-2 mt-2"
          >
            {plugin.homepage}
            <ExternalLinkIcon className="h-4 w-4" />
          </a>
        </Card>
      )}

      {/* Timestamps */}
      <Card>
        <Title>Metadata</Title>
        <Grid className="grid grid-cols-1 sm:grid-cols-2 gap-4 mt-4">
          <div>
            <Text className="text-gray-600 text-xs">Created At</Text>
            <Text className="font-semibold mt-1">
              {formatDateString(plugin.created_at)}
            </Text>
          </div>
          <div>
            <Text className="text-gray-600 text-xs">Updated At</Text>
            <Text className="font-semibold mt-1">
              {formatDateString(plugin.updated_at)}
            </Text>
          </div>
          {plugin.created_by && (
            <div className="col-span-2">
              <Text className="text-gray-600 text-xs">Created By</Text>
              <Text className="font-semibold mt-1">{plugin.created_by}</Text>
            </div>
          )}
        </Grid>
      </Card>
    </div>
  );
};

export default PluginInfoView;
