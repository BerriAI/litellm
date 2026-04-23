import {
  ArrowLeft,
  Copy,
  ExternalLink,
  Loader2,
} from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Switch } from "@/components/ui/switch";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";
import React, { useCallback, useEffect, useState } from "react";
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
  getSourceLink,
} from "./helpers";
import { Plugin } from "./types";

// Map tremor-style color tokens to categorical Tailwind palette classes.
const BADGE_COLOR_CLASSES: Record<string, string> = {
  gray: "bg-muted text-muted-foreground",
  green:
    "bg-emerald-100 text-emerald-700 dark:bg-emerald-950 dark:text-emerald-300",
  red: "bg-red-100 text-red-700 dark:bg-red-950 dark:text-red-300",
  blue: "bg-blue-100 text-blue-700 dark:bg-blue-950 dark:text-blue-300",
  indigo:
    "bg-indigo-100 text-indigo-700 dark:bg-indigo-950 dark:text-indigo-300",
  purple:
    "bg-purple-100 text-purple-700 dark:bg-purple-950 dark:text-purple-300",
  orange:
    "bg-orange-100 text-orange-700 dark:bg-orange-950 dark:text-orange-300",
  amber: "bg-amber-100 text-amber-700 dark:bg-amber-950 dark:text-amber-300",
  yellow: "bg-amber-100 text-amber-700 dark:bg-amber-950 dark:text-amber-300",
  cyan: "bg-cyan-100 text-cyan-700 dark:bg-cyan-950 dark:text-cyan-300",
  pink: "bg-pink-100 text-pink-700 dark:bg-pink-950 dark:text-pink-300",
  rose: "bg-rose-100 text-rose-700 dark:bg-rose-950 dark:text-rose-300",
  teal: "bg-teal-100 text-teal-700 dark:bg-teal-950 dark:text-teal-300",
};

const badgeClasses = (color: string): string =>
  BADGE_COLOR_CLASSES[color] || BADGE_COLOR_CLASSES.gray;

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

  const fetchPluginInfo = useCallback(async () => {
    if (!accessToken) return;

    setIsLoading(true);
    try {
      const data = await getClaudeCodePluginDetails(
        accessToken,
        pluginId as string,
      );
      setPlugin(data.plugin);
    } catch (error) {
      console.error("Error fetching plugin info:", error);
      NotificationsManager.error("Failed to load plugin information");
    } finally {
      setIsLoading(false);
    }
  }, [accessToken, pluginId]);

  useEffect(() => {
    fetchPluginInfo();
  }, [fetchPluginInfo]);

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
      // eslint-disable-next-line @typescript-eslint/no-unused-vars
    } catch (_error) {
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
        <Loader2 className="h-6 w-6 animate-spin text-primary" />
      </div>
    );
  }

  if (!plugin) {
    return (
      <div className="p-8 text-center text-muted-foreground">
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
      <div className="flex items-center gap-3 mb-6">
        <ArrowLeft
          className="h-5 w-5 cursor-pointer text-muted-foreground hover:text-foreground"
          onClick={onClose}
        />
        <h2 className="text-2xl font-bold">{plugin.name}</h2>
        {plugin.version && (
          <Badge className={cn("text-xs", badgeClasses("blue"))}>
            v{plugin.version}
          </Badge>
        )}
        {plugin.category && (
          <Badge className={cn("text-xs", badgeClasses(categoryBadgeColor))}>
            {plugin.category}
          </Badge>
        )}
        <Badge
          className={cn(
            "text-xs",
            badgeClasses(plugin.enabled ? "green" : "gray"),
          )}
        >
          {plugin.enabled ? "Enabled" : "Disabled"}
        </Badge>
      </div>

      {/* Install Command */}
      <Card className="p-4">
        <div className="flex items-center justify-between">
          <div className="flex-1">
            <p className="text-muted-foreground text-xs mb-2">
              Install Command
            </p>
            <div className="font-mono bg-muted px-3 py-2 rounded text-sm">
              {installCommand}
            </div>
          </div>
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  variant="secondary"
                  size="sm"
                  onClick={() => copyToClipboard(installCommand)}
                  className="ml-4"
                >
                  <Copy className="h-3.5 w-3.5" />
                  Copy
                </Button>
              </TooltipTrigger>
              <TooltipContent>Copy install command</TooltipContent>
            </Tooltip>
          </TooltipProvider>
        </div>
      </Card>

      {/* Plugin Details */}
      <Card className="p-4">
        <h3 className="text-lg font-semibold">Plugin Details</h3>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6 mt-4">
          <div>
            <p className="text-muted-foreground text-xs">Plugin ID</p>
            <div className="flex items-center gap-2 mt-1">
              <span className="font-mono text-xs">{plugin.id}</span>
              <Copy
                className="h-3 w-3 cursor-pointer text-muted-foreground hover:text-primary"
                onClick={() => copyToClipboard(plugin.id)}
              />
            </div>
          </div>

          <div>
            <p className="text-muted-foreground text-xs">Name</p>
            <p className="font-semibold mt-1">{plugin.name}</p>
          </div>

          <div>
            <p className="text-muted-foreground text-xs">Version</p>
            <p className="font-semibold mt-1">{plugin.version || "N/A"}</p>
          </div>

          <div className="col-span-2">
            <p className="text-muted-foreground text-xs">Source</p>
            <div className="flex items-center gap-2 mt-1">
              <p className="font-semibold">
                {getSourceDisplayText(plugin.source)}
              </p>
              {sourceLink && (
                <a
                  href={sourceLink}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-primary hover:text-primary/80"
                >
                  <ExternalLink className="h-4 w-4" />
                </a>
              )}
            </div>
          </div>

          <div>
            <p className="text-muted-foreground text-xs">Category</p>
            <div className="mt-1">
              {plugin.category ? (
                <Badge
                  className={cn("text-xs", badgeClasses(categoryBadgeColor))}
                >
                  {plugin.category}
                </Badge>
              ) : (
                <p className="text-muted-foreground">Uncategorized</p>
              )}
            </div>
          </div>

          {isAdmin && (
            <div className="col-span-3">
              <p className="text-muted-foreground text-xs">Status</p>
              <div className="flex items-center gap-3 mt-2">
                <Switch
                  checked={plugin.enabled}
                  disabled={isToggling}
                  onCheckedChange={handleToggleEnabled}
                />
                <p className="text-sm">
                  {plugin.enabled
                    ? "Plugin is enabled and visible in marketplace"
                    : "Plugin is disabled and hidden from marketplace"}
                </p>
              </div>
            </div>
          )}
        </div>
      </Card>

      {plugin.description && (
        <Card className="p-4">
          <h3 className="text-lg font-semibold">Description</h3>
          <p className="mt-2">{plugin.description}</p>
        </Card>
      )}

      {plugin.keywords && plugin.keywords.length > 0 && (
        <Card className="p-4">
          <h3 className="text-lg font-semibold">Keywords</h3>
          <div className="flex flex-wrap gap-2 mt-2">
            {plugin.keywords.map((keyword, index) => (
              <Badge
                key={index}
                className={cn("text-xs", badgeClasses("gray"))}
              >
                {keyword}
              </Badge>
            ))}
          </div>
        </Card>
      )}

      {plugin.author && (
        <Card className="p-4">
          <h3 className="text-lg font-semibold">Author Information</h3>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 mt-4">
            {plugin.author.name && (
              <div>
                <p className="text-muted-foreground text-xs">Name</p>
                <p className="font-semibold mt-1">{plugin.author.name}</p>
              </div>
            )}
            {plugin.author.email && (
              <div>
                <p className="text-muted-foreground text-xs">Email</p>
                <p className="font-semibold mt-1">
                  <a
                    href={`mailto:${plugin.author.email}`}
                    className="text-primary hover:text-primary/80"
                  >
                    {plugin.author.email}
                  </a>
                </p>
              </div>
            )}
          </div>
        </Card>
      )}

      {plugin.homepage && (
        <Card className="p-4">
          <h3 className="text-lg font-semibold">Homepage</h3>
          <a
            href={plugin.homepage}
            target="_blank"
            rel="noopener noreferrer"
            className="text-primary hover:text-primary/80 flex items-center gap-2 mt-2"
          >
            {plugin.homepage}
            <ExternalLink className="h-4 w-4" />
          </a>
        </Card>
      )}

      <Card className="p-4">
        <h3 className="text-lg font-semibold">Metadata</h3>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 mt-4">
          <div>
            <p className="text-muted-foreground text-xs">Created At</p>
            <p className="font-semibold mt-1">
              {formatDateString(plugin.created_at)}
            </p>
          </div>
          <div>
            <p className="text-muted-foreground text-xs">Updated At</p>
            <p className="font-semibold mt-1">
              {formatDateString(plugin.updated_at)}
            </p>
          </div>
          {plugin.created_by && (
            <div className="col-span-2">
              <p className="text-muted-foreground text-xs">Created By</p>
              <p className="font-semibold mt-1">{plugin.created_by}</p>
            </div>
          )}
        </div>
      </Card>
    </div>
  );
};

export default PluginInfoView;
