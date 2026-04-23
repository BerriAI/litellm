import { ColumnDef } from "@tanstack/react-table";
import { Button, Badge, Text } from "@tremor/react";
import { Tooltip } from "antd";
import { CopyOutlined } from "@ant-design/icons";
import { MarketplacePluginEntry } from "@/components/claude_code_plugins/types";
import {
  formatInstallCommand,
  getCategoryBadgeColor,
  getSourceDisplayText,
} from "@/components/claude_code_plugins/helpers";

export const getMarketplaceTableColumns = (
  copyToClipboard: (text: string) => void,
  publicPage: boolean = false,
): ColumnDef<MarketplacePluginEntry>[] => {
  const allColumns: ColumnDef<MarketplacePluginEntry>[] = [
    {
      header: "Plugin Name",
      accessorKey: "name",
      enableSorting: true,
      sortingFn: "alphanumeric",
      cell: ({ row }) => {
        const plugin = row.original;
        const installCommand = formatInstallCommand(plugin);

        return (
          <div className="space-y-1">
            <div className="flex items-center space-x-2">
              <Text className="font-medium text-sm">{plugin.name}</Text>
              <Tooltip title="Copy install command">
                <CopyOutlined
                  onClick={() => copyToClipboard(installCommand)}
                  className="cursor-pointer text-gray-500 hover:text-blue-500 text-xs"
                />
              </Tooltip>
            </div>
            {/* Show description on mobile */}
            <div className="md:hidden">
              <Text className="text-xs text-gray-600">
                {plugin.description || "No description"}
              </Text>
            </div>
          </div>
        );
      },
    },
    {
      header: "Description",
      accessorKey: "description",
      enableSorting: true,
      sortingFn: "alphanumeric",
      cell: ({ row }) => {
        const plugin = row.original;

        return (
          <Text className="text-xs line-clamp-2">
            {plugin.description || "-"}
          </Text>
        );
      },
      meta: {
        className: "hidden md:table-cell",
      },
    },
    {
      header: "Version",
      accessorKey: "version",
      enableSorting: true,
      sortingFn: "alphanumeric",
      cell: ({ row }) => {
        const plugin = row.original;

        return plugin.version ? (
          <Badge color="blue" size="sm">
            v{plugin.version}
          </Badge>
        ) : (
          <Text className="text-xs text-gray-400">-</Text>
        );
      },
      meta: {
        className: "hidden lg:table-cell",
      },
    },
    {
      header: "Category",
      accessorKey: "category",
      enableSorting: true,
      sortingFn: "alphanumeric",
      cell: ({ row }) => {
        const plugin = row.original;
        const badgeColor = getCategoryBadgeColor(plugin.category);

        return plugin.category ? (
          <Badge color={badgeColor} size="sm">
            {plugin.category}
          </Badge>
        ) : (
          <Badge color="gray" size="sm">
            Uncategorized
          </Badge>
        );
      },
      meta: {
        className: "hidden lg:table-cell",
      },
    },
    {
      header: "Source",
      accessorKey: "source",
      enableSorting: false,
      cell: ({ row }) => {
        const plugin = row.original;
        const sourceText = getSourceDisplayText(plugin.source);

        return <Text className="text-xs text-gray-600">{sourceText}</Text>;
      },
      meta: {
        className: "hidden xl:table-cell",
      },
    },
    {
      header: "Keywords",
      accessorKey: "keywords",
      enableSorting: false,
      cell: ({ row }) => {
        const plugin = row.original;
        const keywords = plugin.keywords?.slice(0, 3) || [];
        const remaining = (plugin.keywords?.length || 0) - 3;

        return (
          <div className="flex flex-wrap gap-1">
            {keywords.map((keyword, index) => (
              <Badge key={index} color="gray" size="xs">
                {keyword}
              </Badge>
            ))}
            {remaining > 0 && (
              <Badge color="gray" size="xs">
                +{remaining}
              </Badge>
            )}
          </div>
        );
      },
      meta: {
        className: "hidden xl:table-cell",
      },
    },
    {
      header: "Install Command",
      id: "install_command",
      enableSorting: false,
      cell: ({ row }) => {
        const plugin = row.original;
        const installCommand = formatInstallCommand(plugin);

        return (
          <div className="flex items-center space-x-2">
            <code className="text-xs bg-gray-100 px-2 py-1 rounded font-mono truncate max-w-[200px]">
              {installCommand}
            </code>
            <Tooltip title="Copy command">
              <Button
                size="xs"
                variant="secondary"
                icon={CopyOutlined}
                onClick={() => copyToClipboard(installCommand)}
              />
            </Tooltip>
          </div>
        );
      },
    },
  ];

  return allColumns;
};
