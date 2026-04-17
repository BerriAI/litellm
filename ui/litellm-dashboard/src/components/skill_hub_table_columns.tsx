import { ColumnDef } from "@tanstack/react-table";
import { Badge, Text } from "@tremor/react";
import { Tooltip } from "antd";
import { CopyOutlined, LinkOutlined } from "@ant-design/icons";
import { Plugin } from "./claude_code_plugins/types";

export const skillHubColumns = (
  showModal: (skill: Plugin) => void,
  copyToClipboard: (text: string) => void,
  publicPage: boolean = false,
): ColumnDef<Plugin>[] => [
  {
    header: "Skill Name",
    accessorKey: "name",
    enableSorting: true,
    sortingFn: "alphanumeric",
    cell: ({ row }) => {
      const skill = row.original;
      return (
        <div className="space-y-1">
          <div className="flex items-center space-x-2">
            <button
              type="button"
              className="font-medium text-sm cursor-pointer text-blue-600 hover:underline bg-transparent border-none p-0"
              onClick={() => showModal(skill)}
            >
              {skill.name}
            </button>
            <Tooltip title="Copy skill name">
              <CopyOutlined
                onClick={() => copyToClipboard(skill.name)}
                className="cursor-pointer text-gray-500 hover:text-blue-500 text-xs"
              />
            </Tooltip>
          </div>
          {skill.description && (
            <Text className="text-xs text-gray-500 line-clamp-1 md:hidden">
              {skill.description}
            </Text>
          )}
        </div>
      );
    },
  },
  {
    header: "Description",
    accessorKey: "description",
    enableSorting: false,
    cell: ({ row }) => (
      <Text className="text-xs line-clamp-2">{row.original.description || "-"}</Text>
    ),
  },
  {
    header: "Category",
    accessorKey: "category",
    enableSorting: true,
    cell: ({ row }) => {
      const cat = row.original.category;
      if (!cat) return <Text className="text-xs text-gray-400">-</Text>;
      return <Badge color="blue" size="xs">{cat}</Badge>;
    },
  },
  {
    header: "Domain",
    accessorKey: "domain",
    enableSorting: true,
    cell: ({ row }) => (
      <Text className="text-xs">{row.original.domain || "-"}</Text>
    ),
  },
  {
    header: "Source",
    accessorKey: "source",
    enableSorting: false,
    cell: ({ row }) => {
      const src = row.original.source;
      let url: string | null = null;
      let label = "-";
      if (src?.source === "github" && src.repo) {
        url = `https://github.com/${src.repo}`;
        label = src.repo;
      } else if (src?.source === "git-subdir" && src.url) {
        url = src.path ? `${src.url}/tree/main/${src.path}` : src.url;
        label = url.replace("https://github.com/", "");
      } else if (src?.source === "url" && src.url) {
        url = src.url;
        label = src.url.replace(/^https?:\/\//, "");
      }
      if (!url) return <Text className="text-xs text-gray-400">-</Text>;
      return (
        <a
          href={url}
          target="_blank"
          rel="noopener noreferrer"
          className="flex items-center gap-1 text-xs text-blue-600 hover:underline truncate max-w-[180px]"
          title={label}
        >
          <span className="truncate">{label}</span>
          <LinkOutlined className="shrink-0" style={{ fontSize: 10 }} />
        </a>
      );
    },
  },
  {
    header: "Status",
    accessorKey: "enabled",
    enableSorting: true,
    cell: ({ row }) => (
      <Badge color={row.original.enabled ? "green" : "gray"} size="xs">
        {row.original.enabled ? "Public" : "Draft"}
      </Badge>
    ),
  },
];
