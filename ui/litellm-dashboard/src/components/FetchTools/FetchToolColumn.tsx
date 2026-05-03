import { Tag } from "antd";
import { ColumnsType } from "antd/es/table";
import TableIconActionButton from "../common_components/IconActionButton/TableIconActionButtons/TableIconActionButton";
import { FetchTool } from "./types";

export const fetchToolColumns = (
  onView: (fetchToolId: string) => void,
  onEdit: (fetchToolId: string) => void,
  onDelete: (fetchToolId: string) => void,
  availableProviders: Array<{ provider_name: string; ui_friendly_name: string }>,
): ColumnsType<FetchTool> => [
  {
    title: "Fetch Tool ID",
    dataIndex: "fetch_tool_id",
    key: "fetch_tool_id",
    render: (_, tool) => {
      const isFromConfig = tool.is_from_config;

      if (isFromConfig) {
        return <span className="text-xs">-</span>;
      }

      return (
        <button
          onClick={() => onView(tool.fetch_tool_id!)}
          className="font-mono text-blue-500 bg-blue-50 hover:bg-blue-100 text-xs font-normal px-2 py-0.5 text-left cursor-pointer max-w-40"
        >
          <span className="truncate block">{tool.fetch_tool_id}</span>
        </button>
      );
    },
  },
  {
    title: "Name",
    dataIndex: "fetch_tool_name",
    key: "fetch_tool_name",
    render: (name: string) => <span className="font-medium">{name}</span>,
  },
  {
    title: "Provider",
    key: "provider",
    render: (_, tool) => {
      const provider = tool.litellm_params.fetch_provider;
      const providerInfo = availableProviders.find((p) => p.provider_name === provider);
      const displayName = providerInfo?.ui_friendly_name || provider;

      return <span className="text-sm">{displayName}</span>;
    },
  },
  {
    title: "Created At",
    dataIndex: "created_at",
    key: "created_at",
    render: (_, tool) => {
      return <span className="text-xs">{tool.created_at ? new Date(tool.created_at).toLocaleDateString() : "-"}</span>;
    },
  },
  {
    title: "Updated At",
    dataIndex: "updated_at",
    key: "updated_at",
    render: (_, tool) => {
      return <span className="text-xs">{tool.updated_at ? new Date(tool.updated_at).toLocaleDateString() : "-"}</span>;
    },
  },
  {
    title: "Source",
    key: "source",
    render: (_, tool) => {
      const isFromConfig = tool.is_from_config ?? false;

      return <Tag color={isFromConfig ? "default" : "blue"}>{isFromConfig ? "Config" : "DB"}</Tag>;
    },
  },
  {
    title: "Actions",
    key: "actions",
    render: (_, tool) => {
      const toolId = tool.fetch_tool_id;
      const isFromConfig = tool.is_from_config ?? false;

      return (
        <div className="flex items-center gap-2">
          <TableIconActionButton
            variant="Edit"
            tooltipText="Edit fetch tool"
            disabled={isFromConfig}
            disabledTooltipText="Config fetch tool cannot be edited on the dashboard. Please edit it from the config file."
            onClick={() => {
              if (toolId && !isFromConfig) {
                onEdit(toolId);
              }
            }}
          />
          <TableIconActionButton
            variant="Delete"
            tooltipText="Delete fetch tool"
            disabled={isFromConfig}
            disabledTooltipText="Config fetch tool cannot be deleted on the dashboard. Please delete it from the config file."
            onClick={() => {
              if (toolId && !isFromConfig) {
                onDelete(toolId);
              }
            }}
          />
        </div>
      );
    },
  },
];
