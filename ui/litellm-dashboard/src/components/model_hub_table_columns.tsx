import { ColumnDef } from "@tanstack/react-table";
import { Button, Badge, Text } from "@tremor/react";
import { Tooltip, Tag } from "antd";
import { CopyOutlined, InfoCircleOutlined } from "@ant-design/icons";

interface ModelHubData {
  model_group: string;
  providers: string[];
  max_input_tokens?: number;
  max_output_tokens?: number;
  input_cost_per_token?: number;
  output_cost_per_token?: number;
  mode?: string;
  tpm?: number;
  rpm?: number;
  supports_parallel_function_calling: boolean;
  supports_vision: boolean;
  supports_function_calling: boolean;
  supported_openai_params?: string[];
  is_public_model_group: boolean;
  [key: string]: any;
}

const formatCapabilityName = (key: string) => {
  return key
    .replace(/^supports_/, "")
    .split("_")
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");
};

const getModelCapabilities = (model: ModelHubData) => {
  return Object.entries(model)
    .filter(([key, value]) => key.startsWith("supports_") && value === true)
    .map(([key]) => key);
};

const formatCost = (cost: number) => {
  return `$${(cost * 1_000_000).toFixed(2)}`;
};

const formatTokens = (tokens: number) => {
  if (tokens >= 1_000_000) {
    return `${(tokens / 1_000_000).toFixed(1)}M`;
  } else if (tokens >= 1_000) {
    return `${(tokens / 1_000).toFixed(1)}K`;
  }
  return tokens.toString();
};

export const modelHubColumns = (
  showModal: (model: ModelHubData) => void,
  copyToClipboard: (text: string) => void,
  publicPage: boolean = false,
): ColumnDef<ModelHubData>[] => {
  const allColumns: ColumnDef<ModelHubData>[] = [
    {
      header: "Public Model Name",
      accessorKey: "model_group",
      enableSorting: true,
      sortingFn: "alphanumeric",
      cell: ({ row }) => {
        const model = row.original;

        return (
          <div className="space-y-1">
            <div className="flex items-center space-x-2">
              <Text className="font-medium text-sm">{model.model_group}</Text>
              <Tooltip title="Copy model name">
                <CopyOutlined
                  onClick={() => copyToClipboard(model.model_group)}
                  className="cursor-pointer text-gray-500 hover:text-blue-500 text-xs"
                />
              </Tooltip>
            </div>
            {/* Show provider on mobile when provider column is hidden */}
            <div className="md:hidden">
              <Text className="text-xs text-gray-600">{model.providers.join(", ")}</Text>
            </div>
          </div>
        );
      },
    },
    {
      header: "Provider",
      accessorKey: "providers",
      enableSorting: true,
      sortingFn: (rowA, rowB) => {
        const providersA = rowA.original.providers.join(", ");
        const providersB = rowB.original.providers.join(", ");
        return providersA.localeCompare(providersB);
      },
      cell: ({ row }) => {
        const model = row.original;

        return (
          <div className="flex flex-wrap gap-1">
            {model.providers.slice(0, 2).map((provider) => (
              <Tag key={provider} color="blue" className="text-xs">
                {provider}
              </Tag>
            ))}
            {model.providers.length > 2 && <Text className="text-xs text-gray-500">+{model.providers.length - 2}</Text>}
          </div>
        );
      },
      meta: {
        className: "hidden md:table-cell",
      },
    },
    {
      header: "Mode",
      accessorKey: "mode",
      enableSorting: true,
      sortingFn: "alphanumeric",
      cell: ({ row }) => {
        const model = row.original;

        return model.mode ? (
          <Badge color="green" size="sm">
            {model.mode}
          </Badge>
        ) : (
          <Text className="text-gray-500">-</Text>
        );
      },
      meta: {
        className: "hidden lg:table-cell",
      },
    },
    {
      header: "Tokens",
      accessorKey: "max_input_tokens",
      enableSorting: true,
      sortingFn: (rowA, rowB) => {
        const tokensA = (rowA.original.max_input_tokens || 0) + (rowA.original.max_output_tokens || 0);
        const tokensB = (rowB.original.max_input_tokens || 0) + (rowB.original.max_output_tokens || 0);
        return tokensA - tokensB;
      },
      cell: ({ row }) => {
        const model = row.original;

        return (
          <div className="space-y-1">
            <Text className="text-xs">
              {model.max_input_tokens ? formatTokens(model.max_input_tokens) : "-"} /{" "}
              {model.max_output_tokens ? formatTokens(model.max_output_tokens) : "-"}
            </Text>
          </div>
        );
      },
      meta: {
        className: "hidden lg:table-cell",
      },
    },
    {
      header: "Cost/1M",
      accessorKey: "input_cost_per_token",
      enableSorting: true,
      sortingFn: (rowA, rowB) => {
        const costA = (rowA.original.input_cost_per_token || 0) + (rowA.original.output_cost_per_token || 0);
        const costB = (rowB.original.input_cost_per_token || 0) + (rowB.original.output_cost_per_token || 0);
        return costA - costB;
      },
      cell: ({ row }) => {
        const model = row.original;

        return (
          <div className="space-y-1">
            <Text className="text-xs">{model.input_cost_per_token ? formatCost(model.input_cost_per_token) : "-"}</Text>
            <Text className="text-xs text-gray-500">
              {model.output_cost_per_token ? formatCost(model.output_cost_per_token) : "-"}
            </Text>
          </div>
        );
      },
    },
    {
      header: "Features",
      accessorKey: "capabilities",
      enableSorting: false,
      cell: ({ row }) => {
        const model = row.original;
        const capabilities = getModelCapabilities(model);
        const colors = ["green", "blue", "purple", "orange", "red", "yellow"];

        return (
          <div className="flex flex-wrap gap-1">
            {capabilities.length === 0 ? (
              <Text className="text-gray-500 text-xs">-</Text>
            ) : (
              capabilities.map((capability, index) => (
                <Badge key={capability} color={colors[index % colors.length]} size="xs">
                  {formatCapabilityName(capability)}
                </Badge>
              ))
            )}
          </div>
        );
      },
    },
    {
      header: "Public",
      accessorKey: "is_public_model_group",
      enableSorting: true,
      sortingFn: (rowA, rowB) => {
        const publicA = rowA.original.is_public_model_group === true ? 1 : 0;
        const publicB = rowB.original.is_public_model_group === true ? 1 : 0;
        return publicA - publicB;
      },
      cell: ({ row }) => {
        const model = row.original;

        return model.is_public_model_group === true ? (
          <Badge color="green" size="xs">
            Yes
          </Badge>
        ) : (
          <Badge color="gray" size="xs">
            No
          </Badge>
        );
      },
      meta: {
        className: "hidden md:table-cell",
      },
    },
    {
      header: "Details",
      id: "details",
      enableSorting: false,
      cell: ({ row }) => {
        const model = row.original;

        return (
          <Button size="xs" variant="secondary" onClick={() => showModal(model)} icon={InfoCircleOutlined}>
            <span className="hidden lg:inline">Details</span>
            <span className="lg:hidden">Info</span>
          </Button>
        );
      },
    },
  ];

  // Filter out columns based on publicPage setting
  if (publicPage) {
    return allColumns.filter((column) => {
      // Remove the public column
      if ("accessorKey" in column && column.accessorKey === "is_public_model_group") return false;

      return true;
    });
  }

  return allColumns;
};
