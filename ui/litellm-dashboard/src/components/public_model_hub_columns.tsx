import React from "react";
import { ColumnDef } from "@tanstack/react-table";
import { Text } from "@tremor/react";
import { EyeIcon, CogIcon } from "@heroicons/react/outline";
import { Tag } from "antd";

interface ModelGroupInfo {
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
  [key: string]: any;
}

const formatCost = (cost: number) => {
  return `$${(cost * 1_000_000).toFixed(4)}`;
};

const formatTokens = (tokens: number | undefined) => {
  if (!tokens) return "N/A";
  if (tokens >= 1000) {
    return `${(tokens / 1000).toFixed(0)}K`;
  }
  return tokens.toString();
};

const formatLimits = (rpm?: number, tpm?: number) => {
  const limits = [];
  if (rpm) limits.push(`RPM: ${rpm.toLocaleString()}`);
  if (tpm) limits.push(`TPM: ${tpm.toLocaleString()}`);
  return limits.length > 0 ? limits.join(", ") : "N/A";
};

export const publicModelHubColumns = (): ColumnDef<ModelGroupInfo>[] => [
  {
    header: "#",
    id: "index",
    enableSorting: false,
    cell: ({ row }) => {
      const index = row.index + 1;
      return <Text className="text-center">{index}</Text>;
    },
    size: 50,
  },
  {
    header: "Model Group",
    accessorKey: "model_group",
    enableSorting: true,
    cell: ({ row }) => <Text className="font-medium">{row.original.model_group}</Text>,
    size: 150,
  },
  {
    header: "Providers",
    accessorKey: "providers",
    enableSorting: true,
    cell: ({ row }) => {
      const providers = row.original.providers;
      const getProviderColor = (provider: string) => {
        switch (provider.toLowerCase()) {
          case "openai":
            return "green";
          case "anthropic":
            return "orange";
          case "cohere":
            return "blue";
          default:
            return "gray";
        }
      };

      return (
        <div className="flex flex-wrap gap-1">
          {providers.map((provider) => (
            <Tag key={provider} color={getProviderColor(provider)} className="text-xs">
              {provider}
            </Tag>
          ))}
        </div>
      );
    },
    size: 120,
  },
  {
    header: "Mode",
    accessorKey: "mode",
    enableSorting: true,
    cell: ({ row }) => {
      const mode = row.original.mode;
      const getModeIcon = (mode: string) => {
        switch (mode?.toLowerCase()) {
          case "chat":
            return "💬";
          case "rerank":
            return "🔄";
          case "embedding":
            return "📄";
          default:
            return "🤖";
        }
      };

      return (
        <div className="flex items-center space-x-2">
          <span>{getModeIcon(mode || "")}</span>
          <Text>{mode || "Chat"}</Text>
        </div>
      );
    },
    size: 100,
  },
  {
    header: "Max Input",
    accessorKey: "max_input_tokens",
    enableSorting: true,
    cell: ({ row }) => <Text className="text-center">{formatTokens(row.original.max_input_tokens)}</Text>,
    size: 100,
  },
  {
    header: "Max Output",
    accessorKey: "max_output_tokens",
    enableSorting: true,
    cell: ({ row }) => <Text className="text-center">{formatTokens(row.original.max_output_tokens)}</Text>,
    size: 100,
  },
  {
    header: "Input $/1K",
    accessorKey: "input_cost_per_token",
    enableSorting: true,
    cell: ({ row }) => {
      const cost = row.original.input_cost_per_token;
      return <Text className="text-center">{cost ? formatCost(cost) : "Free"}</Text>;
    },
    size: 100,
  },
  {
    header: "Output $/1K",
    accessorKey: "output_cost_per_token",
    enableSorting: true,
    cell: ({ row }) => {
      const cost = row.original.output_cost_per_token;
      return <Text className="text-center">{cost ? formatCost(cost) : "Free"}</Text>;
    },
    size: 100,
  },
  {
    header: "Features",
    accessorKey: "supports_vision",
    enableSorting: false,
    cell: ({ row }) => {
      const model = row.original;
      const features = [];

      if (model.supports_vision) {
        features.push(
          <div key="vision" className="flex items-center space-x-1" title="Vision">
            <EyeIcon className="w-4 h-4 text-blue-600" />
          </div>,
        );
      }

      if (model.supports_function_calling || model.supports_parallel_function_calling) {
        features.push(
          <div key="functions" className="flex items-center space-x-1" title="Functions">
            <CogIcon className="w-4 h-4 text-green-600" />
          </div>,
        );
      }

      return features.length > 0 ? (
        <div className="flex space-x-2">{features}</div>
      ) : (
        <Text className="text-gray-400">-</Text>
      );
    },
    size: 100,
  },
  {
    header: "Limits",
    accessorKey: "rpm",
    enableSorting: true,
    cell: ({ row }) => {
      const model = row.original;
      return <Text className="text-xs text-gray-600">{formatLimits(model.rpm, model.tpm)}</Text>;
    },
    size: 150,
  },
];
