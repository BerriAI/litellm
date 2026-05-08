"use client";

import React, { useState } from "react";
import { Flex, Table, Tabs, Tag, Tooltip, Typography, Button } from "antd";
import type { ColumnsType } from "antd/es/table";
import { BranchesOutlined, DeleteOutlined, EditOutlined, CodeOutlined } from "@ant-design/icons";
import type { RoutingGroup } from "./types";

const { Text, Paragraph } = Typography;

interface RoutingGroupsTableProps {
  groups: RoutingGroup[];
  loading?: boolean;
  onEdit: (group: RoutingGroup) => void;
  onDelete: (group: RoutingGroup) => void;
  proxyBaseUrl?: string;
}

const formatStrategyLabel = (strategy: string): string => {
  switch (strategy) {
    case "simple-shuffle":
      return "Simple Shuffle";
    case "least-busy":
      return "Least Busy";
    case "usage-based-routing":
      return "Usage Based";
    case "latency-based-routing":
      return "Latency Based";
    default:
      return strategy;
  }
};

const resolveBaseUrl = (proxyBaseUrl?: string): string => {
  if (proxyBaseUrl && proxyBaseUrl.trim()) return proxyBaseUrl;
  if (typeof window !== "undefined" && window.location?.origin) return window.location.origin;
  return "<your_proxy_base_url>";
};

const exampleModel = (group: RoutingGroup): string => group.models[0] ?? "<your-model>";

const buildCurlSnippet = (group: RoutingGroup, baseUrl: string): string =>
  `curl -X POST '${baseUrl}/v1/chat/completions' \\
  -H 'Content-Type: application/json' \\
  -H 'Authorization: Bearer $LITELLM_API_KEY' \\
  -d '{
    "model": "${exampleModel(group)}",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'`;

const buildPythonSnippet = (group: RoutingGroup, baseUrl: string): string =>
  `from openai import OpenAI

client = OpenAI(
    api_key="$LITELLM_API_KEY",
    base_url="${baseUrl}",
)

response = client.chat.completions.create(
    model="${exampleModel(group)}",
    messages=[{"role": "user", "content": "Hello!"}],
)

print(response)`;

const buildJsSnippet = (group: RoutingGroup, baseUrl: string): string =>
  `import OpenAI from "openai";

const client = new OpenAI({
  apiKey: process.env.LITELLM_API_KEY,
  baseURL: "${baseUrl}",
});

const response = await client.chat.completions.create({
  model: "${exampleModel(group)}",
  messages: [{ role: "user", content: "Hello!" }],
});

console.log(response);`;

interface RoutingGroupSnippetProps {
  group: RoutingGroup;
  baseUrl: string;
}

const SNIPPET_BLOCK_STYLE: React.CSSProperties = {
  backgroundColor: "#111827",
  color: "#f3f4f6",
  borderRadius: 6,
  padding: 16,
  fontSize: 12,
  whiteSpace: "pre",
  overflowX: "auto",
};

const RoutingGroupSnippet: React.FC<RoutingGroupSnippetProps> = ({ group, baseUrl }) => {
  const snippets = {
    curl: buildCurlSnippet(group, baseUrl),
    python: buildPythonSnippet(group, baseUrl),
    javascript: buildJsSnippet(group, baseUrl),
  } as const;
  type SnippetKey = keyof typeof snippets;
  const [activeKey, setActiveKey] = useState<SnippetKey>("curl");

  const items = [
    { key: "curl", label: "cURL" },
    { key: "python", label: "Python (OpenAI SDK)" },
    { key: "javascript", label: "JavaScript (OpenAI SDK)" },
  ].map(({ key, label }) => ({
    key,
    label,
    children: (
      <Paragraph code className="!mb-0" style={SNIPPET_BLOCK_STYLE}>
        {snippets[key as SnippetKey]}
      </Paragraph>
    ),
  }));

  return (
    <Tabs
      size="small"
      activeKey={activeKey}
      onChange={(k) => setActiveKey(k as SnippetKey)}
      items={items}
      tabBarExtraContent={
        <Paragraph
          copyable={{ text: snippets[activeKey], tooltips: ["Copy", "Copied"] }}
          className="!mb-0"
        />
      }
    />
  );
};

const RoutingGroupsTable: React.FC<RoutingGroupsTableProps> = ({
  groups,
  loading,
  onEdit,
  onDelete,
  proxyBaseUrl,
}) => {
  const [expandedRowKeys, setExpandedRowKeys] = useState<React.Key[]>([]);
  const baseUrl = resolveBaseUrl(proxyBaseUrl);

  const columns: ColumnsType<RoutingGroup> = [
    {
      title: "GROUP NAME",
      dataIndex: "group_name",
      key: "group_name",
      render: (name: string) => (
        <Text strong className="text-blue-600">
          {name}
        </Text>
      ),
    },
    {
      title: "MODELS",
      dataIndex: "models",
      key: "models",
      render: (models: string[]) => (
        <Flex wrap="wrap" gap={4}>
          {models.map((m) => (
            <Tag key={m}>{m}</Tag>
          ))}
        </Flex>
      ),
    },
    {
      title: "STRATEGY",
      dataIndex: "routing_strategy",
      key: "routing_strategy",
      render: (strategy: string) => (
        <span className="inline-flex items-center gap-1.5">
          <BranchesOutlined className="text-gray-400" />
          <Text>{formatStrategyLabel(strategy)}</Text>
        </span>
      ),
    },
    {
      title: "ACTIONS",
      key: "actions",
      width: 120,
      align: "right",
      render: (_, group) => (
        <Flex justify="flex-end" align="center" gap={8}>
          <Tooltip title="Edit">
            <Button
              type="text"
              icon={<EditOutlined />}
              onClick={(e) => {
                e.stopPropagation();
                onEdit(group);
              }}
            />
          </Tooltip>
          <Tooltip title="Delete">
            <Button
              type="text"
              danger
              icon={<DeleteOutlined />}
              onClick={(e) => {
                e.stopPropagation();
                onDelete(group);
              }}
            />
          </Tooltip>
        </Flex>
      ),
    },
  ];

  return (
    <Table<RoutingGroup>
      rowKey="group_name"
      columns={columns}
      dataSource={groups}
      loading={loading}
      pagination={false}
      expandable={{
        expandedRowKeys,
        onExpandedRowsChange: (keys) => setExpandedRowKeys([...keys]),
        expandedRowRender: (group) => (
          <div className="bg-gray-50 border border-gray-200 rounded-md p-4 my-2">
            <Flex align="center" gap={8} className="mb-2">
              <CodeOutlined className="text-blue-500" />
              <Text strong>How routing works for this group</Text>
            </Flex>
            <Paragraph className="text-sm text-gray-600 mb-3">
              Callers request any model in the group by name — LiteLLM picks a deployment behind the
              scenes using the{" "}
              <Text strong>{formatStrategyLabel(group.routing_strategy)}</Text> strategy.
            </Paragraph>
            <RoutingGroupSnippet group={group} baseUrl={baseUrl} />
          </div>
        ),
      }}
    />
  );
};

export default RoutingGroupsTable;
