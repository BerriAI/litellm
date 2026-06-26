"use client";

import React, { useState } from "react";
import { Flex, Table, Tabs, Tag, Tooltip, Typography, Button } from "antd";
import type { ColumnsType } from "antd/es/table";
import { BranchesOutlined, DeleteOutlined, EditOutlined, CodeOutlined } from "@ant-design/icons";
import { Trans, useTranslation } from "react-i18next";
import type { TFunction } from "i18next";
import type { RoutingGroup } from "./types";

const { Text, Paragraph } = Typography;

interface RoutingGroupsTableProps {
  groups: RoutingGroup[];
  loading?: boolean;
  onEdit: (group: RoutingGroup) => void;
  onDelete: (group: RoutingGroup) => void;
  proxyBaseUrl?: string;
}

const formatStrategyLabel = (t: TFunction, strategy: string): string => {
  switch (strategy) {
    case "simple-shuffle":
      return t("routingGroups.routingGroupsTable.strategySimpleShuffle");
    case "least-busy":
      return t("routingGroups.routingGroupsTable.strategyLeastBusy");
    case "usage-based-routing":
      return t("routingGroups.routingGroupsTable.strategyUsageBased");
    case "latency-based-routing":
      return t("routingGroups.routingGroupsTable.strategyLatencyBased");
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
  const { t } = useTranslation();
  const snippets = {
    curl: buildCurlSnippet(group, baseUrl),
    python: buildPythonSnippet(group, baseUrl),
    javascript: buildJsSnippet(group, baseUrl),
  } as const;
  type SnippetKey = keyof typeof snippets;
  const [activeKey, setActiveKey] = useState<SnippetKey>("curl");

  const items = [
    { key: "curl", label: t("routingGroups.routingGroupsTable.tabCurl") },
    { key: "python", label: t("routingGroups.routingGroupsTable.tabPython") },
    { key: "javascript", label: t("routingGroups.routingGroupsTable.tabJavascript") },
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
          copyable={{ text: snippets[activeKey], tooltips: [t("common.copy"), t("common.copied")] }}
          className="!mb-0"
        />
      }
    />
  );
};

const RoutingGroupsTable: React.FC<RoutingGroupsTableProps> = ({ groups, loading, onEdit, onDelete, proxyBaseUrl }) => {
  const { t } = useTranslation();
  const [expandedRowKeys, setExpandedRowKeys] = useState<React.Key[]>([]);
  const baseUrl = resolveBaseUrl(proxyBaseUrl);

  const columns: ColumnsType<RoutingGroup> = [
    {
      title: t("routingGroups.routingGroupsTable.colGroupName"),
      dataIndex: "group_name",
      key: "group_name",
      render: (name: string) => (
        <Text strong className="text-blue-600">
          {name}
        </Text>
      ),
    },
    {
      title: t("routingGroups.routingGroupsTable.colModels"),
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
      title: t("routingGroups.routingGroupsTable.colStrategy"),
      dataIndex: "routing_strategy",
      key: "routing_strategy",
      render: (strategy: string) => (
        <span className="inline-flex items-center gap-1.5">
          <BranchesOutlined className="text-gray-400" />
          <Text>{formatStrategyLabel(t, strategy)}</Text>
        </span>
      ),
    },
    {
      title: t("common.actions"),
      key: "actions",
      width: 120,
      align: "right",
      render: (_, group) => (
        <Flex justify="flex-end" align="center" gap={8}>
          <Tooltip title={t("common.edit")}>
            <Button
              type="text"
              icon={<EditOutlined />}
              onClick={(e) => {
                e.stopPropagation();
                onEdit(group);
              }}
            />
          </Tooltip>
          <Tooltip title={t("common.delete")}>
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
              <Text strong>{t("routingGroups.routingGroupsTable.howRoutingWorksTitle")}</Text>
            </Flex>
            <Paragraph className="text-sm text-gray-600 mb-3">
              <Trans
                i18nKey="routingGroups.routingGroupsTable.howRoutingWorksDesc"
                values={{ strategy: formatStrategyLabel(t, group.routing_strategy) }}
                components={{ strong: <Text strong /> }}
              />
            </Paragraph>
            <RoutingGroupSnippet group={group} baseUrl={baseUrl} />
          </div>
        ),
      }}
    />
  );
};

export default RoutingGroupsTable;
