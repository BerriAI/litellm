import { ColumnDef } from "@tanstack/react-table";
import { Button, Badge, Text } from "@tremor/react";
import { Tooltip, Tag } from "antd";
import { CopyOutlined, InfoCircleOutlined } from "@ant-design/icons";
import { TFunction } from "i18next";

export interface AgentHubData {
  agent_id?: string;
  protocolVersion: string;
  name: string;
  description: string;
  url: string;
  version: string;
  capabilities?: {
    streaming?: boolean;
    [key: string]: any;
  };
  defaultInputModes?: string[];
  defaultOutputModes?: string[];
  skills?: Array<{
    id: string;
    name: string;
    description: string;
    tags?: string[];
    examples?: string[];
  }>;
  supportsAuthenticatedExtendedCard?: boolean;
  is_public?: boolean;
  [key: string]: any;
}

export const getAgentHubTableColumns = (
  t: TFunction,
  showModal: (agent: AgentHubData) => void,
  copyToClipboard: (text: string) => void,
  publicPage: boolean = false,
): ColumnDef<AgentHubData>[] => {
  const allColumns: ColumnDef<AgentHubData>[] = [
    {
      header: t("aiHub.agentHubTableColumns.colAgentName"),
      accessorKey: "name",
      enableSorting: true,
      sortingFn: "alphanumeric",
      cell: ({ row }) => {
        const agent = row.original;

        return (
          <div className="space-y-1">
            <div className="flex items-center space-x-2">
              <Text className="font-medium text-sm">{agent.name}</Text>
              <Tooltip title={t("aiHub.agentHubTableColumns.copyAgentName")}>
                <CopyOutlined
                  onClick={() => copyToClipboard(agent.name)}
                  className="cursor-pointer text-gray-500 hover:text-blue-500 text-xs"
                />
              </Tooltip>
            </div>
            {/* Show description on mobile */}
            <div className="md:hidden">
              <Text className="text-xs text-gray-600">{agent.description}</Text>
            </div>
          </div>
        );
      },
    },
    {
      header: t("common.description"),
      accessorKey: "description",
      enableSorting: true,
      sortingFn: "alphanumeric",
      cell: ({ row }) => {
        const agent = row.original;

        return <Text className="text-xs line-clamp-2">{agent.description || "-"}</Text>;
      },
      meta: {
        className: "hidden md:table-cell",
      },
    },
    {
      header: t("aiHub.agentHubTableColumns.colVersion"),
      accessorKey: "version",
      enableSorting: true,
      sortingFn: "alphanumeric",
      cell: ({ row }) => {
        const agent = row.original;

        return (
          <Badge color="blue" size="sm">
            v{agent.version}
          </Badge>
        );
      },
      meta: {
        className: "hidden lg:table-cell",
      },
    },
    {
      header: t("aiHub.agentHubTableColumns.colProtocol"),
      accessorKey: "protocolVersion",
      enableSorting: true,
      sortingFn: "alphanumeric",
      cell: ({ row }) => {
        const agent = row.original;

        return <Text className="text-xs">{agent.protocolVersion || "-"}</Text>;
      },
      meta: {
        className: "hidden lg:table-cell",
      },
    },
    {
      header: t("aiHub.agentHubTableColumns.colSkills"),
      accessorKey: "skills",
      enableSorting: false,
      cell: ({ row }) => {
        const agent = row.original;
        const skills = agent.skills || [];

        return (
          <div className="space-y-1">
            <Text className="text-xs font-medium">
              {t("aiHub.agentHubTableColumns.skillCount", { count: skills.length })}
            </Text>
            {skills.length > 0 && (
              <div className="flex flex-wrap gap-1">
                {skills.slice(0, 2).map((skill) => (
                  <Tag key={skill.id} color="purple" className="text-xs">
                    {skill.name}
                  </Tag>
                ))}
                {skills.length > 2 && <Text className="text-xs text-gray-500">+{skills.length - 2}</Text>}
              </div>
            )}
          </div>
        );
      },
    },
    {
      header: t("aiHub.agentHubTableColumns.colCapabilities"),
      accessorKey: "capabilities",
      enableSorting: false,
      cell: ({ row }) => {
        const agent = row.original;
        const capabilities = agent.capabilities || {};
        const capabilityList = Object.entries(capabilities)
          .filter(([_, value]) => value === true)
          .map(([key]) => key);

        return (
          <div className="flex flex-wrap gap-1">
            {capabilityList.length === 0 ? (
              <Text className="text-gray-500 text-xs">-</Text>
            ) : (
              capabilityList.map((capability) => (
                <Badge key={capability} color="green" size="xs">
                  {capability}
                </Badge>
              ))
            )}
          </div>
        );
      },
    },
    {
      header: t("aiHub.agentHubTableColumns.colIOModes"),
      accessorKey: "defaultInputModes",
      enableSorting: false,
      cell: ({ row }) => {
        const agent = row.original;
        const inputModes = agent.defaultInputModes || [];
        const outputModes = agent.defaultOutputModes || [];

        return (
          <div className="space-y-1">
            <Text className="text-xs">
              <span className="font-medium">{t("aiHub.agentHubTableColumns.ioIn")}</span> {inputModes.join(", ") || "-"}
            </Text>
            <Text className="text-xs">
              <span className="font-medium">{t("aiHub.agentHubTableColumns.ioOut")}</span>{" "}
              {outputModes.join(", ") || "-"}
            </Text>
          </div>
        );
      },
      meta: {
        className: "hidden xl:table-cell",
      },
    },
    {
      header: t("aiHub.agentHubTableColumns.colPublic"),
      accessorKey: "is_public",
      enableSorting: true,
      sortingFn: (rowA, rowB) => {
        const publicA = rowA.original.is_public === true ? 1 : 0;
        const publicB = rowB.original.is_public === true ? 1 : 0;
        return publicA - publicB;
      },
      cell: ({ row }) => {
        const agent = row.original;

        return agent.is_public === true ? (
          <Badge color="green" size="xs">
            {t("common.yes")}
          </Badge>
        ) : (
          <Badge color="gray" size="xs">
            {t("common.no")}
          </Badge>
        );
      },
      meta: {
        className: "hidden md:table-cell",
      },
    },
    {
      header: t("common.details"),
      id: "details",
      enableSorting: false,
      cell: ({ row }) => {
        const agent = row.original;

        return (
          <Button size="xs" variant="secondary" onClick={() => showModal(agent)} icon={InfoCircleOutlined}>
            <span className="hidden lg:inline">{t("common.details")}</span>
            <span className="lg:hidden">{t("aiHub.agentHubTableColumns.detailsShort")}</span>
          </Button>
        );
      },
    },
  ];

  return allColumns;
};
