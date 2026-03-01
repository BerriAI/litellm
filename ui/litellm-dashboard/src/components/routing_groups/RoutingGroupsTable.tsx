"use client";

import React, { useEffect, useState } from "react";
import { Button, Popconfirm, Table, Tag, Typography } from "antd";
import { DeleteOutlined, EditOutlined, ThunderboltOutlined } from "@ant-design/icons";
import { routingGroupDeleteCall, routingGroupListCall } from "@/components/networking";

interface RoutingGroupsTableProps {
  accessToken: string | null;
  refreshKey: number;
  onEdit: (group: Record<string, unknown>) => void;
  onTest?: (group: Record<string, unknown>) => void;
}

const strategyColors: Record<string, string> = {
  "priority-failover": "blue",
  weighted: "green",
  "cost-based-routing": "gold",
  "latency-based-routing": "cyan",
  "least-busy": "purple",
  "usage-based-routing-v2": "orange",
  "simple-shuffle": "geekblue",
};

const strategyLabels: Record<string, string> = {
  "priority-failover": "Priority Failover",
  weighted: "Weighted",
  "cost-based-routing": "Cost-Based",
  "latency-based-routing": "Latency-Based",
  "least-busy": "Least Busy",
  "usage-based-routing-v2": "Usage-Based",
  "simple-shuffle": "Round Robin",
};

export default function RoutingGroupsTable({
  accessToken,
  refreshKey,
  onEdit,
  onTest,
}: RoutingGroupsTableProps) {
  const [groups, setGroups] = useState<Record<string, unknown>[]>([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!accessToken) return;
    setLoading(true);
    routingGroupListCall(accessToken, 1, 50)
      .then((resp) => {
        setGroups(resp.routing_groups || []);
      })
      .catch(console.error)
      .finally(() => setLoading(false));
  }, [accessToken, refreshKey]);

  const handleDelete = async (groupId: string) => {
    if (!accessToken) return;
    try {
      await routingGroupDeleteCall(accessToken, groupId);
      setGroups((prev) =>
        prev.filter((g) => (g.routing_group_id as string) !== groupId)
      );
    } catch (err) {
      console.error("Failed to delete routing group:", err);
    }
  };

  const columns = [
    {
      title: "Name",
      dataIndex: "routing_group_name",
      key: "routing_group_name",
      render: (name: string) => (
        <Typography.Text strong>{name}</Typography.Text>
      ),
    },
    {
      title: "Strategy",
      dataIndex: "routing_strategy",
      key: "routing_strategy",
      render: (strategy: string) => {
        const label = strategyLabels[strategy] ?? strategy;
        const color = strategyColors[strategy] ?? "default";
        return <Tag color={color}>{label}</Tag>;
      },
    },
    {
      title: "Deployments",
      dataIndex: "deployments",
      key: "deployments",
      render: (deployments: unknown) => {
        const count = Array.isArray(deployments) ? deployments.length : 0;
        return <Typography.Text>{count}</Typography.Text>;
      },
    },
    {
      title: "Status",
      dataIndex: "status",
      key: "status",
      render: (status: string) => {
        const isActive = !status || status === "active";
        return (
          <Tag color={isActive ? "success" : "default"}>
            {isActive ? "Active" : "Inactive"}
          </Tag>
        );
      },
    },
    {
      title: "Created",
      dataIndex: "created_at",
      key: "created_at",
      render: (created: string) => {
        if (!created) return "-";
        return new Date(created).toLocaleDateString();
      },
    },
    {
      title: "Actions",
      key: "actions",
      render: (_: unknown, record: Record<string, unknown>) => {
        const groupId = record.routing_group_id as string;
        return (
          <div style={{ display: "flex", gap: 8 }}>
            {onTest && (
              <Button
                size="small"
                icon={<ThunderboltOutlined />}
                onClick={() => onTest(record)}
                style={{ color: "#14b8a6", borderColor: "#14b8a6" }}
              >
                Test
              </Button>
            )}
            <Button
              size="small"
              icon={<EditOutlined />}
              onClick={() => onEdit(record)}
            >
              Edit
            </Button>
            <Popconfirm
              title="Delete routing group?"
              description="This action cannot be undone."
              onConfirm={() => handleDelete(groupId)}
              okText="Delete"
              okButtonProps={{ danger: true }}
              cancelText="Cancel"
            >
              <Button size="small" danger icon={<DeleteOutlined />}>
                Delete
              </Button>
            </Popconfirm>
          </div>
        );
      },
    },
  ];

  return (
    <Table
      dataSource={groups}
      columns={columns}
      loading={loading}
      rowKey={(record) => (record.routing_group_id as string) ?? Math.random().toString()}
      pagination={{ pageSize: 20 }}
    />
  );
}
