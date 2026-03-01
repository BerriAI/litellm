"use client";

import React, { useState } from "react";
import { Button, Divider, Typography } from "antd";
import { PlusOutlined, CloseOutlined } from "@ant-design/icons";
import RoutingGroupsTable from "./RoutingGroupsTable";
import RoutingGroupBuilder from "./RoutingGroupBuilder";
import LiveTester from "./LiveTester";

interface RoutingGroupsViewProps {
  accessToken: string | null;
  userRole: string;
  userId: string;
}

export default function RoutingGroupsView({
  accessToken,
  userRole,
  userId,
}: RoutingGroupsViewProps) {
  const [createModalVisible, setCreateModalVisible] = useState(false);
  const [editTarget, setEditTarget] = useState<Record<string, unknown> | null>(null);
  const [refreshKey, setRefreshKey] = useState(0);
  const [selectedGroup, setSelectedGroup] = useState<Record<string, unknown> | null>(null);

  const handleEdit = (group: Record<string, unknown>) => {
    setEditTarget(group);
    setCreateModalVisible(true);
  };

  const handleTest = (group: Record<string, unknown>) => {
    setSelectedGroup(group);
  };

  const handleClose = () => {
    setCreateModalVisible(false);
    setEditTarget(null);
  };

  const handleSuccess = () => {
    setCreateModalVisible(false);
    setEditTarget(null);
    setRefreshKey((k) => k + 1);
  };

  return (
    <div className="w-full mx-4">
      <div className="flex justify-between items-center mb-6">
        <div>
          <Typography.Title level={4} style={{ margin: 0 }}>
            Routing Groups
          </Typography.Title>
          <Typography.Text type="secondary">
            Configure named routing pipelines with fallback strategies
          </Typography.Text>
        </div>
        <Button
          type="primary"
          icon={<PlusOutlined />}
          onClick={() => setCreateModalVisible(true)}
        >
          Create Routing Group
        </Button>
      </div>

      <RoutingGroupsTable
        accessToken={accessToken}
        refreshKey={refreshKey}
        onEdit={handleEdit}
        onTest={handleTest}
      />

      {selectedGroup && accessToken && (
        <>
          <Divider />
          <div className="flex items-center justify-between mb-4">
            <Typography.Text type="secondary" className="text-sm">
              Testing:{" "}
              <span className="font-mono font-semibold">
                {selectedGroup.routing_group_name as string}
              </span>
            </Typography.Text>
            <Button
              size="small"
              icon={<CloseOutlined />}
              onClick={() => setSelectedGroup(null)}
              style={{ color: "#64748b", borderColor: "#334155" }}
            >
              Close
            </Button>
          </div>
          <LiveTester
            accessToken={accessToken}
            routingGroupId={selectedGroup.routing_group_id as string}
            routingGroupName={selectedGroup.routing_group_name as string}
            routingStrategy={(selectedGroup.routing_strategy as string) ?? "weighted"}
          />
        </>
      )}

      {createModalVisible && (
        <RoutingGroupBuilder
          accessToken={accessToken}
          visible={createModalVisible}
          editTarget={editTarget}
          onClose={handleClose}
          onSuccess={handleSuccess}
        />
      )}
    </div>
  );
}
