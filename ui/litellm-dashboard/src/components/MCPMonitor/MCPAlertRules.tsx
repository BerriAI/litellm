import {
  BellOutlined,
  DeleteOutlined,
  PlusOutlined,
} from "@ant-design/icons";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { Button, Input, Modal, Spin, Switch, message } from "antd";
import React, { useState } from "react";
import {
  getMCPAlertRules,
  createMCPAlertRule,
  deleteMCPAlertRule,
} from "@/components/networking";

interface MCPAlertRulesProps {
  accessToken?: string | null;
  mcpServerName?: string;
}

export function MCPAlertRules({
  accessToken = null,
  mcpServerName,
}: MCPAlertRulesProps) {
  const queryClient = useQueryClient();
  const [createModalOpen, setCreateModalOpen] = useState(false);
  const [formData, setFormData] = useState({
    alert_name: "",
    tool_name_pattern: "",
    webhook_url: "",
    description: "",
    mcp_server_name: mcpServerName || "",
  });

  const { data, isLoading } = useQuery({
    queryKey: ["mcp-alert-rules", mcpServerName],
    queryFn: () => getMCPAlertRules(accessToken!, mcpServerName),
    enabled: !!accessToken,
  });

  const createMutation = useMutation({
    mutationFn: (newRule: any) => createMCPAlertRule(accessToken!, newRule),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["mcp-alert-rules"] });
      setCreateModalOpen(false);
      setFormData({
        alert_name: "",
        tool_name_pattern: "",
        webhook_url: "",
        description: "",
        mcp_server_name: mcpServerName || "",
      });
      message.success("Alert rule created");
    },
    onError: (err: any) => {
      message.error(`Failed to create alert rule: ${err.message}`);
    },
  });

  const deleteMutation = useMutation({
    mutationFn: (ruleId: string) => deleteMCPAlertRule(accessToken!, ruleId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["mcp-alert-rules"] });
      message.success("Alert rule deleted");
    },
    onError: (err: any) => {
      message.error(`Failed to delete: ${err.message}`);
    },
  });

  const rules = data?.rules ?? [];

  const handleCreate = () => {
    if (!formData.alert_name || !formData.tool_name_pattern || !formData.webhook_url) {
      message.warning("Please fill in all required fields");
      return;
    }
    createMutation.mutate({
      alert_name: formData.alert_name,
      tool_name_pattern: formData.tool_name_pattern,
      webhook_url: formData.webhook_url,
      description: formData.description || undefined,
      mcp_server_name: formData.mcp_server_name || undefined,
      enabled: true,
    });
  };

  return (
    <div className="bg-white border border-gray-200 rounded-lg">
      <div className="p-4 border-b border-gray-200">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-base font-semibold text-gray-900 flex items-center gap-2">
              <BellOutlined className="text-amber-500" />
              Alert Rules
            </h3>
            <p className="text-xs text-gray-500 mt-0.5">
              Get notified when specific MCP tools are invoked (e.g., delete operations)
            </p>
          </div>
          <Button
            type="primary"
            icon={<PlusOutlined />}
            size="small"
            onClick={() => setCreateModalOpen(true)}
          >
            New Rule
          </Button>
        </div>
      </div>

      {isLoading ? (
        <div className="flex items-center justify-center py-8">
          <Spin />
        </div>
      ) : rules.length === 0 ? (
        <div className="py-8 text-center text-sm text-gray-500">
          No alert rules configured. Create one to get notified about specific tool operations.
        </div>
      ) : (
        <div className="divide-y divide-gray-100">
          {rules.map((rule: any) => (
            <div
              key={rule.id}
              className="px-4 py-3 flex items-center justify-between"
            >
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2 mb-1">
                  <span className="font-medium text-gray-900 text-sm">
                    {rule.alert_name}
                  </span>
                  <span
                    className={`inline-flex items-center px-1.5 py-0.5 text-[10px] font-medium rounded ${
                      rule.enabled
                        ? "bg-green-50 text-green-700 border border-green-200"
                        : "bg-gray-100 text-gray-500 border border-gray-200"
                    }`}
                  >
                    {rule.enabled ? "Active" : "Disabled"}
                  </span>
                </div>
                <div className="flex items-center gap-3 text-xs text-gray-500">
                  <span>
                    Pattern:{" "}
                    <code className="bg-gray-100 px-1 rounded">
                      {rule.tool_name_pattern}
                    </code>
                  </span>
                  {rule.mcp_server_name && (
                    <span>Server: {rule.mcp_server_name}</span>
                  )}
                  {rule.description && (
                    <span className="text-gray-400">{rule.description}</span>
                  )}
                </div>
              </div>
              <Button
                type="text"
                danger
                icon={<DeleteOutlined />}
                size="small"
                loading={deleteMutation.isPending}
                onClick={() => {
                  Modal.confirm({
                    title: "Delete Alert Rule",
                    content: `Are you sure you want to delete "${rule.alert_name}"?`,
                    onOk: () => deleteMutation.mutate(rule.id),
                  });
                }}
              />
            </div>
          ))}
        </div>
      )}

      <Modal
        title="Create MCP Alert Rule"
        open={createModalOpen}
        onCancel={() => setCreateModalOpen(false)}
        onOk={handleCreate}
        confirmLoading={createMutation.isPending}
        okText="Create"
      >
        <div className="space-y-4 mt-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Alert Name *
            </label>
            <Input
              placeholder="e.g., Delete Operation Alert"
              value={formData.alert_name}
              onChange={(e) =>
                setFormData({ ...formData, alert_name: e.target.value })
              }
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Tool Name Pattern *
            </label>
            <Input
              placeholder="e.g., *delete*, *remove*, drop_*"
              value={formData.tool_name_pattern}
              onChange={(e) =>
                setFormData({ ...formData, tool_name_pattern: e.target.value })
              }
            />
            <p className="text-xs text-gray-400 mt-1">
              Glob-style pattern. Use * as wildcard. Examples: *delete*, *remove*, drop_*
            </p>
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Webhook URL *
            </label>
            <Input
              placeholder="https://hooks.slack.com/services/..."
              value={formData.webhook_url}
              onChange={(e) =>
                setFormData({ ...formData, webhook_url: e.target.value })
              }
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              MCP Server (optional)
            </label>
            <Input
              placeholder="Leave empty to match all servers"
              value={formData.mcp_server_name}
              onChange={(e) =>
                setFormData({ ...formData, mcp_server_name: e.target.value })
              }
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Description (optional)
            </label>
            <Input.TextArea
              placeholder="Describe what this alert monitors"
              value={formData.description}
              onChange={(e) =>
                setFormData({ ...formData, description: e.target.value })
              }
              rows={2}
            />
          </div>
        </div>
      </Modal>
    </div>
  );
}
