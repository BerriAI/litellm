"use client";

import { getConfigurableNonAdminUsageViews } from "@/components/UsagePage/components/UsageViewSelect/UsageViewSelect";
import { Button, Checkbox, Collapse, Space, Tag, Typography } from "antd";
import { useMemo, useState } from "react";

interface UsageViewVisibilitySettingsProps {
  enabledViewsInternalUsers: string[] | null | undefined;
  enabledViewsPropertyDescription?: string;
  isUpdating: boolean;
  onUpdate: (settings: { enabled_usage_views_internal_users: string[] | null }) => void;
}

export default function UsageViewVisibilitySettings({
  enabledViewsInternalUsers,
  enabledViewsPropertyDescription,
  isUpdating,
  onUpdate,
}: UsageViewVisibilitySettingsProps) {
  const isVisibilitySet = enabledViewsInternalUsers !== null && enabledViewsInternalUsers !== undefined;

  const availableViews = useMemo(() => getConfigurableNonAdminUsageViews(), []);

  const [selectedViews, setSelectedViews] = useState<string[]>(enabledViewsInternalUsers ?? []);
  const [syncedFrom, setSyncedFrom] = useState(enabledViewsInternalUsers);
  if (syncedFrom !== enabledViewsInternalUsers) {
    setSyncedFrom(enabledViewsInternalUsers);
    setSelectedViews(enabledViewsInternalUsers ?? []);
  }

  const handleSave = () => {
    onUpdate({ enabled_usage_views_internal_users: selectedViews.length > 0 ? selectedViews : null });
  };

  const handleResetToDefault = () => {
    setSelectedViews([]);
    onUpdate({ enabled_usage_views_internal_users: null });
  };

  return (
    <Space direction="vertical" size="middle" style={{ width: "100%" }}>
      <Space direction="vertical" size={4}>
        <Space align="center">
          <Typography.Text strong>Internal User Usage View Visibility</Typography.Text>
          {!isVisibilitySet && (
            <Tag color="default" style={{ marginLeft: "8px" }}>
              Not set (all views visible)
            </Tag>
          )}
          {isVisibilitySet && (
            <Tag color="blue" style={{ marginLeft: "8px" }}>
              {selectedViews.length} view{selectedViews.length !== 1 ? "s" : ""} selected
            </Tag>
          )}
        </Space>
        {enabledViewsPropertyDescription && (
          <Typography.Text type="secondary">{enabledViewsPropertyDescription}</Typography.Text>
        )}
        <Typography.Text type="secondary" style={{ fontSize: "12px", fontStyle: "italic" }}>
          By default, all non-admin views are visible to internal users in the Usage page dropdown. Select specific
          views to restrict which options they can choose.
        </Typography.Text>
        <Typography.Text type="secondary" style={{ fontSize: "12px", color: "#8b5cf6" }}>
          Note: Admins always see every usage view regardless of this setting.
        </Typography.Text>
      </Space>

      <Collapse
        items={[
          {
            key: "usage-view-visibility",
            label: "Configure Usage View Visibility",
            children: (
              <Space direction="vertical" size="middle" style={{ width: "100%" }}>
                <Checkbox.Group value={selectedViews} onChange={setSelectedViews} style={{ width: "100%" }}>
                  <Space direction="vertical" size="small" style={{ width: "100%" }}>
                    {availableViews.map((view) => (
                      <div key={view.value} style={{ marginBottom: "4px" }}>
                        <Checkbox value={view.value}>
                          <Space direction="vertical" size={0}>
                            <Typography.Text>{view.label}</Typography.Text>
                            <Typography.Text type="secondary" style={{ fontSize: "12px" }}>
                              {view.description}
                            </Typography.Text>
                          </Space>
                        </Checkbox>
                      </div>
                    ))}
                  </Space>
                </Checkbox.Group>

                <Space>
                  <Button type="primary" onClick={handleSave} loading={isUpdating} disabled={isUpdating}>
                    Save Usage View Visibility Settings
                  </Button>
                  {isVisibilitySet && (
                    <Button onClick={handleResetToDefault} loading={isUpdating} disabled={isUpdating}>
                      Reset to Default (All Views)
                    </Button>
                  )}
                </Space>
              </Space>
            ),
          },
        ]}
      />
    </Space>
  );
}
