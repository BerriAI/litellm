"use client";

import { getAvailablePages } from "@/components/page_utils";
import { Button, Checkbox, Collapse, Space, Tag, Typography } from "antd";
import { useMemo, useState } from "react";

interface PageVisibilitySettingsProps {
  enabledPagesInternalUsers: string[] | null | undefined;
  enabledPagesPropertyDescription?: string;
  isUpdating: boolean;
  onUpdate: (settings: { enabled_ui_pages_internal_users: string[] | null }) => void;
}

export default function PageVisibilitySettings({
  enabledPagesInternalUsers,
  enabledPagesPropertyDescription,
  isUpdating,
  onUpdate,
}: PageVisibilitySettingsProps) {
  // Check if page visibility is set (null/undefined means "not set" = all pages visible)
  const isPageVisibilitySet = enabledPagesInternalUsers !== null && enabledPagesInternalUsers !== undefined;

  // Get available pages from leftnav configuration
  const availablePages = useMemo(() => getAvailablePages(), []);

  // Group pages by their group for better UI
  const pagesByGroup = useMemo(() => {
    const grouped: Record<string, typeof availablePages> = {};
    availablePages.forEach((page) => {
      if (!grouped[page.group]) {
        grouped[page.group] = [];
      }
      grouped[page.group].push(page);
    });
    return grouped;
  }, [availablePages]);

  // Local state for page selection
  const [selectedPages, setSelectedPages] = useState<string[]>(enabledPagesInternalUsers || []);

  // Update local state when data changes
  useMemo(() => {
    if (enabledPagesInternalUsers) {
      setSelectedPages(enabledPagesInternalUsers);
    } else {
      setSelectedPages([]);
    }
  }, [enabledPagesInternalUsers]);

  const handleSavePageVisibility = () => {
    onUpdate({ enabled_ui_pages_internal_users: selectedPages.length > 0 ? selectedPages : null });
  };

  const handleResetToDefault = () => {
    setSelectedPages([]);
    onUpdate({ enabled_ui_pages_internal_users: null });
  };

  return (
    <Space direction="vertical" size="middle" style={{ width: "100%" }}>
      <Space direction="vertical" size={4}>
        <Space align="center">
          <Typography.Text strong>Internal User Page Visibility</Typography.Text>
          {!isPageVisibilitySet && (
            <Tag color="default" style={{ marginLeft: "8px" }}>
              Not set (all pages visible)
            </Tag>
          )}
          {isPageVisibilitySet && (
            <Tag color="blue" style={{ marginLeft: "8px" }}>
              {selectedPages.length} page{selectedPages.length !== 1 ? "s" : ""} selected
            </Tag>
          )}
        </Space>
        {enabledPagesPropertyDescription && (
          <Typography.Text type="secondary">{enabledPagesPropertyDescription}</Typography.Text>
        )}
        <Typography.Text type="secondary" style={{ fontSize: "12px", fontStyle: "italic" }}>
          By default, all pages are visible to internal users. Select specific pages to restrict visibility.
        </Typography.Text>
        <Typography.Text type="secondary" style={{ fontSize: "12px", color: "#8b5cf6" }}>
          Note: Only pages accessible to internal user roles are shown here. Admin-only pages are excluded as they
          cannot be made visible to internal users regardless of this setting.
        </Typography.Text>
      </Space>

      <Collapse
        items={[
          {
            key: "page-visibility",
            label: "Configure Page Visibility",
            children: (
              <Space direction="vertical" size="middle" style={{ width: "100%" }}>
                <Checkbox.Group value={selectedPages} onChange={setSelectedPages} style={{ width: "100%" }}>
                  <Space direction="vertical" size="middle" style={{ width: "100%" }}>
                    {Object.entries(pagesByGroup).map(([groupName, pages]) => (
                      <div key={groupName}>
                        <Typography.Text
                          strong
                          style={{
                            fontSize: "11px",
                            color: "#6b7280",
                            letterSpacing: "0.05em",
                            display: "block",
                            marginBottom: "8px",
                          }}
                        >
                          {groupName}
                        </Typography.Text>
                        <Space direction="vertical" size="small" style={{ marginLeft: "16px", width: "100%" }}>
                          {pages.map((page) => (
                            <div key={page.page} style={{ marginBottom: "4px" }}>
                              <Checkbox value={page.page}>
                                <Space direction="vertical" size={0}>
                                  <Typography.Text>{page.label}</Typography.Text>
                                  <Typography.Text type="secondary" style={{ fontSize: "12px" }}>
                                    {page.description}
                                  </Typography.Text>
                                </Space>
                              </Checkbox>
                            </div>
                          ))}
                        </Space>
                      </div>
                    ))}
                  </Space>
                </Checkbox.Group>

                <Space>
                  <Button type="primary" onClick={handleSavePageVisibility} loading={isUpdating} disabled={isUpdating}>
                    Save Page Visibility Settings
                  </Button>
                  {isPageVisibilitySet && (
                    <Button onClick={handleResetToDefault} loading={isUpdating} disabled={isUpdating}>
                      Reset to Default (All Pages)
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
