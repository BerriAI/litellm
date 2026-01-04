"use client";

import { Card, Descriptions, Skeleton, Space, Typography } from "antd";
import { Shield } from "lucide-react";

const { Title, Text } = Typography;
export default function SSOSettingsLoadingSkeleton() {
  const descriptionsConfig = {
    column: {
      xxl: 1,
      xl: 1,
      lg: 1,
      md: 1,
      sm: 1,
      xs: 1,
    },
  };

  return (
    <Card>
      <Space direction="vertical" size="large" className="w-full">
        {/* Header Section */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Shield className="w-6 h-6 text-gray-400" />
            <div>
              <Title level={3}>SSO Configuration</Title>
              <Text type="secondary">Manage Single Sign-On authentication settings</Text>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <Skeleton.Button active size="default" style={{ width: 170, height: 32 }} />
            <Skeleton.Button active size="default" style={{ width: 190, height: 32 }} />
          </div>
        </div>

        {/* Descriptions Table Skeleton */}
        <Descriptions bordered {...descriptionsConfig}>
          {/* Provider Row */}
          <Descriptions.Item label={<Skeleton.Node active style={{ width: 80, height: 16 }} />}>
            <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
              <Skeleton.Node active style={{ width: 100, height: 16 }} />
            </div>
          </Descriptions.Item>

          <Descriptions.Item label={<Skeleton.Node active style={{ width: 80, height: 16 }} />}>
            <Skeleton.Node active style={{ width: 200, height: 16 }} />
          </Descriptions.Item>

          <Descriptions.Item label={<Skeleton.Node active style={{ width: 80, height: 16 }} />}>
            <Skeleton.Node active style={{ width: 250, height: 16 }} />
          </Descriptions.Item>

          <Descriptions.Item label={<Skeleton.Node active style={{ width: 80, height: 16 }} />}>
            <Skeleton.Node active style={{ width: 180, height: 16 }} />
          </Descriptions.Item>

          <Descriptions.Item label={<Skeleton.Node active style={{ width: 80, height: 16 }} />}>
            <Skeleton.Node active style={{ width: 220, height: 16 }} />
          </Descriptions.Item>
        </Descriptions>
      </Space>
    </Card>
  );
}
