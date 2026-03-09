"use client";

import { Alert, Typography } from "antd";
import { LockOutlined } from "@ant-design/icons";

interface PageAccessDeniedProps {
  pageName?: string;
  onNavigateToDefault?: () => void;
}

export default function PageAccessDenied({
  pageName = "this page",
  onNavigateToDefault,
}: PageAccessDeniedProps) {
  return (
    <div className="flex flex-1 items-center justify-center p-8">
      <div className="max-w-lg w-full">
        <Alert
          type="warning"
          showIcon
          icon={<LockOutlined />}
          message={
            <Typography.Title level={4} style={{ margin: 0 }}>
              Access to {pageName} is restricted
            </Typography.Title>
          }
          description={
            <div className="mt-2 space-y-2">
              <Typography.Paragraph style={{ marginBottom: 8 }}>
                Your proxy administrator has limited which pages are visible to internal users.
                {pageName !== "this page" && (
                  <> The &quot;{pageName}&quot; page is not included in your allowed pages.</>
                )}
              </Typography.Paragraph>
              <Typography.Paragraph style={{ marginBottom: 0 }}>
                If you need access to this page, please contact your proxy administrator.
                They can update the page visibility settings in Admin Settings to grant you access.
              </Typography.Paragraph>
            </div>
          }
        />
      </div>
    </div>
  );
}
