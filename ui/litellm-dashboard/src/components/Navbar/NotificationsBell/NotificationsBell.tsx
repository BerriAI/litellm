"use client";

import { BellOutlined } from "@ant-design/icons";
import { Badge, Button, Popover, Typography } from "antd";
import React, { useEffect, useState } from "react";

const STORAGE_KEY = "litellmHideAgentPlatformBanner";
export const AGENT_PLATFORM_URL = "https://github.com/BerriAI/litellm-agent-platform";

export const NotificationsBell: React.FC = () => {
  const [hasUnread, setHasUnread] = useState(false);
  const [open, setOpen] = useState(false);

  useEffect(() => {
    try {
      setHasUnread(localStorage.getItem(STORAGE_KEY) !== "true");
    } catch {
      setHasUnread(true);
    }
  }, []);

  const markDismissed = () => {
    try {
      localStorage.setItem(STORAGE_KEY, "true");
    } catch {
      /* ignore */
    }
    setHasUnread(false);
    setOpen(false);
  };

  const content = (
    <div className="max-w-[280px]">
      <Typography.Title level={5} className="!mt-0 !mb-2">
        LiteLLM Agent Platform
      </Typography.Title>
      <Typography.Paragraph type="secondary" className="!mb-3 text-sm leading-snug">
        Open-source agent infra — sandboxes, durable sessions, and workers on AWS Fargate.
      </Typography.Paragraph>
      <div className="flex flex-wrap items-center gap-2">
        <Button type="primary" size="small" href={AGENT_PLATFORM_URL} target="_blank" rel="noopener noreferrer">
          GitHub
        </Button>
        {hasUnread ? (
          <Button type="link" size="small" className="!px-1" onClick={markDismissed}>
            Mark as read
          </Button>
        ) : null}
      </div>
    </div>
  );

  return (
    <Popover content={content} trigger="click" open={open} onOpenChange={setOpen} placement="bottomRight">
      <Button
        type="text"
        className="!flex !h-9 !w-9 items-center justify-center !rounded-md text-gray-600 transition-colors hover:!bg-gray-100 hover:!text-gray-900"
        aria-label="Notifications"
      >
        <Badge dot={hasUnread} color="#1677ff" size="small" offset={[8, 2]}>
          <BellOutlined className="text-base" aria-hidden />
        </Badge>
      </Button>
    </Popover>
  );
};
