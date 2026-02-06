"use client";

import React, { useEffect } from "react";
import { App } from "antd";
import { setNotificationInstance } from "@/components/molecules/notifications_manager";

// Inner component to use the hook
const AntdAppInit = () => {
  const { notification } = App.useApp();

  useEffect(() => {
    setNotificationInstance(notification);
  }, [notification]);

  return null;
};

export default function AntdGlobalProvider({ children }: { children: React.ReactNode }) {
  return (
    <App>
      <AntdAppInit />
      {children}
    </App>
  );
}
