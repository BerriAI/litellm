"use client";

import React, { useEffect, useRef } from "react";
import { notification } from "antd";
import { setNotificationInstance } from "@/components/molecules/notifications_manager";

export default function AntdGlobalProvider({ children }: { children: React.ReactNode }) {
  const [api, contextHolder] = notification.useNotification();
  const initialized = useRef(false);

  useEffect(() => {
    if (!initialized.current) {
      setNotificationInstance(api);
      initialized.current = true;
    }
  }, [api]);

  return (
    <>
      {contextHolder}
      {children}
    </>
  );
}
