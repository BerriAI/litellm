"use client";

import React, { useEffect, useRef } from "react";
import { notification } from "antd";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { setNotificationInstance } from "@/components/molecules/notifications_manager";

const queryClient = new QueryClient();

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
    <QueryClientProvider client={queryClient}>
      {contextHolder}
      {children}
    </QueryClientProvider>
  );
}
