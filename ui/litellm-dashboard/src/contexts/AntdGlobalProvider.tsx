"use client";

import "@/lib/i18n";
import React, { useEffect, useRef } from "react";
import { notification, message, ConfigProvider } from "antd";
import type { Locale } from "antd/es/locale";
import zhCN from "antd/locale/zh_CN";
import { useTranslation } from "react-i18next";
import { setNotificationInstance } from "@/components/molecules/notifications_manager";
import { setMessageInstance } from "@/components/molecules/message_manager";

// antd falls back to its built-in English locale when undefined
const ANTD_LOCALES: Record<string, Locale> = { "zh-CN": zhCN };

export default function AntdGlobalProvider({ children }: { children: React.ReactNode }) {
  const [notificationApi, notificationContextHolder] = notification.useNotification();
  const [messageApi, messageContextHolder] = message.useMessage();
  const initialized = useRef(false);
  const { i18n } = useTranslation();

  useEffect(() => {
    if (!initialized.current) {
      setNotificationInstance(notificationApi);
      setMessageInstance(messageApi);
      initialized.current = true;
    }
  }, [notificationApi, messageApi]);

  return (
    <ConfigProvider locale={ANTD_LOCALES[i18n.language]}>
      {notificationContextHolder}
      {messageContextHolder}
      {children}
    </ConfigProvider>
  );
}
