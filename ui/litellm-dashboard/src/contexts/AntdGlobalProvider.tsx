"use client";

import React, { useEffect, useRef } from "react";
import { ConfigProvider, notification, message } from "antd";
import { StyleProvider } from "@ant-design/cssinjs";
import { setNotificationInstance } from "@/components/molecules/notifications_manager";
import { setMessageInstance } from "@/components/molecules/message_manager";
import { useDashboardLanguage } from "@/i18n/I18nProvider";
// eslint-disable-next-line no-restricted-imports -- locale data is required while the existing antd UI remains in service
import enUS from "antd/locale/en_US";
// eslint-disable-next-line no-restricted-imports -- locale data is required while the existing antd UI remains in service
import ruRU from "antd/locale/ru_RU";

export default function AntdGlobalProvider({ children }: { children: React.ReactNode }) {
  const { language } = useDashboardLanguage();
  const [notificationApi, notificationContextHolder] = notification.useNotification();
  const [messageApi, messageContextHolder] = message.useMessage();
  const initialized = useRef(false);

  useEffect(() => {
    if (!initialized.current) {
      setNotificationInstance(notificationApi);
      setMessageInstance(messageApi);
      initialized.current = true;
    }
  }, [notificationApi, messageApi]);

  return (
    <StyleProvider layer>
      <ConfigProvider locale={language === "ru" ? ruRU : enUS} theme={{ cssVar: true }}>
        {notificationContextHolder}
        {messageContextHolder}
        {children}
      </ConfigProvider>
    </StyleProvider>
  );
}
