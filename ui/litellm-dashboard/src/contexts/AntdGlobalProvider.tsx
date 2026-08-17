"use client";

import React, { useEffect, useRef } from "react";
import { ConfigProvider, notification, message } from "antd";
// eslint-disable-next-line no-restricted-imports -- Ant Design locales are required while legacy components remain.
import enUS from "antd/locale/en_US";
// eslint-disable-next-line no-restricted-imports -- Ant Design locales are required while legacy components remain.
import zhCN from "antd/locale/zh_CN";
import { StyleProvider } from "@ant-design/cssinjs";
import { setNotificationInstance } from "@/components/molecules/notifications_manager";
import { setMessageInstance } from "@/components/molecules/message_manager";
import { useTranslation } from "react-i18next";

export default function AntdGlobalProvider({ children }: { children: React.ReactNode }) {
  const { i18n } = useTranslation();
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
      <ConfigProvider locale={i18n.resolvedLanguage === "zh-CN" ? zhCN : enUS} theme={{ cssVar: true }}>
        {notificationContextHolder}
        {messageContextHolder}
        {children}
      </ConfigProvider>
    </StyleProvider>
  );
}
