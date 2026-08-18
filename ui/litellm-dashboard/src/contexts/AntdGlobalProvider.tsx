"use client";

import React from "react";
import { ConfigProvider } from "antd";
import { StyleProvider } from "@ant-design/cssinjs";

export default function AntdGlobalProvider({ children }: { children: React.ReactNode }) {
  return (
    <StyleProvider layer>
      <ConfigProvider theme={{ cssVar: true }}>{children}</ConfigProvider>
    </StyleProvider>
  );
}
