"use client";

import APIRef from "@/components/api_ref";
import { useState } from "react";

interface ProxySettings {
  PROXY_BASE_URL: string;
  PROXY_LOGOUT_URL: string;
}

const APIReferencePage = () => {
  const [proxySettings, setProxySettings] = useState<ProxySettings>({ PROXY_BASE_URL: "", PROXY_LOGOUT_URL: "" });

  return <APIRef proxySettings={proxySettings} />;
};

export default APIReferencePage;
