"use client";

import APIReferenceView from "@/app/(dashboard)/api-reference/APIReferenceView";
import { useState } from "react";

interface ProxySettings {
  PROXY_BASE_URL: string;
  PROXY_LOGOUT_URL: string;
}

const APIReferencePage = () => {
  const [proxySettings, setProxySettings] = useState<ProxySettings>({ PROXY_BASE_URL: "", PROXY_LOGOUT_URL: "" });

  return <APIReferenceView proxySettings={proxySettings} />;
};

export default APIReferencePage;
