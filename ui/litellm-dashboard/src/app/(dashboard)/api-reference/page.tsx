"use client";

import APIReferenceView from "@/app/(dashboard)/api-reference/APIReferenceView";
import useProxySettings from "@/app/(dashboard)/hooks/proxySettings/useProxySettings";

const APIReferencePage = () => {
  const proxySettings = useProxySettings();

  return <APIReferenceView proxySettings={proxySettings} />;
};

export default APIReferencePage;
