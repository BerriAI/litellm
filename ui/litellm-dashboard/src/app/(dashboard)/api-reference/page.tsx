"use client";

import APIReferenceView from "@/app/(dashboard)/api-reference/APIReferenceView";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import useProxySettings from "@/app/(dashboard)/hooks/proxySettings/useProxySettings";

const APIReferencePage = () => {
  const { accessToken } = useAuthorized();
  const proxySettings = useProxySettings(accessToken);

  return <APIReferenceView proxySettings={proxySettings} />;
};

export default APIReferencePage;
