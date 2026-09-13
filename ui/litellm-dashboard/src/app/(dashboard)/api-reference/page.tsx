"use client";

import APIReferenceView from "./_components/APIReferenceView";
import { DeprecationBanner } from "@/components/DeprecationBanner";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import useProxySettings from "@/app/(dashboard)/hooks/proxySettings/useProxySettings";

const APIReferencePage = () => {
  const { accessToken } = useAuthorized();
  const proxySettings = useProxySettings(accessToken);

  return (
    <>
      <DeprecationBanner featureName="The API Reference tab" />
      <APIReferenceView proxySettings={proxySettings} />
    </>
  );
};

export default APIReferencePage;
