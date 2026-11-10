"use client";

import { useTranslation } from "react-i18next";
import APIReferenceView from "./_components/APIReferenceView";
import { DeprecationBanner } from "@/components/DeprecationBanner";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import useProxySettings from "@/app/(dashboard)/hooks/proxySettings/useProxySettings";

const APIReferencePage = () => {
  const { t } = useTranslation();
  const { accessToken } = useAuthorized();
  const proxySettings = useProxySettings(accessToken);

  return (
    <>
      <DeprecationBanner featureName={t("apiReference.page.featureName")} />
      <APIReferenceView proxySettings={proxySettings} />
    </>
  );
};

export default APIReferencePage;
