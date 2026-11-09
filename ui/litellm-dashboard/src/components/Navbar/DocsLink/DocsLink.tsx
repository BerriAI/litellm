import { NAV_PRODUCT_LINK_CLASS } from "@/components/Navbar/navProductLinkClass";
import { ChevronDown } from "lucide-react";
import React from "react";
import { useTranslation } from "react-i18next";

export const DOCS_URL = "https://docs.litellm.ai/docs/";

const ChevronWidthSpacer: React.FC = () => (
  <ChevronDown className="pointer-events-none size-2.5 opacity-0" aria-hidden />
);

export const DocsLink: React.FC = () => {
  const { t } = useTranslation();
  return (
    <a href={DOCS_URL} target="_blank" rel="noopener noreferrer" className={NAV_PRODUCT_LINK_CLASS}>
      {t("shell.docsLink.label")}
      <ChevronWidthSpacer />
    </a>
  );
};

export default DocsLink;
