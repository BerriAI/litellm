import { NAV_PRODUCT_LINK_CLASS } from "@/components/Navbar/navProductLinkClass";
import { ChevronDown } from "lucide-react";
import React from "react";

export const DOCS_URL = "https://docs.litellm.ai/docs/";

export const DocsLink: React.FC = () => (
  <a href={DOCS_URL} target="_blank" rel="noopener noreferrer" className={NAV_PRODUCT_LINK_CLASS}>
    Docs
    {/* Docs is a single outbound link; the hidden chevron keeps its box identical to the Blog dropdown trigger. */}
    <ChevronDown className="pointer-events-none size-2.5 opacity-0" aria-hidden />
  </a>
);

export default DocsLink;
