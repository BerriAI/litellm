import React from "react";

import useDocusaurusContext from "@docusaurus/useDocusaurusContext";
import { usePluginData } from "@docusaurus/useGlobalData";

import Canary from "./Canary";

export default function Index() {
  const { siteConfig } = useDocusaurusContext();
  const [path, setPath] = React.useState("");

  React.useEffect(() => {
    setPath(`${siteConfig.baseUrl}pagefind/pagefind.js`);
  }, [siteConfig]);

  const { options } = usePluginData("docusaurus-theme-search-pagefind");
  const { styles, ...rest } = options;

  React.useEffect(() => {
    if (options.styles) {
      Object.entries(options.styles).forEach(([key, value]) => {
        document.body.style.setProperty(key, value);
      });
    }
  }, [options]);

  if (!path) {
    return null;
  }

  return <Canary options={{ ...rest, path: path }} />;
}
