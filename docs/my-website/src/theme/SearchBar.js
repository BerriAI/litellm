import React from "react";
import SearchBar from "@theme-original/SearchBar";

import useDocusaurusContext from "@docusaurus/useDocusaurusContext";
import { usePluginData } from "@docusaurus/useGlobalData";

export default function SearchBarWrapper(props) {
  const { siteConfig } = useDocusaurusContext();
  const { options } = usePluginData("docusaurus-theme-search-pagefind");

  const [path, setPath] = React.useState("");
  const [loaded, setLoaded] = React.useState(false);

  React.useEffect(() => {
    setPath(`${siteConfig.baseUrl}pagefind/pagefind.js`);
  }, [siteConfig]);

  React.useEffect(() => {
    Promise.all([
      import("@getcanary/web/components/canary-root"),
      import("@getcanary/web/components/canary-provider-pagefind"),
      import("@getcanary/web/components/canary-modal"),
      import("@getcanary/web/components/canary-trigger-logo"),
      import("@getcanary/web/components/canary-content"),
      import("@getcanary/web/components/canary-search"),
      import("@getcanary/web/components/canary-search-input"),
      import("@getcanary/web/components/canary-search-results-tabs"),
    ])
      .then(() => setLoaded(true))
      .catch(console.error);
  }, []);

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "row",
        alignItems: "center",
        gap: "6px",
      }}
    >
      {!loaded || !path ? (
        <button
          style={{
            fontSize: "2rem",
            backgroundColor: "transparent",
            border: "none",
            outline: "none",
            padding: "0",
            marginRight: "6px",
          }}
        >
          🐤
        </button>
      ) : (
        <canary-root framework="docusaurus">
          <canary-provider-pagefind options={JSON.stringify(options)}>
            <canary-modal>
              <canary-trigger-logo slot="trigger"></canary-trigger-logo>
              <canary-content slot="content">
                <canary-search slot="mode">
                  <canary-search-input slot="input"></canary-search-input>
                  <canary-search-results-tabs
                    slot="body"
                    tabs={JSON.stringify(options.tabs)}
                    group
                  ></canary-search-results-tabs>
                </canary-search>
              </canary-content>
            </canary-modal>
          </canary-provider-pagefind>
        </canary-root>
      )}

      <SearchBar {...props} />
    </div>
  );
}