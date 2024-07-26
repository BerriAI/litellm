import React from "react";
import SearchBar from "@theme-original/SearchBar";

import useDocusaurusContext from "@docusaurus/useDocusaurusContext";
import { usePluginData } from "@docusaurus/useGlobalData";

export default function SearchBarWrapper(props) {
  const { siteConfig } = useDocusaurusContext();
  const { options } = usePluginData("docusaurus-plugin-pagefind-canary");

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
      import("@getcanary/web/components/canary-search-results-group"),
      import("@getcanary/web/components/canary-footer"),
      import("@getcanary/web/components/canary-callout-calendly"),
      import("@getcanary/web/components/canary-callout-discord"),
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
          <canary-provider-pagefind
            options={JSON.stringify({ ...options, path })}
          >
            <canary-modal>
              <canary-trigger-logo slot="trigger"></canary-trigger-logo>
              <canary-content slot="content">
                <canary-search slot="search">
                  <canary-search-input slot="input"></canary-search-input>
                  <canary-search-results-group
                    slot="results"
                    groups="SDK:*;Proxy:/docs/(simple_proxy|proxy/.*)"
                  ></canary-search-results-group>
                  <canary-callout-discord
                    slot="callout"
                    message="👋 Looking for help?"
                    url="https://discord.com/invite/wuPM9dRgDw"
                    keywords="discord,help,support,community"
                  ></canary-callout-discord>
                  <canary-callout-calendly
                    slot="callout"
                    message="🚅 Interested in enterprise features?"
                    keywords="sso,enterprise,security,audit"
                    url="https://calendly.com/d/4mp-gd3-k5k/litellm-1-1-onboarding-chat"
                  ></canary-callout-calendly>
                </canary-search>
                <canary-footer slot="footer"></canary-footer>
              </canary-content>
            </canary-modal>
          </canary-provider-pagefind>
        </canary-root>
      )}

      <SearchBar {...props} />
    </div>
  );
}
