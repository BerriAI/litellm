import React from "react";
import SearchBar from "@theme-original/SearchBar";

export default function SearchBarWrapper(props) {
  const [loaded, setLoaded] = React.useState(false);

  React.useEffect(() => {
    Promise.all([
      import("@getcanary/web/components/canary-root"),
      import("@getcanary/web/components/canary-provider-cloud"),
      import("@getcanary/web/components/canary-modal"),
      import("@getcanary/web/components/canary-trigger-logo"),
      import("@getcanary/web/components/canary-input"),
      import("@getcanary/web/components/canary-content"),
      import("@getcanary/web/components/canary-search"),
      import("@getcanary/web/components/canary-search-results"),
      import("@getcanary/web/components/canary-search-match-github-issue"),
      import("@getcanary/web/components/canary-search-match-github-discussion"),
      import("@getcanary/web/components/canary-filter-tabs-glob.js"),
      import("@getcanary/web/components/canary-filter-tags.js"),
      import("@getcanary/web/components/canary-footer.js"),
    ])
      .then(() => setLoaded(true))
      .catch(console.error);
  }, []);

  const PUBLIC_KEY = "cp1a506f13";

  const TAGS = "All,Proxy";

  const TABS = JSON.stringify([
    { name: "Docs", pattern: "**/docs.litellm.ai/**" },
    { name: "Github", pattern: "**/github.com/**" },
  ]);

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "row",
        alignItems: "center",
        gap: "6px",
      }}
    >
      {!loaded ? (
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
          <canary-provider-cloud project-key={PUBLIC_KEY}>
            <canary-modal>
              <canary-trigger-logo slot="trigger"></canary-trigger-logo>
              <canary-content slot="content">
                <canary-filter-tags
                  slot="head"
                  tags={TAGS}
                ></canary-filter-tags>
                <canary-input slot="input" autofocus></canary-input>
                <canary-search slot="mode">
                  <canary-filter-tabs-glob
                    slot="head"
                    tabs={TABS}
                  ></canary-filter-tabs-glob>
                  <canary-search-results slot="body"></canary-search-results>
                </canary-search>
                <canary-footer slot="footer"></canary-footer>
              </canary-content>
            </canary-modal>
          </canary-provider-cloud>
        </canary-root>
      )}

      <SearchBar {...props} />
    </div>
  );
}
