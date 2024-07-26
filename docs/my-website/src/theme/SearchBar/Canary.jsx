import React from "react";

export default function Canary({ path }) {
  const [loaded, setLoaded] = React.useState(false);

  React.useEffect(() => {
    Promise.all([
      import("@getcanary/web/components/canary-root"),
      import("@getcanary/web/components/canary-provider-pagefind"),
      import("@getcanary/web/components/canary-modal"),
      import("@getcanary/web/components/canary-trigger-searchbar"),
      import("@getcanary/web/components/canary-content"),
      import("@getcanary/web/components/canary-search"),
      import("@getcanary/web/components/canary-search-input"),
      import("@getcanary/web/components/canary-search-results-group"),
      import("@getcanary/web/components/canary-callout-calendly"),
      import("@getcanary/web/components/canary-callout-discord"),
    ])
      .then(() => setLoaded(true))
      .catch((e) =>
        console.error("Maybe you forgot to install '@getcanary/web'?", e),
      );
  }, []);

  if (!loaded) {
    return null;
  }

  return (
    <canary-root framework="docusaurus">
      <canary-provider-pagefind path={path}>
        <canary-modal>
          <canary-trigger-searchbar slot="trigger"></canary-trigger-searchbar>
          <canary-content slot="content">
            <canary-search slot="search">
              <canary-search-input slot="input"></canary-search-input>
              <canary-search-results-group
                slot="results"
                groups="SDK:*;Proxy:proxy"
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
          </canary-content>
        </canary-modal>
      </canary-provider-pagefind>
    </canary-root>
  );
}
