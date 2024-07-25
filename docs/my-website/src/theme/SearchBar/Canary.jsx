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
      import("@getcanary/web/components/canary-search-results"),
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
              <canary-search-results slot="results"></canary-search-results>
            </canary-search>
          </canary-content>
        </canary-modal>
      </canary-provider-pagefind>
    </canary-root>
  );
}
