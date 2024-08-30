import React from "react";

export default function Canary({ options }) {
  const [loaded, setLoaded] = React.useState(false);

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
      .catch((e) =>
        console.error("Maybe you forgot to install '@getcanary/web'?", e),
      );
  }, []);

  if (!loaded) {
    return null;
  }

  return (
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
  );
}
