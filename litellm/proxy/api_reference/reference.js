import { groupSchema } from "./route-categories.mjs";

(() => {
  "use strict";

  const config = JSON.parse(
    document.getElementById("reference-config").textContent,
  );
  for (const link of document.querySelectorAll("[data-link]")) {
    link.href = config[link.dataset.link];
  }

  const apiBase = new URL(
    config.apiBase || "/",
    window.location.origin,
  ).href.replace(/\/$/, "");
  const modelUrl = `${apiBase}/v1/models`;
  document.getElementById("request-url").textContent = modelUrl;
  document.getElementById("api-version").textContent =
    `LITELLM / v${config.version}`;
  document.getElementById("lazy-notice").hidden = !config.lazyTags.length;

  const copyButton = document.getElementById("copy-request");
  copyButton.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(
        `curl ${modelUrl} \\\n  -H "Authorization: Bearer $LITELLM_API_KEY"`,
      );
      copyButton.textContent = "Copied ✓";
    } catch {
      copyButton.textContent = "Select code to copy";
    }
    setTimeout(() => {
      copyButton.textContent = "Copy ⧉";
    }, 2400);
  });

  const status = document.getElementById("reference-status");
  async function loadReference() {
    try {
      const response = await fetch(config.schemaUrl, {
        credentials: "same-origin",
      });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const schema = await response.json();
      if (!schema.openapi && !schema.swagger)
        throw new Error("Invalid OpenAPI document");
      if (!window.Scalar)
        throw new Error("API reference assets could not be loaded");

      const groups = groupSchema(schema);
      const buttons = [...document.querySelectorAll("[data-category]")];
      const currentGroup = () =>
        groups.find(
          (group) =>
            group.id ===
            new URL(window.location.href).searchParams.get("category"),
        ) || groups[0];
      const referenceContent = document.getElementById("reference-content");
      const baseOptions = {
        slug: "litellm-reference",
        baseServerURL: window.location.origin,
        ...(schema.servers?.length ? {} : { servers: [{ url: apiBase }] }),
        theme: "none",
        layout: "modern",
        withDefaultFonts: false,
        darkMode: false,
        showSidebar: true,
        defaultOpenAllTags: false,
        defaultOpenFirstTag: false,
        hideModels: true,
        showDeveloperTools: "never",
        hideClientButton: true,
        persistAuth: false,
        telemetry: false,
        agent: { disabled: true },
        proxyUrl: "",
        defaultHttpClient: { targetKey: "shell", clientKey: "curl" },
        setPageTitle: ({ title }) => `${title} · ${config.title}`,
        onLoaded: () => referenceContent.removeAttribute("aria-busy"),
      };
      const updatePicker = (group) => {
        for (const button of buttons) {
          button.setAttribute(
            "aria-pressed",
            String(button.dataset.category === group.id),
          );
          button.disabled = false;
        }
        document.getElementById("category-title").textContent = group.label;
        document.getElementById("category-description").textContent =
          group.description;
      };
      for (const group of groups) {
        document.querySelector(`[data-count="${group.id}"]`).textContent =
          group.count;
      }
      const sortPicker = document.getElementById("sidebar-order");
      const initialGroup = currentGroup();
      updatePicker(initialGroup);
      const reference = window.Scalar.createApiReference("#scalar-reference", {
        ...baseOptions,
        content: initialGroup.schema,
      });
      const selectGroup = (group) => {
        referenceContent.setAttribute("aria-busy", "true");
        updatePicker(group);
        reference.updateConfiguration({
          ...baseOptions,
          content:
            sortPicker.value === "alpha"
              ? {
                  ...group.schema,
                  tags: [...group.schema.tags].sort((a, b) =>
                    a.name.localeCompare(b.name),
                  ),
                }
              : group.schema,
        });
      };
      for (const button of buttons) {
        button.addEventListener("click", () => {
          if (button.getAttribute("aria-pressed") === "true") return;
          const group = groups.find(
            (item) => item.id === button.dataset.category,
          );
          const url = new URL(window.location.href);
          url.searchParams.set("category", group.id);
          url.hash = "";
          window.history.pushState(null, "", url);
          selectGroup(group);
          referenceContent.scrollIntoView({
            block: "start",
            behavior: "instant",
          });
        });
      }
      sortPicker.addEventListener("change", () => selectGroup(currentGroup()));
      window.addEventListener("popstate", () => selectGroup(currentGroup()));
      status.hidden = true;
    } catch {
      status.textContent =
        "We couldn’t load the API reference. Check that the OpenAPI endpoint is available, then try again.";
      const retry = document.createElement("button");
      retry.type = "button";
      retry.textContent = "Retry";
      retry.addEventListener("click", () => {
        status.textContent = "Loading your API reference…";
        loadReference();
      });
      status.append(retry);
    }
  }
  loadReference();
})();
