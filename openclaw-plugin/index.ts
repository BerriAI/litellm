/**
 * ClawRouter OpenClaw Plugin
 *
 * Smart model routing — routes LLM requests to low/mid/top tier models
 * based on regex classification of prompt content.
 */

import { emptyPluginConfigSchema } from "openclaw/plugin-sdk";
import type { OpenClawPluginApi } from "openclaw/plugin-sdk";
import { createClawRouterService } from "./src/service.js";
import { createClawRouterProvider } from "./src/provider.js";

const plugin = {
  id: "clawrouter",
  name: "ClawRouter",
  description:
    "Smart model routing — routes to low/mid/top tier models based on prompt content",
  configSchema: emptyPluginConfigSchema(),

  register(api: OpenClawPluginApi) {
    const port = (api.pluginConfig?.port as number) ?? 4000;
    const masterKey =
      (api.pluginConfig?.masterKey as string) ?? "sk-clawrouter";
    const gitRepo =
      (api.pluginConfig?.gitRepo as string) ??
      "https://github.com/Counterweight-AI/litellm.git";

    api.registerService(
      createClawRouterService({ port, masterKey, gitRepo }),
    );
    api.registerProvider(createClawRouterProvider({ port, masterKey }));
  },
};

export default plugin;
