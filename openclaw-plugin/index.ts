/**
 * ClawRouter OpenClaw Plugin
 *
 * Extracts API keys from OpenClaw auth profiles and runs setup.sh.
 */

import { emptyPluginConfigSchema } from "openclaw/plugin-sdk";
import type { OpenClawPluginApi } from "openclaw/plugin-sdk";
import { createClawRouterService } from "./src/service.js";

const plugin = {
  id: "clawrouter",
  name: "ClawRouter",
  description:
    "Extracts OpenClaw API keys and runs litellm/setup.sh",
  configSchema: emptyPluginConfigSchema(),

  register(api: OpenClawPluginApi) {
    const gitRepo =
      (api.pluginConfig?.gitRepo as string) ??
      "https://github.com/Counterweight-AI/litellm.git";

    api.registerService(createClawRouterService({ gitRepo }));
  },
};

export default plugin;
