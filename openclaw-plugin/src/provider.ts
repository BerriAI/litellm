/**
 * Registers "clawrouter" as a provider in OpenClaw via configPatch.
 *
 * The provider points at the local LiteLLM proxy and exposes a single
 * "auto" model that routes requests through ClawRouter's tier system.
 */

interface ProviderOptions {
  port: number;
  masterKey: string;
}

/**
 * Creates the provider registration descriptor for ClawRouter.
 *
 * Returns an object suitable for `api.registerProvider()` that:
 * - Adds a "clawrouter" provider with baseUrl pointing at the local proxy
 * - Registers the "auto" model (smart routing)
 * - Sets clawrouter/auto as the default model
 */
export function createClawRouterProvider({ port, masterKey }: ProviderOptions) {
  return {
    id: "clawrouter",
    label: "ClawRouter",
    auth: [
      {
        id: "auto-setup",
        label: "Auto-configure from existing keys",
        kind: "custom" as const,
        run: async () => {
          return {
            profiles: [
              {
                profileId: "default",
                credential: {
                  type: "api_key",
                  provider: "clawrouter",
                  key: masterKey,
                },
              },
            ],
            configPatch: {
              models: {
                providers: {
                  clawrouter: {
                    baseUrl: `http://127.0.0.1:${port}/v1`,
                    apiKey: masterKey,
                    api: "openai-completions",
                    models: [
                      {
                        id: "auto",
                        name: "ClawRouter Auto",
                        reasoning: false,
                        input: ["text"],
                        cost: {
                          input: 0,
                          output: 0,
                          cacheRead: 0,
                          cacheWrite: 0,
                        },
                        contextWindow: 128000,
                        maxTokens: 8192,
                      },
                    ],
                  },
                },
              },
              agents: {
                defaults: {
                  model: { primary: "clawrouter/auto" },
                },
              },
            },
            defaultModel: "clawrouter/auto",
          };
        },
      },
    ],
  };
}
