import { describe, it, expect } from "vitest";
import type { ComplexityRouterConfigValue } from "@/components/add_model/ComplexityRouterConfig";
import { BUNDLED_PRESETS_RESPONSE } from "../../tests/mocks/autoRouterPresets";
import {
  hydratePresets,
  AutoRouterPreset,
  getRequiredModelsInPreset,
  getMissingModelsInPreset,
  getRequiredModels,
  getMissingModels,
  getReferencedModelsError,
  buildEmptyPrefill,
  applyPresetTiers,
  buildModelAvailability,
  deploymentRefsFromModelInfo,
  normalizeModelName,
  resolveAvailableModel,
  resolveAvailableModels,
} from "./autorouter_presets";
import { DEFAULT_MATCH_THRESHOLD } from "@/components/add_model/SemanticKeywordMatching";
import { DEFAULT_ESCALATION_KEYWORDS } from "@/components/add_model/EscalationKeywords";

const groupsOnly = (models: Iterable<string>) => buildModelAvailability(models, []);

// Hydrated from the real bundled catalog so a catalog edit flows into these expectations.
const PRESETS = hydratePresets(BUNDLED_PRESETS_RESPONSE);
const getAllPresets = (): AutoRouterPreset[] => PRESETS;
const getPresetByKey = (key: string): AutoRouterPreset | undefined => PRESETS.find((p) => p.key === key);

describe("autorouter_presets", () => {
  it("hydrates exactly the bundled presets", () => {
    const presets = getAllPresets();
    expect(presets.map((p) => p.label).sort()).toEqual([
      "1M Context",
      "Anthropic Family",
      "Gemini Family",
      "Lite",
      "OpenAI Family",
    ]);
    // Every preset carries all four fields the UI relies on; a JSON typo dropping one fails here.
    for (const p of presets) {
      expect(p).toMatchObject({ key: expect.any(String), label: expect.any(String), description: expect.any(String) });
      expect(p.complexity_router_config.tiers).toBeTruthy();
    }
  });

  // applyPresetTiers resolves every model reference through normalizeModelName, so two spellings
  // of the same model in one tier (e.g. "claude-sonnet-4-5" and "claude-sonnet-4.5") collapse to one
  // key. For tier_model_configs that silently drops one model's litellm_params; catch it in the
  // bundled data itself, since nothing else validates preset authoring.
  it("never spells the same model two ways within a single tier", () => {
    for (const preset of getAllPresets()) {
      const { tiers, tier_model_configs: configs } = preset.complexity_router_config;
      for (const tier of Object.keys(tiers) as (keyof typeof tiers)[]) {
        const fromTierList = tiers[tier] ?? [];
        const fromConfigs = (configs?.[tier] ?? []).map((entry) => entry.model_name);
        const names = new Set([...fromTierList, ...fromConfigs]);
        const byNormalized = new Map<string, string[]>();
        for (const name of names) {
          const key = normalizeModelName(name);
          byNormalized.set(key, [...(byNormalized.get(key) ?? []), name]);
        }
        for (const spellings of byNormalized.values()) {
          expect(new Set(spellings).size, `${preset.key}.${tier}: ${spellings.join(", ")}`).toBe(1);
        }
      }
    }
  });

  it("resolves a preset by its stable JSON key, not its display label", () => {
    expect(getPresetByKey("anthropic_family")?.label).toBe("Anthropic Family");
    expect(getPresetByKey("does_not_exist")).toBeUndefined();
  });

  it("keeps every preset free of adaptive/quality settings", () => {
    for (const { complexity_router_config: config } of getAllPresets()) {
      expect(config.adaptive).toBeUndefined();
      expect(config.adaptive_weights).toBeUndefined();
      expect(config.adaptive_eligible).toBeUndefined();
      expect(config.tier_distance_penalty).toBeUndefined();
    }
  });

  it("keeps every preset on the shipped scorer knobs, so a preset cannot pin one to today's numbers", () => {
    for (const { complexity_router_config: config } of getAllPresets()) {
      expect(config.tier_boundaries).toBeUndefined();
      expect(config.token_thresholds).toBeUndefined();
      expect(config.dimension_weights).toBeUndefined();
      expect(config.session_affinity_ttl_seconds).toBeUndefined();
    }
  });

  it("keeps every preset free of custom dimensions, so applying one never adds scoring rows", () => {
    for (const { complexity_router_config: config } of getAllPresets()) {
      expect(config.custom_dimensions).toBeUndefined();
    }
    const config = getPresetByKey("anthropic_family")!.complexity_router_config;
    expect(
      applyPresetTiers(buildEmptyPrefill().complexityRouterConfig, config, groupsOnly([])).custom_dimensions,
    ).toBeUndefined();
  });

  it("resets both scoring overrides when the form falls back to an empty prefill", () => {
    expect(buildEmptyPrefill().complexityRouterConfig.custom_dimensions).toBeUndefined();
    expect(buildEmptyPrefill().complexityRouterConfig.dimension_weights).toBeUndefined();
  });

  it("preserves session affinity and custom scoring settings", () => {
    const config = PRESETS[0].complexity_router_config;
    const current = {
      ...buildEmptyPrefill().complexityRouterConfig,
      session_affinity_ttl_seconds: 300,
      dimension_weights: { codePresence: 0.4 },
      custom_dimensions: [{ id: "domain", name: "domain", weight: 0.9, keywords: ["orbitmesh"] }],
    };
    const selected = applyPresetTiers(current, config, groupsOnly([]));
    expect(selected.session_affinity_ttl_seconds).toBe(300);
    expect(selected.dimension_weights).toEqual(current.dimension_weights);
    expect(selected.custom_dimensions).toEqual(current.custom_dimensions);
  });

  it("keeps the model-family presets on the heuristic classifier", () => {
    for (const key of ["anthropic_family", "gemini_family", "openai_family"]) {
      expect(getPresetByKey(key)!.complexity_router_config.classifier_type).toBe("heuristic");
    }
  });

  // The lite preset ships the LLM classifier with the bundled agentic rubric rather than an inline
  // system_prompt, so rubric tuning in the backend reaches it without a JSON edit. Its classifier
  // model doubles as the SIMPLE tier model, so availability gating stays at exactly four models.
  it("pins the lite preset's LLM classifier config and required models", () => {
    const lite = getPresetByKey("lite")!;
    const config = lite.complexity_router_config;
    expect(config.classifier_type).toBe("llm");
    expect(config.classifier_llm_config).toEqual({
      model: "deepseek-v4-flash",
      timeout_ms: 3000,
      classification_rubric: "agentic",
    });
    expect(config.classifier_context_window_size).toBe(0);
    expect(config.classifier_context_per_turn_chars).toBeUndefined();
    expect(getRequiredModelsInPreset(lite)).toEqual(
      new Set(["deepseek-v4-flash", "muse-spark-1.3", "kimi-k3", "claude-opus-5-5"]),
    );
  });

  it("pins the anthropic preset's reasoning tier to Fable 5.1 at high thinking", () => {
    const config = getPresetByKey("anthropic_family")!.complexity_router_config;
    expect(config.tiers.COMPLEX).toEqual(["claude-opus-5-5"]);
    expect(config.tiers.REASONING).toEqual(["claude-fable-5-1"]);
    expect(config.tier_model_configs).toEqual({
      REASONING: [{ model_name: "claude-fable-5-1", litellm_params: { reasoning_effort: "high" } }],
    });
  });

  // Kimi K3 at max needs the map to declare max for kimi-k3, which is the commit below this one.
  it("pins the lite preset's per-tier reasoning efforts", () => {
    expect(getPresetByKey("lite")!.complexity_router_config.tier_model_configs).toEqual({
      MEDIUM: [{ model_name: "muse-spark-1.3", litellm_params: { reasoning_effort: "xhigh" } }],
      COMPLEX: [{ model_name: "kimi-k3", litellm_params: { reasoning_effort: "max" } }],
    });
  });

  // serializeTierModelConfigs filters on the tier's models, so a stray name drops silently.
  it("never names a model in tier_model_configs that its own tier does not hold", () => {
    for (const preset of getAllPresets()) {
      const { tiers, tier_model_configs: configs } = preset.complexity_router_config;
      for (const [tier, entries] of Object.entries(configs ?? {})) {
        for (const entry of entries) {
          expect(tiers[tier as keyof typeof tiers] ?? [], `${preset.key}.${tier}`).toContain(entry.model_name);
        }
      }
    }
  });

  it.each([false, true])("preserves the chosen modality settings %s", (enabled) => {
    const preset = PRESETS[0];
    const current = {
      ...buildEmptyPrefill().complexityRouterConfig,
      modality_routing: enabled,
      modality_pin_override: enabled,
    };
    const selected = applyPresetTiers(
      current,
      preset.complexity_router_config,
      groupsOnly(getRequiredModelsInPreset(preset)),
    );
    expect(selected.modality_routing).toBe(enabled);
    expect(selected.modality_pin_override).toBe(enabled);
  });

  it("ships every bundled preset with both modality flags written out, since the payload type requires them", () => {
    for (const preset of getAllPresets()) {
      expect(preset.complexity_router_config.modality_routing, preset.key).toBe(false);
      expect(preset.complexity_router_config.modality_pin_override, preset.key).toBe(false);
    }
  });

  it("prefills the anthropic preset's effort through to tier_model_params", () => {
    const preset = getPresetByKey("anthropic_family")!;
    const prefill = applyPresetTiers(
      buildEmptyPrefill().complexityRouterConfig,
      preset.complexity_router_config,
      groupsOnly(getRequiredModelsInPreset(preset)),
    );
    expect(prefill.tier_model_params).toEqual({
      REASONING: { "claude-fable-5-1": { reasoning_effort: "high" } },
    });
  });

  it("prefills the lite preset's efforts through to tier_model_params", () => {
    const lite = getPresetByKey("lite")!;
    const prefill = applyPresetTiers(
      buildEmptyPrefill().complexityRouterConfig,
      lite.complexity_router_config,
      groupsOnly(getRequiredModelsInPreset(lite)),
    );
    expect(prefill.tier_model_params).toEqual({
      MEDIUM: { "muse-spark-1.3": { reasoning_effort: "xhigh" } },
      COMPLEX: { "kimi-k3": { reasoning_effort: "max" } },
    });
  });

  it("pins the OpenAI preset to the Luna, Terra, Sol, and Astra progression", () => {
    const preset = getPresetByKey("openai_family")!;
    const expectedTiers = {
      SIMPLE: ["gpt-6-luna"],
      MEDIUM: ["gpt-5.6-terra"],
      COMPLEX: ["gpt-6-sol"],
      REASONING: ["gpt-6-astra"],
    };
    expect(preset.complexity_router_config.tiers).toEqual(expectedTiers);
    expect(preset.complexity_router_config.tier_model_configs).toEqual({
      REASONING: [{ model_name: "gpt-6-astra", litellm_params: { reasoning_effort: "xhigh" } }],
    });
    const prefill = applyPresetTiers(
      buildEmptyPrefill().complexityRouterConfig,
      preset.complexity_router_config,
      groupsOnly(getRequiredModelsInPreset(preset)),
    );
    expect(prefill.tier_model_params).toEqual({
      REASONING: { "gpt-6-astra": { reasoning_effort: "xhigh" } },
    });
  });

  it("pins the 1M context preset to Luna, Terra, Sol, and Opus at high thinking", () => {
    const preset = getPresetByKey("1m_context")!;
    const expectedTiers = {
      SIMPLE: ["gpt-6-luna"],
      MEDIUM: ["gpt-5.6-terra"],
      COMPLEX: ["gpt-6-sol"],
      REASONING: ["claude-opus-5-5"],
    };
    expect(preset.complexity_router_config.classifier_type).toBe("heuristic_v2");
    expect(preset.complexity_router_config.tiers).toEqual(expectedTiers);
    expect(preset.complexity_router_config.tier_model_configs).toEqual({
      REASONING: [{ model_name: "claude-opus-5-5", litellm_params: { reasoning_effort: "high" } }],
    });
    const prefill = applyPresetTiers(
      buildEmptyPrefill().complexityRouterConfig,
      preset.complexity_router_config,
      groupsOnly(getRequiredModelsInPreset(preset)),
    );
    expect(prefill.tier_model_params).toEqual({
      REASONING: { "claude-opus-5-5": { reasoning_effort: "high" } },
    });
  });

  it("pins the gemini preset to concrete model ids, never Google's hot-swapping -latest aliases", () => {
    const gemini = getPresetByKey("gemini_family")!;
    const config = gemini.complexity_router_config;
    expect(config.classifier_type).toBe("heuristic");
    expect(config.classifier_llm_config).toBeUndefined();
    const expectedTiers = {
      SIMPLE: ["gemini-3.5-flash-lite"],
      MEDIUM: ["gemini-3.8-flash"],
      COMPLEX: ["gemini-3.8-flash"],
      REASONING: ["gemini-3.1-pro-preview"],
    };
    expect(config.tiers).toEqual(expectedTiers);
    const required = getRequiredModelsInPreset(gemini);
    for (const model of required) expect(model).not.toMatch(/-latest$/);
    expect(required.size).toBe(new Set(Object.values(expectedTiers).flat()).size);
  });

  it("collects every tier model as a required model", () => {
    const preset = getPresetByKey("anthropic_family")!;
    const required = getRequiredModelsInPreset(preset);
    const tierModels = Object.values(preset.complexity_router_config.tiers).flat();
    expect(tierModels.length).toBeGreaterThan(0);
    for (const model of tierModels) expect(required.has(model)).toBe(true);
  });

  it("reports only the models the caller is missing, and none when the family is fully available", () => {
    const preset = getPresetByKey("openai_family")!;
    const required = [...getRequiredModelsInPreset(preset)];
    const [held] = required;

    expect(getMissingModelsInPreset(preset, groupsOnly([held]))).toEqual(required.filter((m) => m !== held).sort());
    expect(getMissingModelsInPreset(preset, groupsOnly(required))).toEqual([]);
  });

  // Admins spell version numbers with either "-" or "." (claude-sonnet-4-5 vs claude-sonnet-4.5);
  // a caller who only registered one form still satisfies a preset that names the other. The
  // caller's spellings are derived from the preset itself so that renaming a preset's models in
  // autorouter_presets.json can't quietly turn this into a no-op (the inequality below fails
  // instead, if no preset model carries a version number at all).
  it("treats a preset's model as available under either version-separator spelling", () => {
    const preset = getPresetByKey("anthropic_family")!;
    const required = [...getRequiredModelsInPreset(preset)];
    const dottedSpellings = required.map((model) => model.replace(/(\d)-(\d)/g, "$1.$2"));

    expect(dottedSpellings).not.toEqual(required);
    expect(getMissingModelsInPreset(preset, groupsOnly(dottedSpellings))).toEqual([]);
  });

  // The two-arm mirror: a differently-punctuated preset model must not be reported missing.
  it("does not flag a differently-punctuated model as missing via getMissingModels directly", () => {
    const missing = getMissingModels(
      { tiers: { SIMPLE: ["claude-sonnet-4-5"], MEDIUM: [], COMPLEX: [], REASONING: [] } },
      groupsOnly(["claude-sonnet-4.5"]),
    );
    expect(missing).toEqual([]);
  });

  // A classifier_llm_config placeholder is seeded with model: "" before a caller picks one; an
  // empty string is not a real model reference and must not be reported as an unavailable model.
  it("does not treat an empty-string classifier or embedding model as a required model", () => {
    const required = getRequiredModels({
      tiers: { SIMPLE: ["gpt-5-nano"], MEDIUM: [], COMPLEX: [], REASONING: [] },
      classifier_llm_config: { model: "", timeout_ms: 5000 },
      embedding_model: "",
    });
    expect(required).toEqual(new Set(["gpt-5-nano"]));
  });

  describe("deployment matching (underlying provider model IDs)", () => {
    const availabilityFor = (modelGroup: string, underlyingModel: string) =>
      buildModelAvailability([modelGroup], [{ modelGroup, underlyingModels: [underlyingModel] }]);

    it.each([
      ["provider prefix", "my-claude-fast", "anthropic/claude-haiku-4-5", "claude-haiku-4-5"],
      ["bedrock region+namespace+revision", "bedrock-opus", "bedrock/us.anthropic.claude-opus-5-v1:0", "claude-opus-5"],
      ["bedrock date stamp", "bedrock-opus41", "bedrock/us.anthropic.claude-opus-4-1-20250805-v1:0", "claude-opus-4-1"],
      ["dotted version", "team-gpt", "openai/gpt-5.4", "gpt-5.4"],
      ["vertex @tag", "vertex-frontier", "vertex_ai/claude-fable-5@default", "claude-fable-5"],
      ["bedrock 1m context label", "opus-1m", "bedrock/us.anthropic.claude-opus-5-v1:0[1m]", "claude-opus-5"],
    ])("resolves a %s deployment and prefills the admin's group name", (_label, group, underlying, presetModel) => {
      const availability = availabilityFor(group, underlying);
      const config = {
        tiers: { SIMPLE: [presetModel], MEDIUM: [], COMPLEX: [], REASONING: [] },
        classifier_type: "heuristic" as const,
        classification_mode: "every_request" as const,
        session_affinity: false,
        deployment_affinity: true,
      };
      expect(getMissingModels(config, availability)).toEqual([]);
      expect(applyPresetTiers(buildEmptyPrefill().complexityRouterConfig, config, availability).tiers.SIMPLE).toEqual([
        group,
      ]);
    });

    it.each([
      ["gpt-5.4", "openai/gpt-5.4-mini"],
      ["gpt-5.4-mini", "openai/gpt-5.4"],
      ["o3", "openai/o3-mini"],
      ["o3-mini", "openai/o3"],
      ["some-model", "prov/some-model-20991399"],
    ])("never lets %s be satisfied by a deployment of %s", (presetModel, underlying) => {
      const availability = availabilityFor("some-group", underlying);
      const config = { tiers: { SIMPLE: [presetModel], MEDIUM: [], COMPLEX: [], REASONING: [] } };
      expect(getMissingModels(config, availability)).toEqual([presetModel]);
    });

    it("never indexes a wildcard deployment", () => {
      const availability = availabilityFor("openai-wild", "openai/*");
      expect(availability.underlyingIndex.size).toBe(0);
    });

    it("ignores a deployment whose group is not itself an available model group", () => {
      const availability = buildModelAvailability(
        ["some-other-group"],
        [{ modelGroup: "orphan-group", underlyingModels: ["anthropic/claude-opus-5"] }],
      );
      expect(availability.underlyingIndex.size).toBe(0);
    });

    it("returns every configured group serving the same underlying model", () => {
      const availability = buildModelAvailability(
        ["z-group", "a-group"],
        [
          { modelGroup: "z-group", underlyingModels: ["anthropic/claude-sonnet-5"] },
          { modelGroup: "a-group", underlyingModels: ["bedrock/us.anthropic.claude-sonnet-5-v1:0"] },
        ],
      );

      expect(resolveAvailableModels("anthropic/claude-sonnet-5", availability)).toEqual(["a-group", "z-group"]);
    });

    it.each([
      ["OpenAI", getPresetByKey("openai_family")!.complexity_router_config.tiers.MEDIUM[0], "openai", "azure"],
      [
        "Anthropic",
        getPresetByKey("anthropic_family")!.complexity_router_config.tiers.COMPLEX[0],
        "anthropic",
        "bedrock",
      ],
      ["Gemini", getPresetByKey("gemini_family")!.complexity_router_config.tiers.SIMPLE[0], "gemini", "vertex_ai"],
      ["DeepSeek", getPresetByKey("lite")!.complexity_router_config.tiers.SIMPLE[0], "deepseek", "openrouter"],
      ["Muse", getPresetByKey("lite")!.complexity_router_config.tiers.MEDIUM[0], "meta", "openrouter"],
      ["Kimi", getPresetByKey("lite")!.complexity_router_config.tiers.COMPLEX[0], "moonshot", "openrouter"],
      ["Grok", "grok-4.7", "xai", "openrouter"],
    ])(
      "prefills %s through its native provider and falls back when only the cloud group is available",
      (_family, model, native, cloud) => {
        const deployments = [
          { modelGroup: "a-cloud", underlyingModels: [`${cloud}/${model}`] },
          { modelGroup: "z-native", underlyingModels: [`${native}/${model}`] },
        ];
        const config = {
          tiers: { SIMPLE: [model], MEDIUM: [], COMPLEX: [], REASONING: [] },
          tier_model_configs: { SIMPLE: [{ model_name: model, litellm_params: { reasoning_effort: "high" } }] },
          classifier_type: "llm" as const,
          classifier_llm_config: { model, timeout_ms: 3000 },
          classification_mode: "every_request" as const,
          session_affinity: false,
          deployment_affinity: true,
          modality_routing: false,
          modality_pin_override: false,
        };

        for (const [groups, selected] of [
          [["a-cloud", "z-native"], "z-native"],
          [["a-cloud"], "a-cloud"],
        ] as const) {
          const availability = buildModelAvailability(groups, deployments);
          const prefill = applyPresetTiers(buildEmptyPrefill().complexityRouterConfig, config, availability);

          expect(prefill.tiers.SIMPLE).toEqual([selected]);
          expect(prefill.tier_model_params).toEqual({ SIMPLE: { [selected]: { reasoning_effort: "high" } } });
          expect(prefill.classifier_llm_config).toBeUndefined();
        }
      },
    );

    it.each(["claude-opus-5-5", "claude-opus-5.5"])(
      "prefers a native deployment over the cloud group named %s",
      (cloudGroup) => {
        const availability = buildModelAvailability(
          [cloudGroup, "z-native"],
          [
            { modelGroup: cloudGroup, underlyingModels: ["bedrock/us.anthropic.claude-opus-5-5-v1:0"] },
            { modelGroup: "z-native", underlyingModels: ["anthropic/claude-opus-5-5"] },
          ],
        );

        expect(resolveAvailableModel("claude-opus-5-5", availability)).toBe("z-native");
        expect(resolveAvailableModels("claude-opus-5-5", availability)).toEqual([cloudGroup]);
      },
    );

    it("breaks ties between native groups alphabetically regardless of deployment order", () => {
      const availability = buildModelAvailability(
        ["z-native", "a-native"],
        [
          { modelGroup: "z-native", underlyingModels: ["anthropic/claude-opus-5-5"] },
          { modelGroup: "a-native", underlyingModels: ["anthropic/claude-opus-5-5"] },
        ],
      );

      expect(resolveAvailableModel("claude-opus-5-5", availability)).toBe("a-native");
    });

    it.each(["gpt-6-sol", "claude-opus-5-5"])("recognizes the native default of bare %s", (model) => {
      const availability = buildModelAvailability(
        ["a-cloud", "z-native"],
        [
          { modelGroup: "a-cloud", underlyingModels: [`openrouter/${model}`] },
          { modelGroup: "z-native", underlyingModels: [model] },
        ],
      );

      expect(resolveAvailableModel(model, availability)).toBe("z-native");
    });

    it.each(["bedrock/claude-opus-5-5", "unknown-model"])(
      "prefers an exclusively native group over one that also routes to %s",
      (otherModel) => {
        const deployments = [
          { modelGroup: "a-cloud", underlyingModels: ["bedrock/claude-opus-5-5"] },
          { modelGroup: "b-mixed", underlyingModels: ["anthropic/claude-opus-5-5"] },
          { modelGroup: "b-mixed", underlyingModels: [otherModel] },
          { modelGroup: "z-native", underlyingModels: ["anthropic/claude-opus-5-5"] },
        ];
        const availability = buildModelAvailability(["a-cloud", "b-mixed", "z-native"], deployments);

        expect(resolveAvailableModel("claude-opus-5-5", availability)).toBe("z-native");
        const noNativeGroup = buildModelAvailability(["a-cloud", "b-mixed"], deployments);
        expect(resolveAvailableModel("claude-opus-5-5", noNativeGroup)).toBe("a-cloud");
      },
    );

    it.each([
      { model: "azure/opaque-deployment", base_model: "openai/gpt-6-sol" },
      { model: "openai/gpt-6-sol", custom_llm_provider: "openrouter" },
    ])("keeps cloud routing authoritative over native-looking model metadata: %j", (litellmParams) => {
      const availability = buildModelAvailability(
        ["a-cloud", "z-native"],
        deploymentRefsFromModelInfo([
          { model_name: "a-cloud", litellm_params: litellmParams, model_info: { base_model: "openai/gpt-6-sol" } },
          { model_name: "z-native", litellm_params: { model: "openai/gpt-6-sol" } },
        ]),
      );

      expect(resolveAvailableModel("gpt-6-sol", availability)).toBe("z-native");
    });

    it("recognizes an explicit native provider on an otherwise unqualified model", () => {
      const availability = buildModelAvailability(
        ["a-cloud", "z-native"],
        deploymentRefsFromModelInfo([
          { model_name: "a-cloud", litellm_params: { model: "openrouter/meta/muse-spark-1.3" } },
          { model_name: "z-native", litellm_params: { model: "muse-spark-1.3", custom_llm_provider: "meta" } },
        ]),
      );

      expect(resolveAvailableModel("muse-spark-1.3", availability)).toBe("z-native");
    });

    it("preserves exact group-name precedence when no known native deployment is available", () => {
      const availability = buildModelAvailability(
        ["claude-opus-5", "renamed-opus"],
        [{ modelGroup: "renamed-opus", underlyingModels: ["bedrock/us.anthropic.claude-opus-5-v1:0"] }],
      );
      const config = {
        tiers: { SIMPLE: ["claude-opus-5"], MEDIUM: [], COMPLEX: [], REASONING: [] },
        classifier_type: "heuristic" as const,
        classification_mode: "every_request" as const,
        session_affinity: false,
        deployment_affinity: true,
      };
      expect(applyPresetTiers(buildEmptyPrefill().complexityRouterConfig, config, availability).tiers.SIMPLE).toEqual([
        "claude-opus-5",
      ]);
    });

    it.each(getAllPresets().map((preset) => [preset.key, preset] as const))(
      "fully resolves the %s preset through renamed deployments only",
      (_key, preset) => {
        const required = [...getRequiredModelsInPreset(preset)];
        const groups = required.map((_model, index) => `renamed-${index}`);
        const availability = buildModelAvailability(
          groups,
          required.map((model, index) => ({
            modelGroup: `renamed-${index}`,
            underlyingModels: [`someprovider/${model}`],
          })),
        );
        expect(getMissingModelsInPreset(preset, availability)).toEqual([]);
        const prefilled = applyPresetTiers(
          buildEmptyPrefill().complexityRouterConfig,
          preset.complexity_router_config,
          availability,
        );
        const prefilledModels = Object.values(prefilled.tiers).flat();
        expect(prefilledModels.length).toBeGreaterThan(0);
        for (const model of prefilledModels) expect(groups).toContain(model);
      },
    );
  });

  describe("wildcard deployment matching (expanded model groups)", () => {
    const wildcardDeployment = (pattern: string) => ({ modelGroup: pattern, underlyingModels: [pattern] });

    const simpleTierConfig = (presetModel: string) => ({
      tiers: { SIMPLE: [presetModel], MEDIUM: [], COMPLEX: [], REASONING: [] },
      classifier_type: "heuristic" as const,
      classification_mode: "every_request" as const,
      session_affinity: false,
      deployment_affinity: true,
    });

    it("resolves a preset model to a group expanded from a wildcard deployment", () => {
      const availability = buildModelAvailability(
        ["anthropic/*", "anthropic/claude-opus-5", "bedrock/anthropic.claude-opus-5"],
        [wildcardDeployment("anthropic/*")],
      );
      const config = simpleTierConfig("claude-opus-5");
      expect(getMissingModels(config, availability)).toEqual([]);
      expect(applyPresetTiers(buildEmptyPrefill().complexityRouterConfig, config, availability).tiers.SIMPLE).toEqual([
        "anthropic/claude-opus-5",
      ]);
    });

    it("normalizes an expanded group's namespaced own name the same way as a deployment's", () => {
      const availability = buildModelAvailability(
        ["bedrock/*", "bedrock/us.anthropic.claude-sonnet-5"],
        [wildcardDeployment("bedrock/*")],
      );
      expect(getMissingModels(simpleTierConfig("claude-sonnet-5"), availability)).toEqual([]);
    });

    it("anchors a partial wildcard pattern and treats its dots literally", () => {
      const availability = buildModelAvailability(
        ["bedrock/us.anthropic.claude-opus-5", "bedrock/usXanthropic.claude-fable-5"],
        [wildcardDeployment("bedrock/us.*")],
      );
      expect(getMissingModels(simpleTierConfig("claude-opus-5"), availability)).toEqual([]);
      expect(getMissingModels(simpleTierConfig("claude-fable-5"), availability)).toEqual(["claude-fable-5"]);
    });

    it.each([
      ["gpt-5.4", "openai/gpt-5.4-mini"],
      ["gpt-5.4-mini", "openai/gpt-5.4"],
      ["o3", "openai/o3-mini"],
    ])("never lets %s be satisfied by the expanded group %s", (presetModel, expandedGroup) => {
      const availability = buildModelAvailability(["openai/*", expandedGroup], [wildcardDeployment("openai/*")]);
      expect(getMissingModels(simpleTierConfig(presetModel), availability)).toEqual([presetModel]);
    });

    it("anchors the pattern's suffix and keeps middle segments in order", () => {
      const availability = buildModelAvailability(
        ["bedrock/us.anthropic.claude-opus-5", "bedrock/anthropic.us.claude-sonnet-5"],
        [wildcardDeployment("bedrock/*.anthropic.*")],
      );
      expect(getMissingModels(simpleTierConfig("claude-opus-5"), availability)).toEqual([]);
      expect(getMissingModels(simpleTierConfig("claude-sonnet-5"), availability)).toEqual(["claude-sonnet-5"]);
    });

    it("matches a pathological many-star pattern in linear time instead of backtracking", () => {
      const hostile = `prov/a*${"a*".repeat(30)}b`;
      const nonMatching = `prov/${"a".repeat(120)}`;
      const availability = buildModelAvailability([nonMatching], [wildcardDeployment(hostile)]);
      expect(availability.underlyingIndex.size).toBe(0);
    });

    it("expands a bare-star model_name through its underlying wildcard, not as match-all", () => {
      const availability = buildModelAvailability(
        ["openai/gpt-5.4", "team-a/claude-opus-5"],
        [{ modelGroup: "*", underlyingModels: ["openai/*"] }],
      );
      expect(getMissingModels(simpleTierConfig("gpt-5.4"), availability)).toEqual([]);
      expect(getMissingModels(simpleTierConfig("claude-opus-5"), availability)).toEqual(["claude-opus-5"]);
    });

    it.each([
      ["a bare-star underlying", "*"],
      ["a non-wildcard underlying", "openai/gpt-4o"],
      ["a slashless wildcard underlying", "gpt*"],
    ])("derives no pattern from a bare-star model_name with %s", (_label, underlying) => {
      const availability = buildModelAvailability(
        ["openai/gpt-5.4"],
        [{ modelGroup: "*", underlyingModels: [underlying] }],
      );
      expect(availability.underlyingIndex.size).toBe(0);
    });

    it("derives no pattern from a slashless wildcard model_name", () => {
      const availability = buildModelAvailability(["gpt-5.4"], [wildcardDeployment("gpt*")]);
      expect(availability.underlyingIndex.size).toBe(0);
    });

    it("does not trust a group's name when no wildcard deployment covers it", () => {
      const availability = buildModelAvailability(
        ["team-a/claude-opus-5", "openai/*"],
        [wildcardDeployment("openai/*")],
      );
      expect(getMissingModels(simpleTierConfig("claude-opus-5"), availability)).toEqual(["claude-opus-5"]);
    });

    it("never resolves to the wildcard group itself when the hub lists no expansions", () => {
      const availability = buildModelAvailability(["openai/*"], [wildcardDeployment("openai/*")]);
      expect(getMissingModels(simpleTierConfig("gpt-5.4"), availability)).toEqual(["gpt-5.4"]);
      expect(availability.underlyingIndex.size).toBe(0);
    });

    it("applies a wildcard deployment's pattern even when the wildcard group is not itself listed", () => {
      const availability = buildModelAvailability(["anthropic/claude-opus-5"], [wildcardDeployment("anthropic/*")]);
      expect(getMissingModels(simpleTierConfig("claude-opus-5"), availability)).toEqual([]);
    });

    it("keeps the groups-only availability strict even when expanded groups are listed", () => {
      const availability = groupsOnly(["anthropic/*", "anthropic/claude-opus-5"]);
      expect(getMissingModels(simpleTierConfig("claude-opus-5"), availability)).toEqual(["claude-opus-5"]);
    });

    it("prefers the alphabetically first covered group when several expansions serve the model", () => {
      const availability = buildModelAvailability(
        ["bedrock/us.anthropic.claude-opus-5", "anthropic/claude-opus-5", "bedrock/anthropic.claude-opus-5"],
        [wildcardDeployment("anthropic/*"), wildcardDeployment("bedrock/*")],
      );
      const config = simpleTierConfig("claude-opus-5");
      expect(applyPresetTiers(buildEmptyPrefill().complexityRouterConfig, config, availability).tiers.SIMPLE).toEqual([
        "anthropic/claude-opus-5",
      ]);
    });

    it.each(["native/*", "*"])("ranks wildcard groups using their routing deployment: %s", (nativePattern) => {
      const nativeGroup = nativePattern === "*" ? "openai/gpt-6-sol" : "native/gpt-6-sol";
      const availability = buildModelAvailability(
        ["azure/gpt-6-sol", nativeGroup],
        [
          { modelGroup: "azure/*", underlyingModels: ["openrouter/*"] },
          { modelGroup: nativePattern, underlyingModels: ["openai/*"] },
        ],
      );

      expect(resolveAvailableModel("gpt-6-sol", availability)).toBe(nativeGroup);
    });

    it("does not treat a native-looking wildcard group as native when its deployment uses the cloud", () => {
      const availability = buildModelAvailability(
        ["openai/gpt-6-sol", "z-native/gpt-6-sol"],
        [
          { modelGroup: "openai/*", underlyingModels: ["azure/*"] },
          { modelGroup: "z-native/*", underlyingModels: ["openai/*"] },
        ],
      );

      expect(resolveAvailableModel("gpt-6-sol", availability)).toBe("z-native/gpt-6-sol");
    });

    it("keeps literal native deployments ahead of a matching cloud wildcard", () => {
      const availability = buildModelAvailability(
        ["a-cloud", "team/gpt-6-sol"],
        [
          { modelGroup: "a-cloud", underlyingModels: ["azure/gpt-6-sol"] },
          { modelGroup: "team/gpt-6-sol", underlyingModels: ["openai/gpt-6-sol"] },
          { modelGroup: "team/*", underlyingModels: ["azure/*"] },
        ],
      );

      expect(resolveAvailableModel("gpt-6-sol", availability)).toBe("team/gpt-6-sol");
    });

    it("does not promote a bare-star expansion when its routing group also contains a cloud deployment", () => {
      const availability = buildModelAvailability(
        ["openai/gpt-6-sol", "z-native"],
        [
          { modelGroup: "*", underlyingModels: ["openai/*"] },
          { modelGroup: "*", underlyingModels: ["azure/*"] },
          { modelGroup: "z-native", underlyingModels: ["openai/gpt-6-sol"] },
        ],
      );

      expect(resolveAvailableModel("gpt-6-sol", availability)).toBe("z-native");
    });

    it("retains fallback ordering when overlapping wildcard routes have different providers", () => {
      const availability = buildModelAvailability(
        ["a-cloud", "team/gpt-6-sol"],
        [
          { modelGroup: "a-cloud", underlyingModels: ["azure/gpt-6-sol"] },
          { modelGroup: "team/*", underlyingModels: ["azure/*"] },
          { modelGroup: "team/gpt-*", underlyingModels: ["openai/gpt-*"] },
        ],
      );

      expect(resolveAvailableModel("gpt-6-sol", availability)).toBe("a-cloud");
    });

    it.each(getAllPresets().map((preset) => [preset.key, preset] as const))(
      "fully resolves the %s preset through wildcard-expanded groups only",
      (_key, preset) => {
        const required = [...getRequiredModelsInPreset(preset)];
        const expandedGroups = required.map((model) => `someprovider/${model}`);
        const availability = buildModelAvailability(
          ["someprovider/*", ...expandedGroups],
          [wildcardDeployment("someprovider/*")],
        );
        expect(getMissingModelsInPreset(preset, availability)).toEqual([]);
        const prefilled = applyPresetTiers(
          buildEmptyPrefill().complexityRouterConfig,
          preset.complexity_router_config,
          availability,
        );
        const prefilledModels = Object.values(prefilled.tiers).flat();
        expect(prefilledModels.length).toBeGreaterThan(0);
        for (const model of prefilledModels) expect(expandedGroups).toContain(model);
      },
    );
  });

  describe("deploymentRefsFromModelInfo", () => {
    it("keeps litellm_params.model and model_info.base_model, drops rows with neither or no name", () => {
      const refs = deploymentRefsFromModelInfo([
        {
          model_name: "azure-prod",
          litellm_params: { model: "azure/my-deployment" },
          model_info: { base_model: "azure/gpt-5.4" },
        },
        { model_name: "no-underlying", litellm_params: {}, model_info: {} },
        { litellm_params: { model: "openai/gpt-5.4" } },
      ]);
      expect(refs).toEqual([
        { modelGroup: "azure-prod", underlyingModels: ["azure/my-deployment", "azure/gpt-5.4"], provider: "azure" },
      ]);
    });

    it("lets an azure deployment resolve through base_model declared under litellm_params", () => {
      const availability = buildModelAvailability(
        ["azure-lp"],
        deploymentRefsFromModelInfo([
          {
            model_name: "azure-lp",
            litellm_params: { model: "azure/opaque-deployment-name", base_model: "azure/gpt-5.4" },
          },
        ]),
      );
      const config = { tiers: { SIMPLE: ["gpt-5.4"], MEDIUM: [], COMPLEX: [], REASONING: [] } };
      expect(getMissingModels(config, availability)).toEqual([]);
    });

    it("lets an azure deployment resolve through its admin-declared base_model", () => {
      const availability = buildModelAvailability(
        ["azure-prod"],
        deploymentRefsFromModelInfo([
          {
            model_name: "azure-prod",
            litellm_params: { model: "azure/opaque-deployment-name" },
            model_info: { base_model: "azure/gpt-5.4" },
          },
        ]),
      );
      const config = { tiers: { SIMPLE: ["gpt-5.4"], MEDIUM: [], COMPLEX: [], REASONING: [] } };
      expect(getMissingModels(config, availability)).toEqual([]);
    });
  });

  describe("getReferencedModelsError", () => {
    it.each(["jev", "heuristic"] as const)(
      "only checks a removed decision deployment when %s is active",
      (classifierType) => {
        const config = {
          tiers: { SIMPLE: ["solver"], MEDIUM: [], COMPLEX: [], REASONING: [] },
          classifierType,
          classifierLlmConfig: undefined,
          decisionDeploymentName: "removed-decision",
          semanticMatchingEnabled: false,
          embeddingModel: undefined,
        };
        const availability = groupsOnly(["solver"]);
        expect(getReferencedModelsError(config, availability)).toBe(
          classifierType === "jev" ? "Model(s) no longer available: removed-decision" : null,
        );
      },
    );

    const tiers = { SIMPLE: ["gpt-5-nano"], MEDIUM: [], COMPLEX: [], REASONING: [] };
    const available = groupsOnly(["gpt-5-nano"]);
    // Both fields are always populated with a model missing from `available`; only the
    // enabled/disabled toggles below decide whether that missing model gets reported.
    const params = {
      classifierLlmConfig: { model: "missing-classifier", timeout_ms: 5000 },
      embeddingModel: "missing-embed",
    };

    // Bugbot-found bug class from #35199's history: a classifier/embedding model left selected
    // from a prior toggle must not block submit once that toggle is off again, since
    // buildComplexityRouterConfig never emits the field in that state - only a model whose toggle
    // is on should ever be reported.
    it.each([
      ["both toggles off", "heuristic", false, null],
      ["classifier type llm, semantic matching off", "llm", false, "missing-classifier"],
      ["classifier type heuristic, semantic matching on", "heuristic", true, "missing-embed"],
      ["both toggles on", "llm", true, "missing-classifier, missing-embed"],
    ] as const)("%s", (_label, classifierType, semanticMatchingEnabled, missingModels) => {
      const config = { tiers, classifierType, semanticMatchingEnabled, ...params };
      const error = getReferencedModelsError(config, available);
      expect(error).toBe(missingModels ? `Model(s) no longer available: ${missingModels}` : null);
    });
  });

  describe("buildEmptyPrefill", () => {
    it("resets every field to its default, empty state", () => {
      const expected = {
        complexityRouterConfig: {
          tiers: { SIMPLE: [], MEDIUM: [], COMPLEX: [], REASONING: [] },
          classifier_type: "heuristic",
        },
        customTechnicalKeywords: [],
        keywordTierRules: [],
        semanticMatchingEnabled: false,
        embeddingModel: undefined,
        matchThreshold: DEFAULT_MATCH_THRESHOLD,
        escalationKeywords: DEFAULT_ESCALATION_KEYWORDS,
      };
      expect(buildEmptyPrefill()).toEqual(expected);
    });
  });

  describe("applyPresetTiers", () => {
    const initial: ComplexityRouterConfigValue = {
      classifier_type: "jev",
      jev_classifier_config: { deployment_name: "decision", timeout_ms: 4200 },
      classifier_llm_config: { model: "judge", classification_rubric: "business" },
      classification_mode: "user_turn",
      session_affinity: false,
      classifier_context_window_size: 8,
      heuristic_v2_success_threshold: 0.7,
      tiers: { SIMPLE: ["old"], MEDIUM: [], COMPLEX: [], REASONING: [], NON_REASONING: ["relay"] },
      tier_model_params: { SIMPLE: { old: { temperature: 0.3 } }, NON_REASONING: { relay: { max_tokens: 10 } } },
      enable_non_reasoning_tier: true,
      tier_labels: { SIMPLE: "Quick" },
      default_model: "fallback",
      plan_mode_min_tier: "REASONING",
      adaptive: true,
      deployment_affinity: false,
    };

    describe.each(["jev", "heuristic", "heuristic_v2", "llm"] as const)("%s classifier", (classifier_type) => {
      it.each(PRESETS)("keeps non-template settings when selecting $label repeatedly", (preset) => {
        const current = { ...initial, classifier_type };
        const availability = groupsOnly(PRESETS.flatMap((item) => [...getRequiredModelsInPreset(item)]));
        const selected = applyPresetTiers(current, preset.complexity_router_config, availability);
        const { tiers: _oldTiers, tier_model_params: _oldParams, ...settings } = current;
        const { tiers, tier_model_params, ...selectedSettings } = selected;
        expect(selectedSettings).toEqual(settings);
        expect(tiers).toEqual({ ...preset.complexity_router_config.tiers, NON_REASONING: ["relay"] });
        expect(tier_model_params?.NON_REASONING).toEqual(initial.tier_model_params?.NON_REASONING);
        expect(tier_model_params?.SIMPLE?.old).toBeUndefined();
        const nextPreset = PRESETS.find((item) => item.key !== preset.key)!;
        const next = applyPresetTiers(selected, nextPreset.complexity_router_config, availability);
        expect(next).toEqual(applyPresetTiers(current, nextPreset.complexity_router_config, availability));
      });
    });

    it("leaves a custom tier draft untouched", () => {
      const current = {
        ...initial,
        custom_tier_set: {
          tiers: [{ id: "custom", name: "CUSTOM", definition: "custom work", models: ["old"] }],
          fallback_tier_id: "custom",
        },
      };
      expect(applyPresetTiers(current, PRESETS[0].complexity_router_config, groupsOnly([]))).toBe(current);
    });

    it("checks only tier models when determining template availability", () => {
      const preset = {
        ...PRESETS[0],
        complexity_router_config: {
          ...PRESETS[0].complexity_router_config,
          classifier_llm_config: { model: "unavailable-judge" },
          embedding_model: "unavailable-embedding",
          default_model: "unavailable-default",
        },
      };
      const tierModels = Object.values(preset.complexity_router_config.tiers).flat();
      expect(getRequiredModelsInPreset(preset)).toEqual(new Set(tierModels));
      expect(getMissingModelsInPreset(preset, groupsOnly(tierModels))).toEqual([]);
    });

    it("rewrites a preset's model name to the caller's differently-punctuated registered spelling", () => {
      const config = {
        tiers: { SIMPLE: ["claude-sonnet-4-5"], MEDIUM: [], COMPLEX: [], REASONING: [] },
        classifier_type: "heuristic" as const,
        classification_mode: "every_request" as const,
        session_affinity: false,
        deployment_affinity: true,
      };
      const prefill = applyPresetTiers(
        buildEmptyPrefill().complexityRouterConfig,
        config,
        groupsOnly(["claude-sonnet-4.5"]),
      );
      expect(prefill.tiers.SIMPLE).toEqual(["claude-sonnet-4.5"]);
    });

    it("prefills the per-model litellm_params a preset carries in tier_model_configs", () => {
      const config = {
        tiers: { SIMPLE: ["gpt-5-nano"], MEDIUM: [], COMPLEX: [], REASONING: ["o3"] },
        tier_model_configs: {
          REASONING: [{ model_name: "o3", litellm_params: { reasoning_effort: "high" } }],
        },
        classifier_type: "heuristic" as const,
        classification_mode: "every_request" as const,
        session_affinity: false,
        deployment_affinity: true,
      };
      const prefill = applyPresetTiers(
        buildEmptyPrefill().complexityRouterConfig,
        config,
        groupsOnly(["gpt-5-nano", "o3"]),
      );
      expect(prefill.tier_model_params).toEqual({
        REASONING: { o3: { reasoning_effort: "high" } },
      });
    });

    // The params key on the preset's own spelling while the tier entry gets rewritten to the
    // caller's. Leaving the key alone names a model the tier no longer holds, and
    // serializeTierModelConfigs then drops the params on submit without saying so.
    it("rewrites a param key to the same registered spelling its tier entry was rewritten to", () => {
      const config = {
        tiers: { SIMPLE: [], MEDIUM: [], COMPLEX: [], REASONING: ["claude-sonnet-4-5"] },
        tier_model_configs: {
          REASONING: [{ model_name: "claude-sonnet-4-5", litellm_params: { reasoning_effort: "high" } }],
        },
        classifier_type: "heuristic" as const,
        classification_mode: "every_request" as const,
        session_affinity: false,
        deployment_affinity: true,
      };
      const prefill = applyPresetTiers(
        buildEmptyPrefill().complexityRouterConfig,
        config,
        groupsOnly(["claude-sonnet-4.5"]),
      );
      expect(prefill.tier_model_params).toEqual({
        REASONING: { "claude-sonnet-4.5": { reasoning_effort: "high" } },
      });
    });

    // Two spellings of one model in a tier collapse to a single registered key, and one model can
    // only hold one param set downstream. Merging keeps whatever only one spelling set instead of
    // dropping that spelling's params wholesale.
    it("merges rather than drops params when two spellings resolve to the same registered model", () => {
      const config = {
        tiers: { SIMPLE: [], MEDIUM: [], COMPLEX: [], REASONING: ["claude-sonnet-4-5", "claude-sonnet-4.5"] },
        tier_model_configs: {
          REASONING: [
            { model_name: "claude-sonnet-4-5", litellm_params: { reasoning_effort: "high", temperature: 0.2 } },
            { model_name: "claude-sonnet-4.5", litellm_params: { reasoning_effort: "low" } },
          ],
        },
        classifier_type: "heuristic" as const,
        classification_mode: "every_request" as const,
        session_affinity: false,
        deployment_affinity: true,
      };
      const prefill = applyPresetTiers(
        buildEmptyPrefill().complexityRouterConfig,
        config,
        groupsOnly(["claude-sonnet-4.5"]),
      );
      // temperature survives from the spelling that would otherwise have been overwritten;
      // reasoning_effort, set by both, resolves last-wins.
      expect(prefill.tier_model_params).toEqual({
        REASONING: { "claude-sonnet-4.5": { reasoning_effort: "low", temperature: 0.2 } },
      });
    });

    it("clears old tier parameters for a preset that carries no per-model params", () => {
      const config = {
        tiers: { SIMPLE: ["gpt-5-nano"], MEDIUM: [], COMPLEX: [], REASONING: [] },
        classifier_type: "heuristic" as const,
        classification_mode: "every_request" as const,
        session_affinity: false,
        deployment_affinity: true,
      };
      const prefill = applyPresetTiers(buildEmptyPrefill().complexityRouterConfig, config, groupsOnly(["gpt-5-nano"]));
      expect(prefill.tier_model_params).toEqual({});
    });
  });
});
