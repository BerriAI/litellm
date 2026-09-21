import { describe, expect, test } from "bun:test";

import type { GitHubApi } from "./auto-close-duplicates";
import {
  BODY_CAP_CHARS,
  BUG_SECTIONS,
  EDIT_WINDOW_MS,
  FORM_HEADINGS,
  SECTION_CAP_CHARS,
  FEATURE_SECTIONS,
  MIN_SECTION_CHARS,
  buildRequest,
  classifyIssue,
  gate,
  parseClassification,
  readConfig,
  routesOf,
  sections,
  shouldReclassify,
  userMessage,
  type ChatRequest,
  type IssueForClassification,
  type LlmClient,
  type Schema,
} from "./classify-issue";
import { MANIFEST, NAMESPACES } from "./issue-labels";
import schemaJson from "../.github/prompts/issue-classifier.schema.json";

const schema = schemaJson as Schema;
const routes = routesOf(schema);

const section = (heading: string, text: string): string => `### ${heading}\n\n${text}\n\n`;

const bugBody = (overrides: Partial<Record<(typeof BUG_SECTIONS)[number] | "dropdown" | "deploy", string>> = {}): string =>
  [
    section("Description", overrides.Description ?? "Streaming responses from Bedrock drop the last chunk when tools are used."),
    section("Config", overrides.Config ?? "```yaml\nmodel_list:\n  - model_name: claude\n    litellm_params:\n      model: bedrock/claude\n```"),
    section("LiteLLM Version", overrides["LiteLLM Version"] ?? "v1.100.0"),
    section("Steps to Repro", overrides["Steps to Repro"] ?? "1. curl -X POST http://localhost:4000/v1/chat/completions -d '{...}'\n2. Response: 500"),
    section("Which part of LiteLLM is this about?", overrides.dropdown ?? "LLM translation: a specific provider's request or response"),
    section("How are you deploying?", overrides.deploy ?? "_No response_"),
  ].join("");

const featureBody = (): string =>
  [
    section("Check for existing issues", "- [X] I have searched the existing issues and checked that my issue is not a duplicate."),
    section("The Feature", "Scope guardrail policies to specific MCP servers so one server is masked and another is not."),
    section("User Flow", "Before this feature (today): the admin attaches the policy globally and both servers get masked."),
    section("How far you got", "Config / setup the proxy ran with: two MCP servers and a Presidio guardrail; both calls come back raw."),
    section("Which part of LiteLLM is this about?", "Guardrails: moderation, PII masking, policies"),
  ].join("");

const issue = (overrides: Partial<IssueForClassification> = {}): IssueForClassification => ({
  number: 41700,
  title: "[Bug]: Bedrock streaming drops the last chunk with tools",
  body: bugBody(),
  author_association: "NONE",
  labels: [],
  created_at: "2026-09-17T12:00:00Z",
  ...overrides,
});

const label = (...names: readonly string[]): readonly { readonly name: string }[] => names.map((name) => ({ name }));

const modelAnswer = (overrides: Record<string, unknown> = {}): string =>
  JSON.stringify({
    domain: "llm-translation",
    provider: "bedrock",
    kind: "bug",
    priority: "p1",
    lift: "medium",
    route: "chat_completions",
    version: "v1.100.0",
    needs_repro: false,
    reason: "Bedrock streaming with tools drops the final chunk and no param avoids it.",
    ...overrides,
  });

describe("the schema and the manifest agree", () => {
  test("every labelled enum in the schema is exactly the manifest's values", () => {
    for (const namespace of NAMESPACES.filter((name) => name !== "needs")) {
      const allowed = (schema.properties[namespace]?.enum ?? []).filter((value) => value !== null);
      expect(new Set(allowed)).toEqual(new Set(Object.keys(MANIFEST[namespace])));
    }
  });

  test("provider and route accept null, the labelled-exactly-once fields do not", () => {
    expect(schema.properties.provider?.enum).toContain(null);
    expect(schema.properties.route?.enum).toContain(null);
    for (const field of ["domain", "kind", "priority", "lift"]) {
      expect(schema.properties[field]?.enum).not.toContain(null);
    }
  });

  test("every label description fits GitHub's 100 character limit", () => {
    for (const namespace of NAMESPACES) {
      for (const [value, spec] of Object.entries(MANIFEST[namespace])) {
        expect(spec.description.length, `${namespace}:${value}`).toBeLessThanOrEqual(100);
        expect(spec.color).toMatch(/^[0-9A-Fa-f]{6}$/);
      }
    }
  });
});

describe("sections", () => {
  test("splits an issue form body on its field headings and trims each block", () => {
    const found = sections("preamble\n### Description\n\nIt broke.\n\n### Config\n\n_No response_\n");
    expect([...found.entries()]).toEqual([
      ["Description", "It broke."],
      ["Config", "_No response_"],
    ]);
  });

  test("a heading the reporter typed inside a field stays inside that field", () => {
    const found = sections(
      "### Steps to Repro\n\n### Actual response\n\n500 from the proxy\n\n### Expected\n\n200\n\n### LiteLLM Version\n\nv1.100.0\n",
    );
    expect(found.get("Steps to Repro")).toBe("### Actual response\n\n500 from the proxy\n\n### Expected\n\n200");
    expect(found.get("LiteLLM Version")).toBe("v1.100.0");
  });

  test("a repeated field heading does not overwrite the first value", () => {
    const found = sections("### Description\n\nreal text\n\n### Config\n\n### Description\n\nnot a field\n");
    expect(found.get("Description")).toBe("real text");
    expect(found.get("Config")).toBe("### Description\n\nnot a field");
  });

  test("a body with no headings has no sections", () => {
    expect(sections("just some prose with ### inside a line").size).toBe(0);
    expect(sections("### Open question for OWNER\n\nnot a form field").size).toBe(0);
  });

  test("the known headings are exactly the field labels of the two issue forms", async () => {
    const labels = await Promise.all(
      ["bug_report.yml", "feature_request.yml"].map(async (file) => {
        const form = Bun.YAML.parse(await Bun.file(`${import.meta.dir}/../.github/ISSUE_TEMPLATE/${file}`).text()) as {
          readonly body: readonly { readonly attributes?: { readonly label?: string } }[];
        };
        return form.body.flatMap((field) => (field.attributes?.label === undefined ? [] : [field.attributes.label.trim()]));
      }),
    );
    expect(new Set(labels.flat())).toEqual(new Set(FORM_HEADINGS));
  });
});

describe("gate", () => {
  test("a filled bug template passes with the dropdown hint and the version", () => {
    expect(gate(issue())).toEqual({
      kind: "pass",
      template: "bug",
      domainHint: "LLM translation: a specific provider's request or response",
      version: "v1.100.0",
    });
  });

  test("a filled feature template passes as a feature", () => {
    expect(gate(issue({ title: "[Feature]: scope guardrails", body: featureBody() }))).toMatchObject({
      kind: "pass",
      template: "feature",
      domainHint: "Guardrails: moderation, PII masking, policies",
      version: null,
    });
  });

  test("an empty, placeholder, or too-short section is missing", () => {
    expect(gate(issue({ body: bugBody({ Config: "_No response_" }) }))).toEqual({
      kind: "template",
      template: "bug",
      missing: ["Config"],
    });
    expect(gate(issue({ body: bugBody({ "Steps to Repro": "n/a" }) }))).toMatchObject({ missing: ["Steps to Repro"] });
    expect(gate(issue({ body: bugBody({ Description: "x".repeat(MIN_SECTION_CHARS - 1) }) }))).toMatchObject({
      missing: ["Description"],
    });
    expect(gate(issue({ body: bugBody({ Description: "x".repeat(MIN_SECTION_CHARS) }) })).kind).toBe("pass");
  });

  test("a version has to carry a number", () => {
    expect(gate(issue({ body: bugBody({ "LiteLLM Version": "latest" }) }))).toMatchObject({ missing: ["LiteLLM Version"] });
    expect(gate(issue({ body: bugBody({ "LiteLLM Version": "main-v1.101.3-nightly" }) }))).toMatchObject({
      kind: "pass",
      version: "main-v1.101.3-nightly",
    });
  });

  test("an issue filed without the form is missing every required section of its template", () => {
    expect(gate(issue({ body: "It is broken, please fix." }))).toEqual({
      kind: "template",
      template: "bug",
      missing: [...BUG_SECTIONS],
    });
    expect(gate(issue({ title: "[Feature]: add a thing", body: null }))).toEqual({
      kind: "template",
      template: "feature",
      missing: [...FEATURE_SECTIONS],
    });
  });

  test("the title prefix names the template, and the headings decide only without one", () => {
    const oldBugShape = [section("What happened?", "Vertex AI rejects tools whose parameters use a top-level anyOf."), section("User Flow", "Before a fix: the request fails with a 400 from Vertex AI.")].join("");
    expect(gate(issue({ title: "[Bug]: Vertex AI 400 on anyOf tool schemas", body: oldBugShape }))).toEqual({
      kind: "template",
      template: "bug",
      missing: [...BUG_SECTIONS],
    });
    expect(gate(issue({ title: "Vertex AI 400 on anyOf tool schemas", body: oldBugShape }))).toMatchObject({
      template: "feature",
    });
    expect(gate(issue({ title: "[feature]: scope guardrails", body: bugBody() }))).toMatchObject({ template: "feature" });
  });

  test("a maintainer's issue passes the gate whatever its shape, so the bot never nags the team", () => {
    expect(gate(issue({ body: "internal note", author_association: "MEMBER" }))).toEqual({
      kind: "pass",
      template: "bug",
      domainHint: null,
      version: null,
    });
    expect(gate(issue({ body: "internal note", author_association: "CONTRIBUTOR" })).kind).toBe("template");
  });

  test("'Not sure' and an unanswered dropdown are no hint", () => {
    expect(gate(issue({ body: bugBody({ dropdown: "Not sure" }) }))).toMatchObject({ domainHint: null });
    expect(gate(issue({ body: bugBody({ dropdown: "_No response_" }) }))).toMatchObject({ domainHint: null });
  });
});

describe("buildRequest", () => {
  const passed = { kind: "pass" as const, template: "bug" as const, domainHint: "Caching: response cache", version: "v1.99.0" };

  test("asks for strict JSON against the vendored schema with the prompt as the system message", () => {
    const request = buildRequest("gpt-5.6-luna", "PROMPT", schema, issue(), passed);
    expect(request.model).toBe("gpt-5.6-luna");
    expect(request.messages[0]).toEqual({ role: "system", content: "PROMPT" });
    expect(request.messages[1]?.role).toBe("user");
    expect(request.response_format).toEqual({
      type: "json_schema",
      json_schema: { name: "issue_classification", strict: true, schema },
    });
    expect(Object.keys(request)).toEqual(["model", "messages", "response_format"]);
  });

  test("the user message carries the title, the template, the hint and the version above the body", () => {
    const message = userMessage(issue(), passed);
    expect(message.startsWith("Title: [Bug]: Bedrock streaming drops the last chunk with tools\nTemplate: bug\n")).toBe(true);
    expect(message).toContain("Reporter's pick from the domain dropdown: Caching: response cache");
    expect(message).toContain("LiteLLM Version (from the template): v1.99.0");
    expect(message).toContain("### Steps to Repro");
  });

  test("each field is capped on its own, so a huge config cannot push the repro out of the message", () => {
    const message = userMessage(issue({ body: bugBody({ Config: "y".repeat(SECTION_CAP_CHARS * 3) }) }), passed);
    expect(message).toContain(`[section truncated at ${SECTION_CAP_CHARS} characters]`);
    expect(message).toContain("### Steps to Repro\n\n1. curl -X POST http://localhost:4000/v1/chat/completions");
    expect(message.length).toBeLessThan(SECTION_CAP_CHARS + 1500);
  });

  test("the hiring, contact and duplicate-check fields are left out of the message", () => {
    const message = userMessage(issue({ title: "[Feature]: scope guardrails", body: featureBody() }), passed);
    expect(message).toContain("### The Feature");
    expect(message).not.toContain("Check for existing issues");
  });

  test("a body without form fields is sent whole, capped, and the version survives the cap", () => {
    const body = "x".repeat(BODY_CAP_CHARS * 2);
    const message = userMessage(issue({ body }), passed);
    expect(message.length).toBeLessThan(BODY_CAP_CHARS + 500);
    expect(message).toContain(`[body truncated at ${BODY_CAP_CHARS} characters]`);
    expect(message).toContain("LiteLLM Version (from the template): v1.99.0");
  });

  test("no hint and no version are said plainly", () => {
    const message = userMessage(issue({ body: null }), { ...passed, domainHint: null, version: null });
    expect(message).toContain("Reporter's pick from the domain dropdown: none\n");
    expect(message).not.toContain("LiteLLM Version (from the template)");
  });
});

describe("parseClassification", () => {
  test("accepts the schema's shape and turns it into labels plus needs", () => {
    const parsed = parseClassification(modelAnswer(), MANIFEST, routes);
    expect(parsed).toEqual({
      kind: "classification",
      classification: {
        gate: "pass",
        domain: "llm-translation",
        provider: "bedrock",
        kind: "bug",
        priority: "p1",
        lift: "medium",
        route: "chat_completions",
        version: "v1.100.0",
        needs: [],
        reason: "Bedrock streaming with tools drops the final chunk and no param avoids it.",
      },
    });
  });

  test("a null version needs version, a bug without a repro needs repro, both can stack", () => {
    const both = parseClassification(modelAnswer({ version: null, needs_repro: true }), MANIFEST, routes);
    expect(both.kind === "classification" && both.classification.needs).toEqual(["version", "repro"]);
    const none = parseClassification(modelAnswer({ provider: null, route: null }), MANIFEST, routes);
    expect(none.kind === "classification" && none.classification).toMatchObject({ provider: null, route: null, needs: [] });
  });

  test("kind decides first: a feature or question is p3 whatever the model said, and never needs a repro", () => {
    const feature = parseClassification(modelAnswer({ kind: "feature", priority: "p1", needs_repro: true }), MANIFEST, routes);
    expect(feature.kind === "classification" && feature.classification).toMatchObject({ priority: "p3", needs: [] });
    const question = parseClassification(modelAnswer({ kind: "question", priority: "p0" }), MANIFEST, routes);
    expect(question.kind === "classification" && question.classification.priority).toBe("p3");
  });

  test("a value the manifest does not know is rejected instead of half-applied", () => {
    expect(parseClassification(modelAnswer({ domain: "networking" }), MANIFEST, routes)).toMatchObject({ kind: "invalid" });
    expect(parseClassification(modelAnswer({ provider: "groq" }), MANIFEST, routes)).toMatchObject({ kind: "invalid" });
    expect(parseClassification(modelAnswer({ priority: "p4" }), MANIFEST, routes)).toMatchObject({ kind: "invalid" });
    expect(parseClassification(modelAnswer({ lift: "huge" }), MANIFEST, routes)).toMatchObject({ kind: "invalid" });
    expect(parseClassification(modelAnswer({ route: "batch" }), MANIFEST, routes)).toMatchObject({ kind: "invalid" });
    expect(parseClassification(modelAnswer({ kind: "bugg" }), MANIFEST, routes)).toMatchObject({ kind: "invalid" });
  });

  test("a malformed answer is rejected", () => {
    expect(parseClassification("not json", MANIFEST, routes)).toMatchObject({ kind: "invalid" });
    expect(parseClassification("[]", MANIFEST, routes)).toMatchObject({ kind: "invalid" });
    expect(parseClassification(modelAnswer({ needs_repro: "yes" }), MANIFEST, routes)).toMatchObject({ kind: "invalid" });
    expect(parseClassification(modelAnswer({ reason: " " }), MANIFEST, routes)).toMatchObject({ kind: "invalid" });
    expect(parseClassification(modelAnswer({ version: "" }), MANIFEST, routes)).toMatchObject({ kind: "invalid" });
  });
});

describe("shouldReclassify", () => {
  const now = new Date("2026-09-17T12:10:00Z");

  test("an issue that already carries a domain label is left alone, whatever else it has", () => {
    expect(shouldReclassify(issue({ labels: label("domain:caching", "kind:bug") }), now)).toBe(false);
    expect(shouldReclassify(issue({ labels: label("needs:template", "domain:caching") }), now)).toBe(false);
  });

  test("a gated issue is re-run however old it is", () => {
    const old = new Date(Date.parse("2026-09-17T12:00:00Z") + EDIT_WINDOW_MS * 48);
    expect(shouldReclassify(issue({ labels: label("bug", "needs:template") }), old)).toBe(true);
  });

  test("an unlabelled issue is re-run inside the edit window and ignored after it", () => {
    expect(shouldReclassify(issue({ labels: label("bug") }), now)).toBe(true);
    const later = new Date(Date.parse("2026-09-17T12:00:00Z") + EDIT_WINDOW_MS);
    expect(shouldReclassify(issue({ labels: label("bug") }), later)).toBe(false);
  });
});

describe("classifyIssue", () => {
  const config = {
    repo: "BerriAI/litellm",
    issueNumber: 41700,
    model: "gpt-5.6-luna",
    action: "opened",
    now: new Date("2026-09-17T12:10:00Z"),
  };

  function fakeApi(fetched: IssueForClassification): GitHubApi {
    return {
      request: async <T>(method: string, path: string): Promise<T> => {
        if (method === "GET" && path === "/repos/BerriAI/litellm/issues/41700") {
          return fetched as T;
        }
        throw new Error(`unexpected ${method} ${path}`);
      },
    };
  }

  function fakeLlm(answer: string): { readonly llm: LlmClient; readonly requests: ChatRequest[] } {
    const requests: ChatRequest[] = [];
    return {
      requests,
      llm: {
        complete: async (request) => {
          requests.push(request);
          return answer;
        },
      },
    };
  }

  test("a gated issue never reaches the model", async () => {
    const { llm, requests } = fakeLlm(modelAnswer());
    const verdict = await classifyIssue(fakeApi(issue({ body: "no template" })), llm, config, "PROMPT", schema);
    expect(verdict).toEqual({ gate: "template", template: "bug", missing: [...BUG_SECTIONS] });
    expect(requests).toEqual([]);
  });

  test("an issue that passes the gate is classified by one call with the configured model", async () => {
    const { llm, requests } = fakeLlm(modelAnswer());
    const verdict = await classifyIssue(fakeApi(issue()), llm, config, "PROMPT", schema);
    expect(verdict).toMatchObject({ gate: "pass", domain: "llm-translation", provider: "bedrock", priority: "p1" });
    expect(requests).toHaveLength(1);
    expect(requests[0]?.model).toBe("gpt-5.6-luna");
    expect(requests[0]?.messages[0]?.content).toBe("PROMPT");
  });

  test("an answer the manifest does not know fails the run instead of returning a partial set", async () => {
    const { llm } = fakeLlm(modelAnswer({ domain: "made-up" }));
    await expect(classifyIssue(fakeApi(issue()), llm, config, "PROMPT", schema)).rejects.toThrow("failed validation");
  });

  test("a pull request number is refused", async () => {
    const { llm, requests } = fakeLlm(modelAnswer());
    await expect(classifyIssue(fakeApi(issue({ pull_request: {} })), llm, config, "PROMPT", schema)).rejects.toThrow(
      "is a pull request",
    );
    expect(requests).toEqual([]);
  });

  test("an edit to an issue that was classified while the edit was pending is ignored", async () => {
    const { llm, requests } = fakeLlm(modelAnswer());
    const edited = { ...config, action: "edited" };
    const labelled = issue({ labels: label("domain:llm-translation", "kind:bug", "priority:p1", "lift:small") });
    expect(await classifyIssue(fakeApi(labelled), llm, edited, "PROMPT", schema)).toBeNull();
    expect(requests).toEqual([]);
  });

  test("an edit that fixes a gated issue is classified against the new body", async () => {
    const { llm, requests } = fakeLlm(modelAnswer());
    const edited = { ...config, action: "edited" };
    const verdict = await classifyIssue(fakeApi(issue({ labels: label("bug", "needs:template") })), llm, edited, "PROMPT", schema);
    expect(verdict).toMatchObject({ gate: "pass", domain: "llm-translation" });
    expect(requests).toHaveLength(1);
  });

  test("an edit during the first run, before any label landed, is classified instead of dropped", async () => {
    const { llm, requests } = fakeLlm(modelAnswer());
    const edited = { ...config, action: "edited" };
    expect(await classifyIssue(fakeApi(issue({ labels: label("bug") })), llm, edited, "PROMPT", schema)).toMatchObject({
      gate: "pass",
    });
    expect(requests).toHaveLength(1);
  });

  test("a manual run classifies an old unlabelled issue that an edit would ignore", async () => {
    const { llm, requests } = fakeLlm(modelAnswer());
    const old = issue({ labels: label("bug"), created_at: "2020-01-01T00:00:00Z" });
    expect(await classifyIssue(fakeApi(old), llm, { ...config, action: "edited" }, "PROMPT", schema)).toBeNull();
    expect(await classifyIssue(fakeApi(old), llm, { ...config, action: "" }, "PROMPT", schema)).toMatchObject({ gate: "pass" });
    expect(requests).toHaveLength(1);
  });
});

describe("readConfig", () => {
  const env = {
    GITHUB_TOKEN: "t",
    GITHUB_REPOSITORY: "BerriAI/litellm",
    ISSUE_NUMBER: "41700",
    LITELLM_API_BASE: "https://llm.example.com",
    LITELLM_API_KEY: "sk-test",
    ISSUE_CLASSIFIER_MODEL: "gpt-5.6-luna",
  };

  const now = new Date("2026-09-17T12:10:00Z");

  test("reads the six settings, and the event action when the workflow passes one", () => {
    expect(readConfig(env, now)).toEqual({
      token: "t",
      repo: "BerriAI/litellm",
      issueNumber: 41700,
      apiBase: "https://llm.example.com",
      apiKey: "sk-test",
      model: "gpt-5.6-luna",
      action: "",
      now,
    });
    expect(readConfig({ ...env, GITHUB_EVENT_ACTION: "edited" }, now)).toMatchObject({ action: "edited" });
  });

  test("refuses a missing or malformed setting by name", () => {
    expect(() => readConfig({ ...env, GITHUB_TOKEN: undefined }, now)).toThrow("GITHUB_TOKEN");
    expect(() => readConfig({ ...env, GITHUB_REPOSITORY: "nope" }, now)).toThrow("GITHUB_REPOSITORY");
    expect(() => readConfig({ ...env, ISSUE_NUMBER: "0" }, now)).toThrow("ISSUE_NUMBER");
    expect(() => readConfig({ ...env, LITELLM_API_BASE: "" }, now)).toThrow("LITELLM_API_BASE");
    expect(() => readConfig({ ...env, LITELLM_API_BASE: "llm.example.com" }, now)).toThrow("LITELLM_API_BASE");
    expect(() => readConfig({ ...env, LITELLM_API_KEY: "" }, now)).toThrow("LITELLM_API_KEY");
    expect(() => readConfig({ ...env, ISSUE_CLASSIFIER_MODEL: undefined }, now)).toThrow("ISSUE_CLASSIFIER_MODEL");
  });
});
