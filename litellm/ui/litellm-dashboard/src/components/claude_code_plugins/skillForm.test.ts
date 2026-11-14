import { describe, expect, it } from "vitest";
import { buildSkillImport, buildSkillMetadata, skillFormValues } from "./skillForm";
import type { SkillHubSkill } from "./types";

const skill: SkillHubSkill = {
  id: "skill-1",
  name: "shared-skill",
  version: "2.7.4",
  description: "Published instructions",
  source: { source: "github", repo: "example/skills", ref: "release", sha: "pinned-commit" },
  keywords: ["workflow", "tools"],
  domain: "Engineering",
  namespace: "workflows",
  visibility: "public",
  updated_at: "2026-10-20T00:00:00Z",
};

describe("skill form payloads", () => {
  it.each([["sales, finance", "review"], ["  workflow  ", "review", "review"], [], null])(
    "preserves unchanged keywords in a complete metadata replacement: %j",
    (keywords) => {
      const published = { ...skill, keywords };
      const values = skillFormValues(published);
      expect(values.version).toBe("");
      const expected = {
        description: "Updated instructions",
        keywords,
        domain: skill.domain,
        namespace: skill.namespace,
        version: null,
        expected_updated_at: skill.updated_at,
      };
      expect(buildSkillMetadata({ ...values, description: "Updated instructions" }, published)).toEqual(expected);
    },
  );

  it("parses intentional edits and clears blank metadata without changing the source", () => {
    const values = {
      ...skillFormValues(skill),
      domain: "",
      namespace: "",
      description: "",
      keywords: "security, review",
      version: "3.0.0",
    };
    const expected = {
      description: null,
      domain: null,
      namespace: null,
      keywords: ["security", "review"],
      version: "3.0.0",
      expected_updated_at: skill.updated_at,
    };
    expect(buildSkillMetadata(values, skill)).toEqual(expected);
    expect(buildSkillMetadata({ ...values, keywords: "" }, skill).keywords).toEqual([]);
  });

  it("leaves source URL parsing to the API", () => {
    const values = { ...skillFormValues(), url: "https://github.com/example/skills/tree/release/v2/review" };
    expect(buildSkillImport(values)).toEqual({ url: values.url });
  });
});
