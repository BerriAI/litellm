import { describe, expect, it } from "vitest";
import {
  buildSkillImport,
  buildSkillMetadata,
  buildSkillRegistration,
  buildSkillUpdate,
  buildSkillUploadMetadata,
  skillFormSchema,
  skillFormValues,
} from "./skillForm";
import type { SkillHubSkill } from "./types";
import { availableSkillUrlModes } from "./helpers";

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
  security_scan_passed: false,
  updated_at: "2026-10-20T00:00:00Z",
};

const metadata = {
  name: "registry-alias",
  description: "Registry description.".repeat(80),
  domain: "Engineering",
  namespace: "tools",
  keywords: ["review", "quality"],
  author: { name: "Maintainer", email: "maintainer@example.test" },
  category: "Productivity",
  homepage: "https://example.test/skill",
  version: "3.1.0",
};
const authored = { ...skill, ...metadata };
const archiveFile = new File(["legacy SKILL.md without frontmatter"], "legacy.zip");
const values = {
  ...skillFormValues(authored),
  mode: "register" as const,
  version: metadata.version,
  url: "https://gitlab.com/team/subgroup/repo/-/tree/release/v2/skills/review",
  ref: "release/v2",
  path: "previous-draft",
  skill: "unused-selector",
  archiveFile,
  sha256: "aBcD".repeat(16),
};
const automaticVersion = { version: null, expected_updated_at: skill.updated_at };

describe("skill form payloads", () => {
  it.each([["sales, finance", "review"], ["  workflow  ", "review", "review"], [], null])(
    "leaves stored keywords untouched during another metadata edit: %j",
    (keywords) => {
      const published = { ...skill, keywords, domain: " Engineering ", namespace: " workflows ", author: null };
      const values = skillFormValues(published);
      expect(values.version).toBe("");
      const expected = {
        description: metadata.description,
        version: null,
        expected_updated_at: skill.updated_at,
      };
      expect(buildSkillMetadata({ ...values, description: metadata.description }, published)).toEqual(expected);
    },
  );

  it.each(["", " \t "])("clears blank metadata %j without changing the source", (blank) => {
    const values = {
      ...skillFormValues(authored),
      domain: blank,
      namespace: blank,
      description: blank,
      authorName: blank,
      authorEmail: blank,
      category: blank,
      homepage: blank,
      keywords: "security, review",
      version: "3.0.0",
    };
    const expected = {
      description: null,
      domain: null,
      namespace: null,
      author: null,
      category: null,
      homepage: null,
      keywords: ["security", "review"],
      version: "3.0.0",
      expected_updated_at: skill.updated_at,
    };
    expect(buildSkillMetadata(skillFormSchema(authored).parse(values), authored)).toEqual(expected);
    expect(buildSkillMetadata({ ...values, keywords: blank }, authored).keywords).toEqual([]);
    expect(buildSkillMetadata({ ...skillFormValues(authored), authorEmail: blank }, authored)).toEqual({
      ...automaticVersion,
      author: { name: metadata.author.name, email: null },
    });
  });

  it("keeps complete creation payloads, long descriptions and normalized replacement sources", () => {
    expect(buildSkillUploadMetadata(skillFormSchema().parse({ ...values, mode: "upload" }))).toEqual(metadata);
    const imported = { ...values, mode: "import" as const, url: "https://github.com/example/repo" };
    const { url, skill, ref, path, version } = imported;
    const expectedImport = { url, skill, ref, path, version, metadata };
    expect(buildSkillImport(skillFormSchema().parse(imported))).toEqual(expectedImport);
    const source = {
      source: "git-subdir",
      url: "https://gitlab.com/team/subgroup/repo",
      path: "skills/review",
      ref: values.ref,
    };
    expect(buildSkillRegistration(skillFormSchema().parse(values))).toEqual({ ...metadata, source });
    const intent = { ...automaticVersion, version: values.version };
    const replacement = skillFormSchema(authored).parse({ ...values, mode: "replace" });
    expect(buildSkillUpdate(replacement, authored)).toEqual({ ...intent, source });
    expect(buildSkillUpdate({ ...replacement, mode: "metadata" }, authored)).toEqual(intent);
    const browserUrl = `${imported.url}/tree/release/v2/skills/review`;
    for (const mode of ["replace", "import"] as const) {
      const unresolved = { ...replacement, mode, url: browserUrl, ref: "" };
      expect(skillFormSchema(authored).safeParse(unresolved).error?.issues).toContainEqual(
        expect.objectContaining({ path: ["ref"] }),
      );
    }
    for (const revision of ["release%2Fv2", "a".repeat(40)]) {
      const unambiguous = { ...values, ref: "", url: values.url.replace("release/v2", revision) };
      expect(skillFormSchema().safeParse(unambiguous).success).toBe(true);
    }
    for (const [ref, path] of [
      ["release", "v2/skills/review"],
      ["release/v2", "skills/review"],
    ]) {
      const reference = skillFormSchema().parse({ ...values, ref });
      const snapshot = skillFormSchema().parse({ ...imported, url: browserUrl, ref });
      expect(buildSkillRegistration(reference)?.source).toMatchObject({ ref, path });
      expect(buildSkillImport(snapshot)).toMatchObject({ ref, path });
    }
    expect(
      skillFormValues({ ...authored, source: { source: "archive", url: "/skill-hub/skills/shared-skill/archive" } })
        .url,
    ).toBe("");
  });
});

describe("skill form validation", () => {
  it.each([
    "https://github.com:8443",
    "https://www.github.com:8443",
    "https://www.skills.sh:8443",
    "https://user@github.com",
    "http://github.com",
    "https://gitlab.com",
    "https://git.example.test",
  ])("offers only external references for %s", (origin) =>
    expect(availableSkillUrlModes(`${origin}/example/repo`)).toEqual(["register"]),
  );
  it.each(["github.com", "skills.sh"])("normalizes %s through validation and request serialization", (host) => {
    const url = `https://${host}/example/repo${host === "skills.sh" ? "/review" : ""}`;
    const expected = { url, ref: "main", path: "skills/review", ...(host === "github.com" ? { skill: "draft" } : {}) };
    for (const inputUrl of [
      url,
      ` ${url.replace(host, host.toUpperCase())} `,
      url.replace("https://", ""),
      url.replace(host, `${host}:443`),
      url.replace(host, `www.${host}`),
      url.replace(host, `www.${host}:443`),
    ]) {
      const draft = {
        ...skillFormValues(),
        mode: "import",
        url: inputUrl,
        skill: "draft",
        ref: "main",
        path: "skills/review",
      };
      expect(availableSkillUrlModes(inputUrl)).toContain("import");
      expect(buildSkillImport(skillFormSchema().parse(draft))).toEqual(expected);
    }
  });

  it.each([
    ["https://github.com/example/repo/tree/main/skills/review", "HEAD"],
    ["https://github.com/example/repo/blob/main/README.md", "main"],
    ["https://github.com/example/repo/tree/main/skills/review?ref=other", "main"],
    ["HTTPS://GitHub.com/example/repo/tree/main/skills/review", "other"],
    ["https://github.com:8443/example/repo", "main"],
    ["https://www.github.com:8443/example/repo", "main"],
    ["https://www.skills.sh:8443/example/repo/review", "main"],
    ["https://user@github.com/example/repo", "main"],
    ["http://github.com/example/repo", "main"],
    ["https://github.com/example/repo#section", "main"],
    ["https://skills.sh/example/repo", "main"],
    ["https://skills.sh/example/repo/review?ref=other", "main"],
    ["https://skills.sh/example/repo/%ZZ", "main"],
    ["https://gitlab.com/example/repo", "main"],
  ])("rejects conflicting or unsupported import selection %s", (url, ref) => {
    const input = { ...values, mode: "import" as const, url, ref };
    expect(skillFormSchema().safeParse(input).error?.issues).toContainEqual(expect.objectContaining({ path: ["url"] }));
    expect(buildSkillImport(input)).toBeNull();
  });

  it.each([
    ["", "skills/with spaces", "skills/with spaces"],
    ["/tree/main", "", undefined],
    ["/tree/main", "skills/review", "skills/review"],
    ["/blob/main/SKILL.md", "previous-draft", ""],
    ["/tree/main/skills/re%76iew", "previous-draft", "skills/review"],
    ["/blob/main/skills/with%20spaces/SKILL.md", "previous-draft", "skills/with spaces"],
  ])("preserves portable import directory rules for %s", (location, path, expected) => {
    const url = "https://github.com/example/repo";
    const draft = { ...values, mode: "import", url: url + location, ref: "main", path };
    const input = skillFormSchema().parse(draft);
    expect(buildSkillImport(input)).toMatchObject({ url, ref: "main" });
    expect(buildSkillImport(input)?.path).toBe(expected);
  });

  it.each(["register", "replace"] as const)("validates and preserves an optional archive digest during %s", (mode) => {
    const published = mode === "replace" ? authored : undefined;
    const input = { ...values, mode, url: "https://artifacts.example.test/skill.zip" };
    const schema = skillFormSchema(published);
    expect(schema.safeParse({ ...input, url: "not a repository" }).success).toBe(false);
    expect(schema.safeParse({ ...input, sha256: "" }).success).toBe(true);
    expect(schema.safeParse({ ...input, sha256: "invalid" }).error?.issues).toContainEqual(
      expect.objectContaining({ path: ["sha256"] }),
    );
    const validated = schema.parse({ ...input, sha256: ` ${values.sha256} ` });
    const payload = published ? buildSkillUpdate(validated, published) : buildSkillRegistration(validated);
    expect(payload?.source).toEqual({ source: "archive", url: input.url, sha256: values.sha256 });
  });

  it.each(["upload", "import", "register"] as const)("requires registry and author names for %s metadata", (mode) => {
    const input = { ...values, mode, url: "https://github.com/example/repo" };
    expect(skillFormSchema().safeParse({ ...input, name: "" }).success).toBe(false);
    expect(skillFormSchema().safeParse({ ...input, authorName: "" }).success).toBe(false);
    expect(skillFormSchema().safeParse(input).success).toBe(true);
  });
});
