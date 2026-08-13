import { describe, it, expect, beforeAll } from "vitest";
import { ESLint } from "eslint";

const FETCH_CODE = `export const load = async () => {\n  const res = await fetch("/api/thing");\n  return res.json();\n};\n`;

const RULE_ID = "no-restricted-syntax";

let eslint: ESLint;

const fetchMessages = async (filePath: string) => {
  const [result] = await eslint.lintText(FETCH_CODE, { filePath });
  return result.messages.filter((m) => m.ruleId === RULE_ID);
};

describe("location-based fetch() rule", () => {
  beforeAll(() => {
    eslint = new ESLint();
  });

  it("flags a raw fetch() in a normal source file", async () => {
    const messages = await fetchMessages("src/components/some_feature.tsx");
    expect(messages).toHaveLength(1);
    expect(messages[0].message).toMatch(/@\/lib\/http\/client/);
  });

  it("allows fetch() inside src/lib/http/ (the one place it lives)", async () => {
    const messages = await fetchMessages("src/lib/http/client.ts");
    expect(messages).toHaveLength(0);
  });
});
