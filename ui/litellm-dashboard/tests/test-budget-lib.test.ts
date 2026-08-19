import { describe, expect, it } from "vitest";
import { TIER_BUDGET_MS, countByTier, overBudgetTests, tierForTestFile } from "../scripts/test-budget-lib.mjs";

const report = (files: { name: string; tests: { title: string; duration: number }[] }[]) => ({
  testResults: files.map((f) => ({
    name: `/repo/${f.name}`,
    assertionResults: f.tests.map((t) => ({ fullName: t.title, title: t.title, duration: t.duration })),
  })),
});

describe("tierForTestFile", () => {
  it("routes a plain .test.ts to the node-only unit tier", () => {
    expect(tierForTestFile("src/utils/dataUtils.test.ts")).toBe("unit");
  });

  it("routes anything rendering JSX to the component tier", () => {
    expect(tierForTestFile("src/components/Teams.test.tsx")).toBe("component");
  });

  it("routes a hook test to the component tier even though it is a .test.ts", () => {
    expect(tierForTestFile("src/app/(dashboard)/hooks/keys/useKeys.test.ts")).toBe("component");
  });

  it("routes an integration file to the integration tier before the .tsx rule can claim it", () => {
    expect(tierForTestFile("src/components/add_pass_through.integration.test.tsx")).toBe("integration");
  });

  it("routes a use_* config test to the component tier whether or not it sits in a subfolder", () => {
    expect(tierForTestFile("src/app/(dashboard)/cost-tracking/_components/use_margin_config.test.ts")).toBe(
      "component",
    );
    expect(
      tierForTestFile("src/app/(dashboard)/cost-tracking/_components/pricing_calculator/use_margin_config.test.ts"),
    ).toBe("component");
  });
});

describe("overBudgetTests", () => {
  it("flags only the tests burning more than their tier budget", () => {
    const violations = overBudgetTests(
      report([
        {
          name: "src/a.test.tsx",
          tests: [
            { title: "fast", duration: 400 },
            { title: "slow", duration: 6000 },
          ],
        },
        { name: "src/b.test.ts", tests: [{ title: "unit slow", duration: 3000 }] },
      ]),
      { repoRoot: "/repo/" },
    );

    expect(violations.map((v) => v.name)).toEqual(["slow", "unit slow"]);
    expect(violations[0]).toMatchObject({ tier: "component", file: "src/a.test.tsx", durationMs: 6000 });
  });

  it("uses each tier's own budget, so the same duration passes in one tier and fails in another", () => {
    const sameDuration = [{ title: "t", duration: 2000 }];
    const violations = overBudgetTests(
      report([
        { name: "src/slow.test.tsx", tests: sameDuration },
        { name: "src/slow.integration.test.tsx", tests: sameDuration },
      ]),
      { repoRoot: "/repo/" },
    );

    expect(violations).toHaveLength(1);
    expect(violations[0].tier).toBe("component");
  });

  it("tightens with the slack factor rather than a fixed millisecond ceiling", () => {
    const oneTest = report([{ name: "src/a.test.tsx", tests: [{ title: "t", duration: 700 }] }]);

    expect(overBudgetTests(oneTest, { repoRoot: "/repo/", slackFactor: 1 })).toHaveLength(0);
    expect(overBudgetTests(oneTest, { repoRoot: "/repo/", slackFactor: 0.5 })).toHaveLength(1);
  });

  it("treats a test exactly at the budget as within it", () => {
    const atBudget = report([{ name: "src/a.test.tsx", tests: [{ title: "t", duration: TIER_BUDGET_MS.component }] }]);

    expect(overBudgetTests(atBudget, { repoRoot: "/repo/" })).toHaveLength(0);
  });
});

describe("countByTier", () => {
  it("counts violations per tier and omits tiers that had none", () => {
    expect(countByTier([{ tier: "component" }, { tier: "component" }, { tier: "unit" }])).toEqual({
      component: 2,
      unit: 1,
    });
  });
});
