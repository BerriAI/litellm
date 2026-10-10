# PR caveats and merge dependencies

Classify by what happens if the caveat is ignored, not by the size of the diff, the effort to fix it, or whether the change is intentional. A one-line docs fix can still block a merge if CI depends on it

## Merge blockers and dependencies

Put unresolved merge prerequisites in `## Merge blockers and dependencies` before the TLDR, not only in Caveats or Relevant issues. Do not merge until the prerequisites are verified as satisfied

For each prerequisite, record:

- The exact PR URL or required action, including the repository for cross-repository dependencies
- What breaks without it, such as CI, a docs build, runtime behavior, or deployment
- The required merge or rollout order and who owns the next action
- Its verified status: pending, merged, deployed, or unknown, with supporting evidence. Unknown means unresolved; merged does not mean deployed

If duplicate or replacement PRs exist, identify the canonical PR and explicitly link which PR it supersedes. Verify that the required change landed on the intended branch; matching titles, identical-looking diffs, or a closed PR are not proof that the dependency is satisfied. If an equivalent duplicate landed instead, verify its contents and target branch and update the dependency link to that PR

Keep this section current as dependencies change. Remove it only when all prerequisites are verified as satisfied, or when there are none. Keep any remaining rollout risks in Caveats

Example: a code PR requires a companion docs PR to keep the docs build passing. This is a merge blocker until the required docs change is verified on the branch the build uses, even if the docs patch is one line

## Caveat severity

Put `## Caveats (if any)` immediately after the TLDR so reviewers see remaining risks before the detailed description. Group short bullets under the following headings, highest severity first, and omit empty tiers

| Tier | Classification | Examples / required action |
| --- | --- | --- |
| Severe | A change can take down or substantially degrade a deployment, rewrite or lose data, break an existing workflow, or change authentication or authorization behavior | A table-locking boot migration or an intentional auth change needs an explicit rollout plan before shipping |
| High | A correctness, security, data-loss, or backward-compatibility defect makes the change unsafe to ship, even with a narrow blast radius | An auth bypass for one route or incorrect results for one supported configuration must be fixed before shipping |
| Medium | A bounded, non-security limitation with a verified safe workaround, where merging independently keeps CI and supported deployments working | An optional feature has a documented limitation and a tested alternative |
| Low | Informational or cosmetic follow-up with no effect on correctness, security, CI, compatibility, or rollout order | Naming cleanup or optional explanatory docs |

Use the highest applicable tier. Intentional changes are not automatically safer, and rarity alone does not make an issue Low. Security and data-loss risks never become Medium or Low merely because they affect few users or have a workaround

Merge readiness is separate from severity: any unresolved prerequisite or unsafe-to-ship defect also belongs in the top merge-blocker section. Required companion PRs, migrations, configuration changes, and release coordination must never be buried under Low

For each caveat, state the affected behavior, consequence, and mitigation or follow-up. Link follow-up work when it exists. If a claim is assumed rather than tested, identify the missing verification and classify by what breaks if the assumption is wrong; do not describe an unverified workaround as safe
