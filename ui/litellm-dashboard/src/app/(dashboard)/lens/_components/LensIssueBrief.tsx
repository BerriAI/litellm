import type { Finding } from "./lensData";

export type IssueBrief = NonNullable<Finding["brief"]>;

export function LensIssueBrief({ brief }: { brief: IssueBrief }) {
  return (
    <div className="space-y-4 border-y py-4 text-sm leading-6">
      <div>
        <p className="font-medium">Problem</p>
        <p className="mt-1">{brief.problem}</p>
      </div>
      <div>
        <p className="font-medium">User goal</p>
        <p className="mt-1">{brief.user_goal}</p>
      </div>
      <div>
        <p className="font-medium">What happened</p>
        <p className="mt-1 whitespace-pre-wrap">{brief.what_happened}</p>
      </div>
      <div>
        <p className="font-medium">Test cases</p>
        <p className="text-xs text-muted-foreground">Run these inputs to confirm the agent behaves as expected.</p>
        <ol className="mt-2 space-y-2">
          {brief.test_cases.map((t) => (
            <li key={t.input} className="rounded-md bg-muted/40 p-3">
              <p>
                <span className="font-medium">Input: </span>
                {t.input}
              </p>
              <p className="mt-1">
                <span className="font-medium">Expect: </span>
                {t.expected}
              </p>
            </li>
          ))}
        </ol>
      </div>
    </div>
  );
}
