import type { ObservedPerson, ObservedPull, ObservedSnapshot, Period } from "./observedData";

const DAY = 86_400_000;
const engineers = [
  { name: "Alex Rivera", login: "alex-demo", weekly: [8, 6, 4] },
  { name: "Sam Chen", login: "sam-demo", weekly: [6, 5, 4] },
  { name: "Jordan Lee", login: "jordan-demo", weekly: [4, 4, 3] },
];
const titles = [
  "Add repository search",
  "Fix retry handling",
  "Speed up activity queries",
  "Add usage export",
  "Improve connection setup",
  "Fix pagination",
];
const isoDate = (timestamp: number) => new Date(timestamp).toISOString().slice(0, 10);

function median(pulls: ObservedPull[]) {
  const hours = pulls.map((pull) => pull.merge_hours ?? 0).toSorted((a, b) => a - b);
  const middle = Math.floor(hours.length / 2);
  if (!hours.length) return null;
  return hours.length % 2 ? hours[middle] : (hours[middle - 1] + hours[middle]) / 2;
}

function samplePeriod(days: number, end: number, comparison: number) {
  const start = end - (days - 1) * DAY;
  const people = engineers.map((engineer, index) => {
    const count = Math.round((engineer.weekly[comparison] * days) / 7);
    const pulls: ObservedPull[] = Array.from({ length: count }, (_, position) => {
      const number = 10000 * (comparison + 1) + index * 1000 + position;
      const repo = index === 1 ? "demo/api" : "demo/web";
      const branch = `feature/sample-${number}`;
      const agent = position % 5 === 0;
      return {
        connection_id: index === 1 ? "demo-gitlab" : "demo-github",
        number,
        title: titles[position % titles.length],
        url: `https://example.com/${repo}/changes/${number}`,
        author: agent ? "demo-agent" : engineer.login,
        agent,
        merged_at: new Date(start + Math.floor((position * days) / count) * DAY + 12 * 3_600_000).toISOString(),
        merge_hours: position === 0 ? 16 / 3600 : 4 + ((position * 7) % 24) + comparison * 6,
        repo,
        source_repo: repo,
        source_branch: branch,
        branch_cost: { repo, branch, spend: 2 + (position % 4), requests: 20 + position, status: "matched" },
      };
    });
    const spend = pulls.reduce((total, pull) => total + (pull.branch_cost.spend ?? 0), 0);
    const metrics: ObservedPerson["periods"]["current"] = {
      merged_prs: pulls.length,
      prs_per_week: (pulls.length * 7) / days,
      median_merge_hours: median(pulls),
      direct_authored: pulls.filter((pull) => !pull.agent).length,
      declared_agent_owned: pulls.filter((pull) => pull.agent).length,
      gateway_recorded_spend: spend,
      recorded_spend_per_attributed_pr: pulls.length ? spend / pulls.length : null,
      spend_observation: "records_present",
      pr_urls: pulls.map((pull) => pull.url),
    };
    return { metrics, pulls };
  });
  const pulls = people.flatMap((person) => person.pulls).toSorted((a, b) => b.merged_at.localeCompare(a.merged_at));
  const metrics: ObservedSnapshot["periods"]["current"] = {
    window: { start: isoDate(start), end: isoDate(end) },
    merged_prs: pulls.length,
    median_merge_hours: median(pulls),
    human_authored: pulls.filter((pull) => !pull.agent).length,
    agent_authored: pulls.filter((pull) => pull.agent).length,
    missing_author: 0,
    agents_without_requester: 0,
    matched_internal_prs: pulls.length,
    new_bug_labeled_issues: Math.round(((comparison + 1) * days) / 7),
    new_regression_labeled_issues: comparison,
    explicitly_titled_revert_prs: 0,
    matched_users_recorded_spend: people.reduce((total, person) => total + person.metrics.gateway_recorded_spend, 0),
    spend_observation: "records_present",
    human_summary: { median_merge_hours: median(pulls.filter((pull) => !pull.agent)) },
  };
  return { metrics, pulls, people };
}

export function createObservedDemo(days: number, now = new Date()): ObservedSnapshot {
  const end = Date.UTC(now.getUTCFullYear(), now.getUTCMonth(), now.getUTCDate()) - DAY;
  const yearAgo = new Date(end);
  const lastYear = yearAgo.getUTCFullYear() - 1;
  const month = yearAgo.getUTCMonth();
  const lastDay = new Date(Date.UTC(lastYear, month + 1, 0)).getUTCDate();
  const lastYearEnd = Date.UTC(lastYear, month, Math.min(yearAgo.getUTCDate(), lastDay));
  const periods = {
    current: samplePeriod(days, end, 0),
    previous: samplePeriod(days, end - days * DAY, 1),
    last_year: samplePeriod(days, lastYearEnd, 2),
  };
  const personPeriod = (period: Period, index: number) => periods[period].people[index].metrics;
  return {
    source_provider: "mixed",
    repos: ["demo/web", "demo/api"],
    unmatched_logins: [],
    unlinked_branches: [],
    captured_at: now.toISOString(),
    periods: {
      current: periods.current.metrics,
      previous: periods.previous.metrics,
      last_year: periods.last_year.metrics,
    },
    people: engineers.map((engineer, index) => ({
      name: engineer.name,
      email: `${engineer.login}@example.com`,
      logins: [engineer.login],
      periods: {
        current: personPeriod("current", index),
        previous: personPeriod("previous", index),
        last_year: personPeriod("last_year", index),
      },
    })),
    pulls: { current: periods.current.pulls, previous: periods.previous.pulls, last_year: periods.last_year.pulls },
  };
}
