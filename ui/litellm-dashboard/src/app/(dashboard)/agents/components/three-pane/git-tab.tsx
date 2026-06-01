"use client";

/**
 * GitTab — right pane Git view.
 *
 * Combines:
 *   - active Run's git.branches[]
 *   - live `git_commit` and `pr_opened` events (latest first)
 */
import { Empty, Tag, Typography, List } from "antd";
import type { CloudAgentRun, CloudAgentRunEvent, GitCommitPayload } from "@/types/cloud-agents";

const { Text, Title, Link: AntLink } = Typography;

interface GitTabProps {
  run: CloudAgentRun | null;
  events: CloudAgentRunEvent[];
}

interface DisplayCommit extends GitCommitPayload {
  seq: number;
  created_at: string;
}

function commitsFromEvents(events: CloudAgentRunEvent[]): DisplayCommit[] {
  return events
    .filter((e) => e.type === "git_commit")
    .map((e) => ({
      ...(e.payload as unknown as GitCommitPayload),
      seq: e.seq,
      created_at: e.created_at,
    }))
    .sort((a, b) => b.seq - a.seq);
}

function prUrlFromEvents(events: CloudAgentRunEvent[]): string | null {
  const latest = [...events].reverse().find((e) => e.type === "pr_opened");
  if (!latest) return null;
  const url = (latest.payload as { url?: unknown }).url;
  return typeof url === "string" ? url : null;
}

export default function GitTab({ run, events }: GitTabProps) {
  const commits = commitsFromEvents(events);
  const livePrUrl = prUrlFromEvents(events);
  const prUrl = livePrUrl ?? run?.git.pr_url ?? null;

  return (
    <div className="flex h-full flex-col gap-4 overflow-y-auto p-4" data-testid="git-tab">
      <section data-testid="git-branches">
        <Title level={5}>Branches</Title>
        {run && run.git.branches.length > 0 ? (
          <div className="flex flex-wrap gap-2">
            {run.git.branches.map((b) => (
              <Tag key={b.name} color="blue">
                {b.name} <Text type="secondary">← {b.base}</Text>
              </Tag>
            ))}
          </div>
        ) : (
          <Empty description="No branches yet" />
        )}
      </section>

      <section data-testid="git-pr">
        <Title level={5}>Pull request</Title>
        {prUrl ? (
          <AntLink href={prUrl} target="_blank" rel="noopener noreferrer" data-testid="git-pr-link">
            {prUrl}
          </AntLink>
        ) : (
          <Text type="secondary">No PR opened yet.</Text>
        )}
      </section>

      <section data-testid="git-commits">
        <Title level={5}>Commits</Title>
        {commits.length === 0 ? (
          <Empty description="No commits yet" />
        ) : (
          <List
            size="small"
            dataSource={commits}
            renderItem={(c) => (
              <List.Item data-testid={`git-commit-${c.sha}`}>
                <div className="flex w-full items-center gap-2">
                  <Tag color="default" className="!m-0 !font-mono !text-xs">
                    {c.sha.slice(0, 7)}
                  </Tag>
                  <Text className="!flex-1 !truncate">{c.message}</Text>
                  <Text type="secondary" className="!text-xs">
                    {c.branch}
                  </Text>
                </div>
              </List.Item>
            )}
          />
        )}
      </section>
    </div>
  );
}
