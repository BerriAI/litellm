import type { FormEvent } from "react";
import { Button, Input, Tag } from "antd";
import { Check, Clock3, FileText, GitBranch, Search, ShieldCheck, ThumbsUp, X } from "lucide-react";
import { Eyebrow, Panel, Success } from "./SharedKnowledgePrimitives";

type JudgeStatus = "pending" | "accepted" | "rejected";
type Proposal = "supported" | "unsupported";

type SharedKnowledgeReviewProps = {
  publishedTitle: string;
  version: number;
  proposalBaseVersion: number;
  proposal: Proposal;
  judge: JudgeStatus;
  chooseProposal: (proposal: Proposal) => void;
  onReview: () => void;
  onOpenEvidence: () => void;
  query: string;
  onQueryChange: (query: string) => void;
  onSearch: (event: FormEvent<HTMLFormElement>) => void;
  hasMatch: boolean;
  loaded: boolean;
  outcome: boolean;
  loadedVersion: number;
  onLoad: () => void;
  onHelpfulOutcome: () => void;
};

const judgeViews = {
  accepted: {
    className: "border-emerald-500/30 bg-emerald-500/5",
    icon: Check,
    iconClassName: "text-emerald-600 dark:text-emerald-400",
    title: (version: number) => `Accepted · v${version} published`,
    description:
      "The query results support filtering before aggregation. The correction adds a verified child-account edge case.",
  },
  rejected: {
    className: "border-amber-500/30 bg-amber-500/5",
    icon: X,
    iconClassName: "text-amber-600 dark:text-amber-400",
    title: (version: number) => `Rejected · v${version} remains published`,
    description: "The proposed change contradicts the query evidence. No new version is published.",
  },
  pending: {
    className: "border-border bg-muted/40",
    icon: Clock3,
    iconClassName: "",
    title: () => "Awaiting review",
    description: "The published memo stays available while the proposed correction is checked.",
  },
};

export function SharedKnowledgeReview({
  publishedTitle,
  version,
  proposalBaseVersion,
  proposal,
  judge,
  chooseProposal,
  onReview,
  onOpenEvidence,
  query,
  onQueryChange,
  onSearch,
  hasMatch,
  loaded,
  outcome,
  loadedVersion,
  onLoad,
  onHelpfulOutcome,
}: SharedKnowledgeReviewProps) {
  const judgeView = judgeViews[judge];
  const JudgeIcon = judgeView.icon;
  return (
    <>
      <div className="grid gap-4 lg:grid-cols-[1.2fr_1fr]">
        <Panel
          title={
            <>
              <GitBranch className="size-4 text-muted-foreground" />
              Proposed correction
            </>
          }
          extra={
            <span className="font-mono text-xs text-muted-foreground">
              v{proposalBaseVersion} → v{proposalBaseVersion + 1}
            </span>
          }
        >
          <div className="space-y-4 p-4">
            <div>
              <h3 className="text-sm font-semibold tracking-tight">{publishedTitle}</h3>
              <p className="mt-1 text-xs text-muted-foreground">Proposed by Reconciliation agent · session b71c</p>
            </div>
            <div className="flex flex-wrap gap-2">
              <Button
                size="small"
                type={proposal === "supported" ? "primary" : "default"}
                disabled={version >= 2}
                onClick={() => chooseProposal("supported")}
              >
                Evidence-backed Update
              </Button>
              <Button
                size="small"
                type={proposal === "unsupported" ? "primary" : "default"}
                onClick={() => chooseProposal("unsupported")}
              >
                Try Unsupported Update
              </Button>
            </div>
            <div className="overflow-hidden rounded-lg border border-border text-sm leading-relaxed">
              <div className="border-b border-border bg-muted/50 p-3">
                <Eyebrow>Current · v{proposalBaseVersion}</Eyebrow>
                {proposalBaseVersion === 1
                  ? "Filter to active accounts when calculating the total."
                  : "Filter inactive parent and child accounts before aggregation."}
              </div>
              <div className={`p-3 ${proposal === "supported" ? "bg-emerald-500/5" : "bg-amber-500/5"}`}>
                <Eyebrow>Proposed · v{proposalBaseVersion + 1}</Eyebrow>
                {proposal === "supported" ? (
                  <>
                    Filter inactive accounts <strong>before aggregation, including child accounts</strong>. A
                    parent-only filter can still overcount.
                  </>
                ) : (
                  <>Remove the active-account filter. All accounts should always be included.</>
                )}
              </div>
            </div>
            <Button size="small" type="text" icon={<FileText className="size-3.5" />} onClick={onOpenEvidence}>
              Evidence: 3 Query Checks in Session b71c
            </Button>
            <div className="flex items-center gap-2 border-t border-border pt-3 text-xs text-muted-foreground">
              <Clock3 className="size-3.5" />
              Version history retained · {version > 1 ? `v1 → v${version} (current)` : "v1 remains current"}
            </div>
          </div>
        </Panel>
        <Panel
          title={
            <>
              <ShieldCheck className="size-4 text-muted-foreground" />
              Service-side LLM judge
            </>
          }
          extra={<Tag>Simulated</Tag>}
        >
          <div className="space-y-4 p-4">
            <p className="text-xs leading-relaxed text-muted-foreground">
              A small model checks the correction against the current memo and its source evidence.
            </p>
            <ul className="space-y-3 text-sm">
              <li className="flex items-center gap-2">
                <ShieldCheck className="size-4 text-muted-foreground" />
                Supported by the evidence
              </li>
              <li className="flex items-center gap-2">
                <ShieldCheck className="size-4 text-muted-foreground" />
                Consistent with the original scope
              </li>
              <li className="flex items-center gap-2">
                <ShieldCheck className="size-4 text-muted-foreground" />
                Specific enough to apply safely
              </li>
            </ul>
            <div aria-live="polite" className={`rounded-lg border p-3 ${judgeView.className}`}>
              <p className="flex items-center gap-2 text-sm font-medium">
                <JudgeIcon className={`size-4 ${judgeView.iconClassName}`} />
                {judgeView.title(version)}
              </p>
              <p className="mt-2 text-xs leading-relaxed text-muted-foreground">{judgeView.description}</p>
            </div>
            <Button
              type="primary"
              block
              disabled={judge !== "pending"}
              onClick={onReview}
              icon={<ShieldCheck className="size-4" />}
            >
              Run Judge Review
            </Button>
          </div>
        </Panel>
      </div>
      <Panel
        title={
          <>
            <Search className="size-4 text-muted-foreground" />
            The next agent finds the published memo
          </>
        }
        extra={<span className="text-[11px] text-muted-foreground">Postgres · vector + text search</span>}
      >
        <div className="space-y-3 p-4">
          <form onSubmit={onSearch} className="flex gap-2">
            <Input
              aria-label="Search shared knowledge"
              prefix={<Search className="size-3.5 text-muted-foreground" />}
              value={query}
              onChange={(e) => onQueryChange(e.target.value)}
            />
            <Button htmlType="submit">Search</Button>
          </form>
          {hasMatch ? (
            <div className="flex flex-wrap items-center justify-between gap-3 rounded-lg border border-border bg-muted/30 p-3">
              <div className="min-w-0">
                <p className="text-sm font-medium">{publishedTitle}</p>
                <p className="mt-1 text-xs text-muted-foreground">
                  v{version} · Semantic + keyword match · Account reporting
                </p>
              </div>
              <Button onClick={onLoad} disabled={loaded}>
                {loaded ? `v${version} Loaded` : `Load v${version} into Session`}
              </Button>
            </div>
          ) : (
            <div className="flex flex-col items-center gap-2 py-4 text-center">
              <Search className="size-6 text-muted-foreground" />
              <p className="text-sm font-medium">No matching demo memos</p>
              <p className="text-xs text-muted-foreground">Try “inactive accounts” or “reporting totals”.</p>
            </div>
          )}
          {loaded && (
            <div className="flex flex-wrap items-center justify-between gap-2" aria-live="polite">
              <Success>
                {outcome
                  ? "Helpful outcome recorded · linked to this memo version"
                  : `Finance assistant loaded v${loadedVersion} into a different user’s session`}
              </Success>
              <Button
                size="small"
                disabled={outcome}
                icon={<ThumbsUp className="size-3.5" />}
                onClick={onHelpfulOutcome}
              >
                {outcome ? "Outcome Recorded" : "Record Helpful Outcome"}
              </Button>
            </div>
          )}
        </div>
      </Panel>
    </>
  );
}
