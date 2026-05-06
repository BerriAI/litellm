"use client";

/**
 * FilesChangedAccordion — collapsible "N Files Changed" block in the
 * conversation pane. Aggregates file_diff events by path: the latest patch
 * wins, additions/deletions are summed per file.
 *
 * Cumulative across the entire run (per spec).
 */
import { Collapse, Tag, Typography } from "antd";
import type { FileDiffPayload } from "@/types/cloud-agents";

const { Paragraph, Text } = Typography;

interface FilesChangedAccordionProps {
  diffs: FileDiffPayload[];
}

interface AggregatedDiff {
  path: string;
  additions: number;
  deletions: number;
  patch: string;
}

function aggregate(diffs: FileDiffPayload[]): AggregatedDiff[] {
  const byPath = new Map<string, AggregatedDiff>();
  for (const d of diffs) {
    const existing = byPath.get(d.path);
    if (existing) {
      existing.additions += d.additions;
      existing.deletions += d.deletions;
      existing.patch = d.patch; // latest wins
    } else {
      byPath.set(d.path, { path: d.path, additions: d.additions, deletions: d.deletions, patch: d.patch });
    }
  }
  return Array.from(byPath.values());
}

export default function FilesChangedAccordion({ diffs }: FilesChangedAccordionProps) {
  const aggregated = aggregate(diffs);
  if (aggregated.length === 0) return null;

  return (
    <Collapse
      size="small"
      className="!mb-3"
      data-testid="files-changed-accordion"
      items={[
        {
          key: "files",
          label: (
            <Text strong className="!text-sm">
              {aggregated.length} {aggregated.length === 1 ? "File" : "Files"} Changed
            </Text>
          ),
          children: (
            <div className="space-y-2" data-testid="files-changed-list">
              {aggregated.map((d) => (
                <div key={d.path} className="flex items-center gap-2 text-xs" data-testid={`file-diff-${d.path}`}>
                  <Tag color="green" className="!m-0">
                    +{d.additions}
                  </Tag>
                  <Tag color="red" className="!m-0">
                    -{d.deletions}
                  </Tag>
                  <Text className="!font-mono">{d.path}</Text>
                </div>
              ))}
              {aggregated.map((d) => (
                <Paragraph
                  key={`${d.path}-patch`}
                  className="!mb-0 whitespace-pre-wrap !text-xs !font-mono !bg-gray-50 !p-2"
                >
                  {d.patch}
                </Paragraph>
              ))}
            </div>
          ),
        },
      ]}
    />
  );
}
