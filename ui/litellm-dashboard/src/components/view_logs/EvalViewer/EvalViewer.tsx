import React from "react";
import { CircleCheck, CircleX, FlaskConical } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Card, CardAction, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Table, TableBody, TableCell, TableFooter, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";

interface EvalVerdict {
  criterion_name: string;
  score: number;
  reasoning: string;
  passed: boolean;
  weight?: number;
}

interface EvalInformation {
  eval_id?: string;
  eval_name: string;
  overall_score: number;
  passed: boolean;
  judge_model: string;
  iteration?: number;
  eval_error?: string | null;
  verdicts?: EvalVerdict[];
  threshold?: number;
}

interface EvalViewerProps {
  data: EvalInformation | EvalInformation[];
}

export default function EvalViewer({ data }: EvalViewerProps) {
  const entries: EvalInformation[] = Array.isArray(data) ? data : [data];

  if (!entries.length) return null;

  return (
    <div className="mb-6">
      <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 12 }}>
        <FlaskConical className="size-4" style={{ color: "#6366f1" }} />
        <span className="font-semibold" style={{ fontSize: 15 }}>
          LLM Judge Results
        </span>
      </div>

      {entries.map((entry, idx) => (
        <EvalEntryCard key={entry.eval_id || idx} entry={entry} />
      ))}
    </div>
  );
}

function EvalEntryCard({ entry }: { entry: EvalInformation }) {
  const passed = entry.passed;
  const scoreColor = passed ? "#52c41a" : "#ff4d4f";

  // Filter out synthetic "Overall" row the judge sometimes appends — it's already in the header
  const verdicts = (entry.verdicts || []).filter((v) => (v.criterion_name || "").toLowerCase() !== "overall");

  const hasWeights = verdicts.some((v) => v.weight != null);
  const weightedTotal = verdicts.reduce((sum, v) => sum + (v.weight != null ? (v.score * v.weight) / 100 : 0), 0);

  return (
    <Card size="sm" className="mb-3" style={{ borderLeft: `3px solid ${scoreColor}` }}>
      <CardHeader>
        <CardTitle>
          <div className="flex flex-wrap items-center gap-2">
            {passed ? (
              <CircleCheck className="size-4" style={{ color: "#52c41a" }} />
            ) : (
              <CircleX className="size-4" style={{ color: "#ff4d4f" }} />
            )}
            <span className="font-semibold">{entry.eval_name}</span>
            <Badge variant={passed ? "secondary" : "destructive"}>{passed ? "PASSED" : "FAILED"}</Badge>
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger
                  render={
                    <span
                      className="text-muted-foreground"
                      style={{ fontSize: 12, cursor: "help", borderBottom: "1px dashed #aaa" }}
                    />
                  }
                >
                  {entry.overall_score?.toFixed(0)} / 100
                  {entry.threshold != null && ` (threshold: ${entry.threshold})`}
                </TooltipTrigger>
                <TooltipContent>
                  Weighted average of all criterion scores. Each criterion has a weight (%) set when the eval was
                  created — higher-weight criteria count more toward the final score.
                </TooltipContent>
              </Tooltip>
            </TooltipProvider>
          </div>
        </CardTitle>
        <CardAction>
          <div className="flex items-center gap-2">
            {entry.judge_model && (
              <span className="text-muted-foreground" style={{ fontSize: 12 }}>
                Judge: {entry.judge_model}
              </span>
            )}
            {entry.iteration != null && (
              <span className="text-muted-foreground" style={{ fontSize: 12 }}>
                Iter: {entry.iteration + 1}
              </span>
            )}
          </div>
        </CardAction>
      </CardHeader>

      <CardContent>
        {entry.eval_error && (
          <span className="text-warning" style={{ display: "block", marginBottom: 8, fontSize: 12 }}>
            Judge error: {entry.eval_error}
          </span>
        )}

        {verdicts.length > 0 ? (
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead style={{ width: 160 }}>Criterion</TableHead>
                <TableHead style={{ width: 65 }}>Weight</TableHead>
                <TableHead style={{ width: 65 }}>Score</TableHead>
                <TableHead style={{ width: 75 }}>
                  <TooltipProvider>
                    <Tooltip>
                      <TooltipTrigger render={<span style={{ borderBottom: "1px dashed #aaa", cursor: "help" }} />}>
                        Weighted
                      </TooltipTrigger>
                      <TooltipContent>
                        Score × Weight — how much each criterion contributes to the final score
                      </TooltipContent>
                    </Tooltip>
                  </TooltipProvider>
                </TableHead>
                <TableHead>Comment</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {verdicts.map((row) => {
                const contrib = row.weight != null ? (row.score * row.weight) / 100 : null;
                return (
                  <TableRow key={row.criterion_name}>
                    <TableCell>
                      <span className="font-semibold" style={{ whiteSpace: "nowrap" }}>
                        {row.criterion_name}
                      </span>
                    </TableCell>
                    <TableCell>
                      {row.weight != null ? (
                        <span className="text-muted-foreground" style={{ fontSize: 12 }}>
                          {row.weight}%
                        </span>
                      ) : null}
                    </TableCell>
                    <TableCell>
                      <span
                        style={{
                          color: row.score >= 70 ? "#52c41a" : row.score >= 50 ? "#faad14" : "#ff4d4f",
                          fontWeight: 600,
                        }}
                      >
                        {row.score}
                      </span>
                    </TableCell>
                    <TableCell>
                      {contrib != null ? (
                        <span className="text-muted-foreground" style={{ fontSize: 12 }}>
                          {contrib % 1 === 0 ? contrib : contrib.toFixed(1)}
                        </span>
                      ) : null}
                    </TableCell>
                    <TableCell>
                      <TooltipProvider>
                        <Tooltip>
                          <TooltipTrigger render={<span style={{ fontSize: 12 }} />}>{row.reasoning}</TooltipTrigger>
                          <TooltipContent>{row.reasoning}</TooltipContent>
                        </Tooltip>
                      </TooltipProvider>
                    </TableCell>
                  </TableRow>
                );
              })}
            </TableBody>
            {hasWeights && (
              <TableFooter>
                <TableRow>
                  <TableCell>
                    <span className="font-semibold" style={{ fontSize: 12 }}>
                      Total
                    </span>
                  </TableCell>
                  <TableCell />
                  <TableCell />
                  <TableCell>
                    <span className="font-semibold" style={{ fontSize: 12, color: scoreColor }}>
                      {weightedTotal % 1 === 0 ? weightedTotal : weightedTotal.toFixed(1)}
                    </span>
                  </TableCell>
                  <TableCell />
                </TableRow>
              </TableFooter>
            )}
          </Table>
        ) : (
          <span className="text-muted-foreground" style={{ fontSize: 12 }}>
            Score: {entry.overall_score?.toFixed(1)} — no per-criterion breakdown available.
          </span>
        )}
      </CardContent>
    </Card>
  );
}
