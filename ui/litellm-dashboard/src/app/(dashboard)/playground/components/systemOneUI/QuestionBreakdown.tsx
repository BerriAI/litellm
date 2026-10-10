import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { cn } from "@/lib/cva.config";
import { ROOT_BLOCK_STYLES } from "./lib/rootBlocks";
import type { PlaygroundRequest } from "./lib/schemas";
import { requestView, type QuestionView } from "./lib/views";

function QuestionCriteria({ question }: { question: QuestionView }) {
  if (!question.criteria || question.criteria.length === 0) {
    return <p className="text-xs text-muted-foreground">No criteria defined</p>;
  }
  const rows = question.criteria.map((criterion) => (
    <div key={criterion.label} className="grid gap-0.5 sm:grid-cols-[minmax(7rem,auto)_1fr] sm:gap-3">
      <dt className={cn("text-xs font-medium", question.type !== "noul" && "font-mono")}>{criterion.label}</dt>
      <dd className="text-xs text-muted-foreground">{criterion.description}</dd>
    </div>
  ));
  return question.ordered ? (
    <ol className="grid gap-2">
      {question.criteria.map((criterion) => (
        <li key={criterion.label} className="grid gap-0.5 sm:grid-cols-[minmax(7rem,auto)_1fr] sm:gap-3">
          <span className="font-mono text-xs font-medium">{criterion.label}</span>
          <span className="text-xs text-muted-foreground">{criterion.description}</span>
        </li>
      ))}
    </ol>
  ) : (
    <dl className="grid gap-2">{rows}</dl>
  );
}

export default function QuestionBreakdown({ payload }: { payload?: PlaygroundRequest }) {
  if (!payload) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Question breakdown</CardTitle>
          <CardDescription>Enter a valid request to preview its input and questions.</CardDescription>
        </CardHeader>
      </Card>
    );
  }

  const view = requestView(payload);

  return (
    <Card>
      <CardHeader>
        <CardTitle>Question breakdown</CardTitle>
        <CardDescription>Review the {view.contextLabel.toLowerCase()} and criteria that will be sent.</CardDescription>
      </CardHeader>
      <CardContent className="grid gap-4 wrap-anywhere">
        <section
          aria-labelledby="decisions-context-heading"
          className={cn("grid gap-2 border-l-2 pl-3", ROOT_BLOCK_STYLES.state.accent)}
        >
          <h3 id="decisions-context-heading" className="text-sm font-medium">
            {view.contextLabel}
          </h3>
          <pre className="max-h-48 overflow-y-auto whitespace-pre-wrap rounded-md bg-muted p-3 font-mono text-xs">
            {view.context}
          </pre>
        </section>
        <div className={cn("grid gap-3 border-l-2 pl-3", ROOT_BLOCK_STYLES.questions.accent)}>
          {view.questions.map((question) => (
            <section
              key={question.id}
              className="grid gap-3 rounded-md border p-3"
              aria-labelledby={`question-${question.id}`}
            >
              <div className="flex flex-wrap items-center justify-between gap-2">
                <h3 id={`question-${question.id}`} className="font-mono text-sm font-medium">
                  {question.id}
                </h3>
                <Badge variant="secondary">{question.type}</Badge>
              </div>
              {question.instructions !== undefined && <p className="text-sm">{question.instructions}</p>}
              {question.criteria !== undefined && (
                <div className="grid gap-2">
                  <h4 className="text-xs font-medium text-muted-foreground">Criteria</h4>
                  <QuestionCriteria question={question} />
                </div>
              )}
            </section>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}
