import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { cn } from "@/lib/cva.config";
import { ROOT_BLOCK_STYLES } from "./lib/rootBlocks";
import type { PlaygroundQuestion, PlaygroundRequest } from "./lib/schemas";

function formatState(state: unknown): string {
  if (typeof state === "string") {
    return state;
  }
  return JSON.stringify(state, null, 2) ?? String(state);
}

function QuestionCriteria({ question }: { question: PlaygroundQuestion }) {
  if (question.type === "choice") {
    return (
      <dl className="grid gap-2">
        {Object.entries(question.criteria).map(([label, description]) => (
          <div key={label} className="grid gap-0.5 sm:grid-cols-[minmax(7rem,auto)_1fr] sm:gap-3">
            <dt className="font-mono text-xs font-medium">{label}</dt>
            <dd className="text-xs text-muted-foreground">{formatState(description)}</dd>
          </div>
        ))}
      </dl>
    );
  }

  if (question.type === "noul") {
    const criteria = Object.entries(question.criteria ?? {}).filter(([, description]) => description !== undefined);
    return criteria.length > 0 ? (
      <dl className="grid gap-2">
        {criteria.map(([label, description]) => (
          <div key={label} className="grid gap-0.5 sm:grid-cols-[minmax(7rem,auto)_1fr] sm:gap-3">
            <dt className="text-xs font-medium">{label}</dt>
            <dd className="text-xs text-muted-foreground">
              {typeof description === "string" ? description : JSON.stringify(description)}
            </dd>
          </div>
        ))}
      </dl>
    ) : (
      <p className="text-xs text-muted-foreground">No criteria defined</p>
    );
  }

  return (
    <ol className="grid gap-2">
      {question.criteria.map((description, index) => (
        <li key={`${index}-${description}`} className="grid gap-0.5 sm:grid-cols-[minmax(7rem,auto)_1fr] sm:gap-3">
          <span className="font-mono text-xs font-medium">{index}</span>
          <span className="text-xs text-muted-foreground">{formatState(description)}</span>
        </li>
      ))}
    </ol>
  );
}

export default function QuestionBreakdown({ payload }: { payload?: PlaygroundRequest }) {
  if (!payload) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Question breakdown</CardTitle>
          <CardDescription>Enter a valid request to preview its state and questions.</CardDescription>
        </CardHeader>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>Question breakdown</CardTitle>
        <CardDescription>Review the state and criteria that will be sent.</CardDescription>
      </CardHeader>
      <CardContent className="grid gap-4 wrap-anywhere">
        <section
          aria-labelledby="system-one-state-heading"
          className={cn("grid gap-2 border-l-2 pl-3", ROOT_BLOCK_STYLES.state.accent)}
        >
          <h3 id="system-one-state-heading" className="text-sm font-medium">
            State
          </h3>
          <pre className="max-h-48 overflow-y-auto whitespace-pre-wrap rounded-md bg-muted p-3 font-mono text-xs">
            {formatState(payload.state)}
          </pre>
        </section>
        <div className={cn("grid gap-3 border-l-2 pl-3", ROOT_BLOCK_STYLES.questions.accent)}>
          {Object.entries(payload.questions).map(([id, question]) => (
            <section key={id} className="grid gap-3 rounded-md border p-3" aria-labelledby={`question-${id}`}>
              <div className="flex flex-wrap items-center justify-between gap-2">
                <h3 id={`question-${id}`} className="font-mono text-sm font-medium">
                  {id}
                </h3>
                <Badge variant="secondary">{question.type}</Badge>
              </div>
              {question.instructions != null && <p className="text-sm">{formatState(question.instructions)}</p>}
              <div className="grid gap-2">
                <h4 className="text-xs font-medium text-muted-foreground">Criteria</h4>
                <QuestionCriteria question={question} />
              </div>
            </section>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}
