import { builderVerdictDotClass, type BuilderVerdict } from "./builderInsightsData";

export function BuilderVerdictBadge({ verdict, label }: { verdict: BuilderVerdict; label: string }) {
  return (
    <span className="inline-flex items-center gap-1.5 text-xs text-muted-foreground">
      <span aria-hidden="true" className={`size-1.5 rounded-full ${builderVerdictDotClass(verdict)}`} />
      {label}
    </span>
  );
}
