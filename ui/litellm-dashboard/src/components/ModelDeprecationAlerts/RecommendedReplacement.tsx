export default function RecommendedReplacement({ model }: { model: string | null | undefined }) {
  return (
    <p className="mt-2 text-xs text-muted-foreground">
      Recommended new model: <span className="font-medium text-foreground">{model || "Not provided"}</span>
    </p>
  );
}
