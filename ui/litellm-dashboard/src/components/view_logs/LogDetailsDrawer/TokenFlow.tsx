interface TokenFlowProps {
  prompt?: number;
  completion?: number;
  total?: number;
}

/**
 * Displays token usage in LiteLLM format: "12 (9 prompt tokens + 3 completion tokens)"
 */
export function TokenFlow({
  prompt = 0,
  completion = 0,
  total = 0,
}: TokenFlowProps) {
  return (
    <span>
      {total.toLocaleString()} ({prompt.toLocaleString()} prompt tokens +{" "}
      {completion.toLocaleString()} completion tokens)
    </span>
  );
}
