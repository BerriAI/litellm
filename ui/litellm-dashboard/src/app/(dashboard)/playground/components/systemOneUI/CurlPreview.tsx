import CopyButton from "@/components/shared/CopyButton";
import { buildCurlCommand, type CurlCommandInput } from "./lib/curlCommand";

interface CurlPreviewProps extends CurlCommandInput {
  errorCount?: number;
}

export default function CurlPreview({ errorCount = 0, ...input }: CurlPreviewProps) {
  const command = buildCurlCommand(input);
  const invalid = errorCount > 0;
  return (
    <div className="flex min-h-0 flex-1 flex-col rounded-md border">
      <div className="flex items-center justify-between gap-2 border-b px-3 py-2 text-sm">
        <span className="font-medium">cURL request</span>
        <div className="flex items-center gap-2 text-xs text-muted-foreground">
          {invalid ? (
            <span role="alert" className="text-destructive">
              Fix {errorCount} {errorCount === 1 ? "issue" : "issues"} on the Form tab before copying
            </span>
          ) : (
            <>
              <span>Replace $LITELLM_API_KEY with a virtual key</span>
              <CopyButton value={command} label="Copy cURL" variant="action" />
            </>
          )}
        </div>
      </div>
      <pre
        aria-label="cURL request"
        className="m-0 min-h-0 flex-1 overflow-auto whitespace-pre-wrap wrap-anywhere px-3 py-3 font-mono text-xs leading-5"
      >
        {command}
      </pre>
    </div>
  );
}
