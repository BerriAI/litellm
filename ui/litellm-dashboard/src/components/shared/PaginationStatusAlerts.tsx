import { ExternalLink, Loader2 } from "lucide-react";

import { Alert, AlertDescription } from "@/components/shared/Alert";
import { Button } from "@/components/ui/button";

interface PaginationStatusAlertsProps {
  isFetchingMore: boolean;
  cancelled: boolean;
  progress: { currentPage: number; totalPages: number };
  cancel: () => void;
  subject?: string;
  failed?: boolean;
}

const failureMessage = (subject: string, progress: { currentPage: number; totalPages: number }) =>
  progress.currentPage === 0
    ? `Fetching ${subject} failed before any of it arrived, so the totals below are empty rather than final. Reload the page to try again.`
    : `Fetching ${subject} failed, so the totals below cover only ${progress.currentPage} of ${progress.totalPages} pages of the range. Reload the page to try again.`;

const PaginationStatusAlerts = ({
  isFetchingMore,
  cancelled,
  progress,
  cancel,
  subject = "spend data",
  failed = false,
}: PaginationStatusAlertsProps) => (
  <>
    {isFetchingMore && (
      <Alert variant="warning" className="mb-2">
        <AlertDescription className="flex items-center justify-between text-inherit">
          <span>
            <Loader2 className="mr-2 inline size-4 animate-spin align-text-bottom" />
            Currently fetching {subject}: fetched {progress.currentPage} / {progress.totalPages} pages. Charts will
            update periodically as data loads. Moving off of this page will stop and reset this. To continue using the
            UI in the meantime,{" "}
            <a href={window.location.href} target="_blank" rel="noopener noreferrer">
              open a new tab <ExternalLink className="inline size-3.5 align-text-bottom" />
            </a>
            .
          </span>
          <Button variant="destructive" onClick={cancel}>
            Stop
          </Button>
        </AlertDescription>
      </Alert>
    )}
    {failed && (
      <Alert variant="error" className="mb-2">
        <AlertDescription className="text-inherit">{failureMessage(subject, progress)}</AlertDescription>
      </Alert>
    )}
    {cancelled && !failed && (
      <Alert variant="info" className="mb-2">
        <AlertDescription className="text-inherit">
          Showing partial {subject} ({progress.currentPage}/{progress.totalPages} pages loaded)
        </AlertDescription>
      </Alert>
    )}
  </>
);

export default PaginationStatusAlerts;
