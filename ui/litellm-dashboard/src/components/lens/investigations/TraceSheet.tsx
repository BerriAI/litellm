import { RunView } from "@/components/view_logs/TraceView/TraceDrawer";
import { useLocalRunSelection } from "@/components/view_logs/TraceView/traceRouting";
import { Sheet, SheetContent, SheetHeader, SheetTitle, SheetDescription } from "@/components/ui/sheet";

export function TraceSheet({
  open,
  traceId,
  traceRef,
  initialSpanId,
  accessToken,
  onClose,
}: {
  open: boolean;
  traceId: string;
  traceRef?: string;
  initialSpanId?: string;
  accessToken: string;
  onClose: () => void;
}) {
  return (
    <Sheet
      open={open}
      onOpenChange={(value) => {
        if (!value) onClose();
      }}
    >
      <SheetContent className="w-full overflow-y-auto data-[side=right]:sm:max-w-[90vw]">
        <SheetHeader>
          <SheetTitle className="text-xl">Original run</SheetTitle>
          <SheetDescription>Recorded agent steps and evidence</SheetDescription>
        </SheetHeader>
        {open && (
          <SheetRun
            key={`${traceId}:${traceRef}:${initialSpanId}`}
            traceId={traceId}
            traceRef={traceRef}
            initialSpanId={initialSpanId ?? null}
            accessToken={accessToken}
            onBack={onClose}
          />
        )}
      </SheetContent>
    </Sheet>
  );
}

function SheetRun({
  traceId,
  traceRef,
  initialSpanId,
  accessToken,
  onBack,
}: {
  traceId: string;
  traceRef?: string;
  initialSpanId: string | null;
  accessToken: string;
  onBack: () => void;
}) {
  const selection = useLocalRunSelection(initialSpanId);
  return (
    <RunView traceId={traceId} traceRef={traceRef} selection={selection} accessToken={accessToken} onBack={onBack} />
  );
}
