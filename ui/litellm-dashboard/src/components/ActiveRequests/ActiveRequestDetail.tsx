"use client";

import Link from "next/link";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Sheet, SheetContent, SheetDescription, SheetFooter, SheetHeader, SheetTitle } from "@/components/ui/sheet";
import type { ActiveRequest } from "./activeRequestsApi";
import { formatAge } from "./ActiveRequestColumns";

interface DetailRow {
  label: string;
  value: string | null | undefined;
  mono?: boolean;
}

const rows = (request: ActiveRequest): DetailRow[] => [
  { label: "Request ID", value: request.request_id, mono: true },
  { label: "Model", value: request.model },
  { label: "Call type", value: request.call_type },
  { label: "Route", value: request.route, mono: true },
  { label: "End user", value: request.end_user_id },
  { label: "User", value: request.user_id },
  { label: "Email", value: request.user_email },
  { label: "Organization", value: request.organization_alias || request.organization_id },
  { label: "Project", value: request.project_alias || request.project_id },
  { label: "Team", value: request.team_alias || request.team_id },
  { label: "Key alias", value: request.key_alias },
  { label: "Key hash", value: request.key_hash, mono: true },
  { label: "Worker", value: request.pod, mono: true },
];

export interface ActiveRequestDetailProps {
  request: ActiveRequest | null;
  now: number;
  onClose: () => void;
  onCancel: (request: ActiveRequest) => void;
  cancelling: boolean;
}

export default function ActiveRequestDetail({ request, now, onClose, onCancel, cancelling }: ActiveRequestDetailProps) {
  if (!request) return null;

  const started = new Date(request.started_at * 1000);

  return (
    <Sheet open onOpenChange={(open) => !open && onClose()}>
      <SheetContent className="w-full gap-0 sm:max-w-xl">
        <SheetHeader>
          <SheetTitle className="flex items-center gap-2">
            Running for {formatAge(request.started_at, now)}
            {request.streaming && <Badge variant="outline">stream</Badge>}
          </SheetTitle>
          <SheetDescription>Started {started.toLocaleString()}</SheetDescription>
        </SheetHeader>

        <dl className="grid gap-px overflow-y-auto bg-border px-(--card-spacing)">
          {rows(request).map((row) => (
            <div key={row.label} className="grid grid-cols-[10rem_1fr] gap-4 bg-background py-2">
              <dt className="text-sm text-muted-foreground">{row.label}</dt>
              <dd className={row.mono ? "min-w-0 truncate font-mono text-xs" : "min-w-0 truncate text-sm"}>
                {row.value || <span className="text-muted-foreground">not set</span>}
              </dd>
            </div>
          ))}
        </dl>

        <SheetFooter className="flex-row items-center justify-between gap-2">
          <Button
            variant="outline"
            nativeButton={false}
            render={<Link href={`/ui/logs?request_id=${encodeURIComponent(request.request_id)}`} />}
          >
            Open in Logs
          </Button>
          <Button variant="destructive" onClick={() => onCancel(request)} disabled={cancelling}>
            {cancelling ? "Cancelling…" : "Cancel request"}
          </Button>
        </SheetFooter>
      </SheetContent>
    </Sheet>
  );
}
