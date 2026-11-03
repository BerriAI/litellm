import { CheckCircleIcon, XCircleIcon } from "@heroicons/react/outline";
import { CopyToClipboard } from "react-copy-to-clipboard";

import { Button } from "@/components/ui/button";
import {
  NUMERIC_CELL_CLASS,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { toast } from "@/lib/toast";

import type { BulkInviteRow } from "./useBulkInvite";

const PREVIEW_PAGE_SIZE = 5;

function FailureStatus({ label, error }: { label: string; error: string | undefined }) {
  return (
    <div>
      <div className="flex items-center">
        <XCircleIcon className="h-5 w-5 text-destructive mr-2" />
        <span className="text-destructive">{label}</span>
      </div>
      {error && <span className="text-sm text-destructive ml-7">{error}</span>}
    </div>
  );
}

function SuccessStatus({ invitationLink }: { invitationLink: string | undefined }) {
  return (
    <div>
      <div className="flex items-center">
        <CheckCircleIcon className="h-5 w-5 text-success mr-2" />
        <span className="text-success">Success</span>
      </div>
      {invitationLink && (
        <div className="mt-1">
          <div className="flex items-center">
            <span className="text-xs text-muted-foreground truncate max-w-[150px]">{invitationLink}</span>
            <CopyToClipboard text={invitationLink} onCopy={() => toast.success("Invitation link copied!")}>
              <button className="ml-1 text-info text-xs hover:text-info/80">Copy</button>
            </CopyToClipboard>
          </div>
        </div>
      )}
    </div>
  );
}

function StatusCell({ row }: { row: BulkInviteRow }) {
  if (!row.isValid) return <FailureStatus label="Invalid" error={row.error} />;
  if (!row.status || row.status === "pending") return <span className="text-muted-foreground">Pending</span>;
  if (row.status === "success") return <SuccessStatus invitationLink={row.invitation_link} />;
  return <FailureStatus label="Failed" error={row.error && JSON.stringify(row.error)} />;
}

interface PreviewTableProps {
  rows: BulkInviteRow[];
  pageIndex: number;
  onPageChange: (pageIndex: number) => void;
}

export function PreviewTable({ rows, pageIndex, onPageChange }: PreviewTableProps) {
  const pageCount = Math.max(1, Math.ceil(rows.length / PREVIEW_PAGE_SIZE));
  const currentPage = Math.min(pageIndex, pageCount - 1);
  const visibleRows = rows.slice(currentPage * PREVIEW_PAGE_SIZE, (currentPage + 1) * PREVIEW_PAGE_SIZE);

  return (
    <>
      <div className="max-h-[300px] overflow-y-auto">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead className="w-20">Row</TableHead>
              <TableHead>Email</TableHead>
              <TableHead>Role</TableHead>
              <TableHead>Teams</TableHead>
              <TableHead className={NUMERIC_CELL_CLASS}>Budget</TableHead>
              <TableHead>Status</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {visibleRows.map((row) => (
              <TableRow key={row.rowNumber} className={!row.isValid ? "bg-destructive/10" : ""}>
                <TableCell>{row.rowNumber}</TableCell>
                <TableCell className="whitespace-normal break-words">{row.user_email}</TableCell>
                <TableCell className="whitespace-normal break-words">{row.user_role}</TableCell>
                <TableCell className="whitespace-normal break-words">{row.teams}</TableCell>
                <TableCell className={NUMERIC_CELL_CLASS}>{row.max_budget}</TableCell>
                <TableCell className="whitespace-normal break-words">
                  <StatusCell row={row} />
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </div>

      {pageCount > 1 && (
        <div className="flex items-center justify-end gap-3 mt-2">
          <span className="text-sm text-muted-foreground">
            Page {currentPage + 1} of {pageCount}
          </span>
          <Button
            variant="outline"
            size="sm"
            onClick={() => onPageChange(currentPage - 1)}
            disabled={currentPage === 0}
          >
            Previous
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={() => onPageChange(currentPage + 1)}
            disabled={currentPage >= pageCount - 1}
          >
            Next
          </Button>
        </div>
      )}
    </>
  );
}
