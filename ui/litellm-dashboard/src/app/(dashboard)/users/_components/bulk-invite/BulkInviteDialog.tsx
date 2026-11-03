import { CheckCircleIcon } from "@heroicons/react/outline";
import { Download, TriangleAlert } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";

import { CsvDropzone } from "./CsvDropzone";
import { PreviewTable } from "./PreviewTable";
import { StepHeading } from "./StepHeading";
import { TemplateStep } from "./TemplateStep";
import { useBulkInvite, type BulkInviteRow } from "./useBulkInvite";

export interface BulkInviteDialogProps {
  open: boolean;
  onClose: () => void;
  accessToken: string;
  teams: readonly { team_id: string }[] | null;
  onUsersCreated?: () => void;
}

type BulkInvite = ReturnType<typeof useBulkInvite>;

function ParseErrorNotice({ parseError, rows }: { parseError: string; rows: BulkInviteRow[] }) {
  return (
    <div className="ml-11 mb-4 p-4 bg-destructive/10 border border-destructive/20 rounded-md">
      <div className="flex items-start">
        <TriangleAlert className="size-4 shrink-0 text-destructive mr-2 mt-1" />
        <div className="min-w-0">
          <p className="text-destructive font-medium break-words">{parseError}</p>
          {rows.some((row) => !row.isValid) && (
            <ul className="mt-2 list-disc list-inside text-destructive text-sm">
              <li>Check the table below for specific errors in each row</li>
              <li>Common issues include invalid email formats, missing required fields, or incorrect role values</li>
              <li>Fix these issues in your CSV file and upload again</li>
            </ul>
          )}
        </div>
      </div>
    </div>
  );
}

function Summary({ rows, hasResults }: { rows: BulkInviteRow[]; hasResults: boolean }) {
  if (!hasResults) {
    return (
      <div className="flex items-center">
        <p className="text-lg font-medium mr-3">User Preview</p>
        <p className="text-sm bg-info/15 text-info px-2 py-1 rounded-sm">
          {rows.filter((row) => row.isValid).length} of {rows.length} users valid
        </p>
      </div>
    );
  }
  const failedCount = rows.filter((row) => row.status === "failed").length;
  return (
    <div className="flex items-center">
      <p className="text-lg font-medium mr-3">Creation Summary</p>
      <p className="text-sm bg-success/15 text-success px-2 py-1 rounded-sm mr-2">
        {rows.filter((row) => row.status === "success").length} Successful
      </p>
      {failedCount > 0 && (
        <p className="text-sm bg-destructive/15 text-destructive px-2 py-1 rounded-sm">{failedCount} Failed</p>
      )}
    </div>
  );
}

function CreateButton({ bulkInvite }: { bulkInvite: BulkInvite }) {
  const validCount = bulkInvite.rows.filter((row) => row.isValid).length;
  return (
    <Button onClick={bulkInvite.createUsers} disabled={validCount === 0 || bulkInvite.isProcessing}>
      {bulkInvite.isProcessing ? "Creating..." : `Create ${validCount} Users`}
    </Button>
  );
}

function CreationCompleteNotice() {
  return (
    <div className="mb-4 p-4 bg-info/10 border border-info/20 rounded-md">
      <div className="flex items-start">
        <div className="mr-3 mt-1">
          <CheckCircleIcon className="h-5 w-5 text-info" />
        </div>
        <div>
          <p className="font-medium text-info">User creation complete</p>
          <p className="block text-sm text-info mt-1">
            <span className="font-medium">Next step:</span> Download the credentials file containing Virtual Keys and
            invitation links. Users will need these Virtual Keys to make LLM requests through LiteLLM.
          </p>
        </div>
      </div>
    </div>
  );
}

function ReviewStep({ bulkInvite }: { bulkInvite: BulkInvite }) {
  const { rows, parseError } = bulkInvite;
  const hasResults = rows.some((row) => row.status === "success" || row.status === "failed");

  return (
    <div className="mb-6">
      <StepHeading step={3} title={hasResults ? "User Creation Results" : "Review and create users"} />

      {parseError && <ParseErrorNotice parseError={parseError} rows={rows} />}

      <div className="ml-11">
        <div className="flex justify-between items-center mb-3">
          <div className="flex items-center">
            <Summary rows={rows} hasResults={hasResults} />
          </div>

          {!hasResults && (
            <div className="flex space-x-3">
              <Button variant="outline" onClick={bulkInvite.backToUpload}>
                Back
              </Button>
              <CreateButton bulkInvite={bulkInvite} />
            </div>
          )}
        </div>

        {rows.some((row) => row.status === "success") && <CreationCompleteNotice />}

        <PreviewTable rows={rows} pageIndex={bulkInvite.pageIndex} onPageChange={bulkInvite.setPageIndex} />

        {hasResults ? (
          <div className="flex justify-end mt-4">
            <Button variant="outline" onClick={bulkInvite.backToUpload} className="mr-3">
              Start New Bulk Import
            </Button>
            <Button onClick={bulkInvite.downloadResults}>
              <Download className="size-4" />
              Download User Credentials
            </Button>
          </div>
        ) : (
          <div className="flex justify-end mt-4">
            <Button variant="outline" onClick={bulkInvite.backToUpload} className="mr-3">
              Back
            </Button>
            <CreateButton bulkInvite={bulkInvite} />
          </div>
        )}
      </div>
    </div>
  );
}

export function BulkInviteDialog({ open, onClose, accessToken, teams, onUsersCreated }: BulkInviteDialogProps) {
  const bulkInvite = useBulkInvite({ accessToken, teams, onUsersCreated });

  return (
    <Dialog open={open} onOpenChange={(isOpen) => !isOpen && onClose()}>
      <DialogContent className="max-h-[calc(100dvh-2rem)] overflow-y-auto sm:max-w-[800px]">
        <DialogHeader>
          <DialogTitle>Bulk Invite Users</DialogTitle>
        </DialogHeader>
        <div className="flex flex-col">
          {bulkInvite.rows.length === 0 ? (
            <div className="mb-6">
              <TemplateStep />
              <StepHeading step={2} title="Upload your completed CSV" />
              <div className="ml-11">
                <CsvDropzone
                  selectedFile={bulkInvite.selectedFile}
                  fileError={bulkInvite.fileError}
                  csvStructureError={bulkInvite.csvStructureError}
                  onFile={bulkInvite.uploadFile}
                  onRemove={bulkInvite.removeFile}
                />
              </div>
            </div>
          ) : (
            <ReviewStep bulkInvite={bulkInvite} />
          )}
        </div>
      </DialogContent>
    </Dialog>
  );
}
