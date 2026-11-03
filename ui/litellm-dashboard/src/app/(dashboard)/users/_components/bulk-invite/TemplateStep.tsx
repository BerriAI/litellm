import { Download } from "lucide-react";

import { downloadBlob } from "@/components/EntityUsageExport/utils";
import { Button } from "@/components/ui/button";

import { TEMPLATE_COLUMNS, TEMPLATE_CSV } from "./columns";
import { StepHeading } from "./StepHeading";

const downloadTemplate = () => downloadBlob(new Blob([TEMPLATE_CSV], { type: "text/csv" }), "bulk_users_template.csv");

export function TemplateStep() {
  return (
    <>
      <StepHeading step={1} title="Download and fill the template" />

      <div className="ml-11 mb-6">
        <p className="mb-4">Add multiple users at once by following these steps:</p>
        <ol className="list-decimal list-inside space-y-2 ml-2 mb-4">
          <li>Download our CSV template</li>
          <li>Add your users&apos; information to the spreadsheet</li>
          <li>Save the file and upload it here</li>
          <li>After creation, download the results file containing the Virtual Keys for each user</li>
        </ol>

        <div className="bg-muted p-4 rounded-md border border-border mb-4">
          <h4 className="font-medium mb-2">Template Column Names</h4>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {TEMPLATE_COLUMNS.map((column) => (
              <div key={column.name} className="flex items-start">
                <div
                  className={`w-3 h-3 rounded-full ${column.required ? "bg-destructive" : "bg-border"} mt-1.5 mr-2 shrink-0`}
                ></div>
                <div>
                  <p className="font-medium">{column.name}</p>
                  <p className="text-sm text-muted-foreground">{column.help}</p>
                </div>
              </div>
            ))}
          </div>
        </div>

        <Button size="lg" className="w-full md:w-auto" onClick={downloadTemplate}>
          <Download className="size-4" />
          Download CSV Template
        </Button>
      </div>
    </>
  );
}
