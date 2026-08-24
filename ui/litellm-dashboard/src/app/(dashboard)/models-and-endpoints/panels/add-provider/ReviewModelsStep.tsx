"use client";

import React from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Switch } from "@/components/ui/switch";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { MultiSelect } from "@/components/shared/MultiSelect";
import { ArrowLeft, Plus, X } from "lucide-react";
import { buildManualRow, type DiscoveredModelRow } from "./wizardLogic";

interface ReviewModelsStepProps {
  rows: DiscoveredModelRow[];
  setRows: React.Dispatch<React.SetStateAction<DiscoveredModelRow[]>>;
  createError: string | null;
  onBack: () => void;
  onCreateModels: () => void;
}

const ReviewModelsStep: React.FC<ReviewModelsStepProps> = ({ rows, setRows, createError, onBack, onCreateModels }) => {
  const [manualId, setManualId] = React.useState("");

  const hasBlankName = rows.some((row) => row.modelName.trim() === "");

  const updateRow = (id: string, patch: Partial<DiscoveredModelRow>) =>
    setRows((current) => current.map((row) => (row.id === id ? { ...row, ...patch } : row)));

  const removeRow = (id: string) => setRows((current) => current.filter((row) => row.id !== id));

  const addManualRow = () => {
    const trimmed = manualId.trim();
    if (!trimmed) return;
    setRows((current) => [...current, buildManualRow(trimmed)]);
    setManualId("");
  };

  return (
    <Card>
      <CardContent className="space-y-4">
        {rows.length === 0 ? (
          <p className="text-sm text-muted-foreground">No models discovered. Add one manually below.</p>
        ) : (
          <div className="overflow-x-auto rounded-md border">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Upstream ID</TableHead>
                  <TableHead>Enabled</TableHead>
                  <TableHead>Model name</TableHead>
                  <TableHead>Alternate names</TableHead>
                  <TableHead />
                </TableRow>
              </TableHeader>
              <TableBody>
                {rows.map((row) => (
                  <TableRow key={row.id}>
                    <TableCell className="whitespace-nowrap font-mono text-xs">{row.upstreamId}</TableCell>
                    <TableCell>
                      <Switch
                        checked={row.enabled}
                        onCheckedChange={(checked) => updateRow(row.id, { enabled: checked })}
                        aria-label={`Enable ${row.upstreamId}`}
                      />
                    </TableCell>
                    <TableCell className="min-w-[160px]">
                      <Input
                        value={row.modelName}
                        onChange={(e) => updateRow(row.id, { modelName: e.target.value })}
                        aria-label={`Model name for ${row.upstreamId}`}
                      />
                    </TableCell>
                    <TableCell className="min-w-[200px]">
                      <MultiSelect
                        value={row.alternateNames}
                        onValueChange={(value) => updateRow(row.id, { alternateNames: value })}
                        options={[]}
                        allowCustomValues
                        placeholder="Add alternate names"
                        emptyText="Type to add an alternate name"
                      />
                    </TableCell>
                    <TableCell>
                      {row.manual && (
                        <Button
                          variant="ghost"
                          size="icon-sm"
                          aria-label={`Remove ${row.upstreamId}`}
                          onClick={() => removeRow(row.id)}
                        >
                          <X className="size-4" />
                        </Button>
                      )}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>
        )}

        <div className="flex items-end gap-2">
          <div className="flex-1">
            <label htmlFor="add-provider-manual-model" className="mb-1 block text-xs text-muted-foreground">
              Add a hidden model manually
            </label>
            <Input
              id="add-provider-manual-model"
              value={manualId}
              onChange={(e) => setManualId(e.target.value)}
              placeholder="upstream model id"
            />
          </div>
          <Button type="button" variant="outline" onClick={addManualRow} disabled={!manualId.trim()}>
            <Plus className="mr-1 size-4" /> Add
          </Button>
        </div>

        {createError && <p className="text-sm text-destructive">{createError}</p>}

        <div className="flex justify-between">
          <Button type="button" variant="outline" onClick={onBack}>
            <ArrowLeft className="mr-1 size-4" /> Back
          </Button>
          <Button disabled={rows.length === 0 || hasBlankName} onClick={onCreateModels}>
            Create {rows.length} model{rows.length === 1 ? "" : "s"}
          </Button>
        </div>
      </CardContent>
    </Card>
  );
};

export default ReviewModelsStep;
