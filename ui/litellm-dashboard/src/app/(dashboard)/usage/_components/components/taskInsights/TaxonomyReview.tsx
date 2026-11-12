"use client";

import React from "react";
import { useTranslation } from "react-i18next";

import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from "@/components/ui/alert-dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { formatCount, type DraftTaskEdit, type TaskDefinition } from "./taskInsightsData";

interface TaxonomyReviewProps {
  drafts: readonly TaskDefinition[];
  canEdit: boolean;
  saving: boolean;
  freezing: boolean;
  onSave: (taskId: string, edit: DraftTaskEdit) => void;
  onFreeze: () => void;
}

function DraftTaskForm({
  task,
  canEdit,
  saving,
  onSave,
}: {
  task: TaskDefinition;
  canEdit: boolean;
  saving: boolean;
  onSave: (taskId: string, edit: DraftTaskEdit) => void;
}) {
  const { t } = useTranslation();
  const [name, setName] = React.useState(task.name);
  const [inclusion, setInclusion] = React.useState(task.inclusion_criteria);
  const [exclusion, setExclusion] = React.useState(task.exclusion_criteria);
  const changed =
    name !== task.name || inclusion !== task.inclusion_criteria || exclusion !== task.exclusion_criteria;
  const id = `draft-${task.task_id}`;
  return (
    <li className="flex flex-col gap-2 rounded-lg border p-3">
      <div className="flex items-center justify-between gap-2">
        <label htmlFor={`${id}-name`} className="sr-only">
          {t("usage.taskInsights.review.name")}
        </label>
        <Input
          id={`${id}-name`}
          value={name}
          disabled={!canEdit}
          onChange={(event) => setName(event.target.value)}
          className="max-w-sm font-medium"
        />
        <span className="shrink-0 text-xs text-muted-foreground">
          {t("usage.taskInsights.review.size", { requests: formatCount(task.discovery_size) })}
        </span>
      </div>
      <p className="text-sm text-muted-foreground">{task.description}</p>
      <div className="grid gap-2 sm:grid-cols-2">
        <div className="flex flex-col gap-1">
          <label htmlFor={`${id}-include`} className="text-xs font-medium text-muted-foreground">
            {t("usage.taskInsights.detail.includes")}
          </label>
          <Textarea
            id={`${id}-include`}
            value={inclusion}
            disabled={!canEdit}
            onChange={(event) => setInclusion(event.target.value)}
          />
        </div>
        <div className="flex flex-col gap-1">
          <label htmlFor={`${id}-exclude`} className="text-xs font-medium text-muted-foreground">
            {t("usage.taskInsights.detail.excludes")}
          </label>
          <Textarea
            id={`${id}-exclude`}
            value={exclusion}
            disabled={!canEdit}
            onChange={(event) => setExclusion(event.target.value)}
          />
        </div>
      </div>
      {task.examples.length > 0 && (
        <ul className="list-disc pl-5 text-xs text-muted-foreground">
          {task.examples.map((example) => (
            <li key={example}>{example}</li>
          ))}
        </ul>
      )}
      {canEdit && (
        <Button
          size="sm"
          variant="outline"
          className="self-end"
          disabled={!changed || saving || name.trim() === ""}
          onClick={() =>
            onSave(task.task_id, { name: name.trim(), inclusion_criteria: inclusion, exclusion_criteria: exclusion })
          }
        >
          {t("usage.taskInsights.review.save")}
        </Button>
      )}
    </li>
  );
}

export default function TaxonomyReview({ drafts, canEdit, saving, freezing, onSave, onFreeze }: TaxonomyReviewProps) {
  const { t } = useTranslation();
  return (
    <div className="flex flex-col gap-3">
      <p className="text-sm text-muted-foreground">{t("usage.taskInsights.review.intro")}</p>
      <ul className="flex flex-col gap-2">
        {drafts.map((task) => (
          <DraftTaskForm key={task.task_id} task={task} canEdit={canEdit} saving={saving} onSave={onSave} />
        ))}
      </ul>
      {canEdit && (
        <AlertDialog>
          <AlertDialogTrigger render={<Button className="self-end" disabled={freezing} />}>
            {t("usage.taskInsights.review.freeze")}
          </AlertDialogTrigger>
          <AlertDialogContent>
            <AlertDialogHeader>
              <AlertDialogTitle>{t("usage.taskInsights.review.freezeTitle")}</AlertDialogTitle>
              <AlertDialogDescription>{t("usage.taskInsights.review.freezeDescription")}</AlertDialogDescription>
            </AlertDialogHeader>
            <AlertDialogFooter>
              <AlertDialogCancel>{t("usage.taskInsights.cancel")}</AlertDialogCancel>
              <AlertDialogAction onClick={onFreeze}>{t("usage.taskInsights.review.freezeConfirm")}</AlertDialogAction>
            </AlertDialogFooter>
          </AlertDialogContent>
        </AlertDialog>
      )}
    </div>
  );
}
