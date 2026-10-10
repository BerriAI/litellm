"use client";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import { useId, useMemo, useState } from "react";
import { AddButton, FormFrame, RemoveButton } from "./DecisionFormControls";
import {
  blankQuestion,
  freeName,
  readForm,
  removeAt,
  renameKey,
  replaceAt,
  retype,
  withNoulCriterion,
  withoutKey,
  type ChoiceQuestion,
  type FormPayload,
  type FormQuestion,
  type NoulQuestion,
  type QuestionType,
  type ScoreQuestion,
} from "./lib/formPayload";
import type { SystemOnePayloadValidation } from "./lib/validatePayload";

const QUESTION_TYPES: readonly QuestionType[] = ["choice", "noul", "score"];
const TYPE_LABELS: Record<QuestionType, string> = { choice: "Choice", noul: "Yes / no", score: "Score" };

const isQuestionType = (value: unknown): value is QuestionType => QUESTION_TYPES.some((type) => type === value);

interface SystemOneFormProps {
  value: string;
  onChange: (value: string) => void;
  validation: SystemOnePayloadValidation;
  onOpenJson: () => void;
}

interface QuestionEditorProps<Q extends FormQuestion> {
  question: Q;
  onChange: (question: FormQuestion) => void;
}

interface KeyInputProps {
  value: string;
  taken: readonly string[];
  onCommit: (value: string) => void;
  label: string;
  placeholder?: string;
}

function keyError(next: string, current: string, taken: readonly string[]): string | undefined {
  if (next === "") {
    return "Enter a name";
  }
  if (next !== current && taken.includes(next)) {
    return `"${next}" is already used`;
  }
  return undefined;
}

function KeyInput({ value, taken, onCommit, label, placeholder }: KeyInputProps) {
  const errorId = useId();
  const [draft, setDraft] = useState(value);
  const next = draft.trim();
  const error = keyError(next, value, taken);

  function commit() {
    if (error === undefined && next !== value) {
      onCommit(next);
      return;
    }
    setDraft(value);
  }

  return (
    <div className="grid min-w-0 flex-1 gap-1">
      <Input
        aria-label={label}
        aria-invalid={error !== undefined}
        aria-describedby={error === undefined ? undefined : errorId}
        value={draft}
        placeholder={placeholder}
        onChange={(event) => setDraft(event.target.value)}
        onBlur={commit}
      />
      {error !== undefined && (
        <p id={errorId} className="text-xs text-destructive">
          {error}
        </p>
      )}
    </div>
  );
}

function ChoiceOptions({ question, onChange }: QuestionEditorProps<ChoiceQuestion>) {
  const labels = Object.keys(question.criteria);
  const writeCriteria = (criteria: Record<string, string>) => onChange({ ...question, criteria });

  return (
    <div className="grid gap-2">
      <span className="text-sm font-medium">Options</span>
      {Object.entries(question.criteria).map(([label, description], index) => (
        <div key={index} className="flex items-start gap-2">
          <KeyInput
            key={label}
            label="Option label"
            value={label}
            taken={labels}
            onCommit={(next) => writeCriteria(renameKey(question.criteria, label, next))}
          />
          <Input
            aria-label={`Description of ${label}`}
            className="flex-2"
            value={description}
            placeholder="Optional description"
            onChange={(event) => writeCriteria({ ...question.criteria, [label]: event.target.value })}
          />
          <RemoveButton
            label={`Remove option ${label}`}
            onClick={() => writeCriteria(withoutKey(question.criteria, label))}
          />
        </div>
      ))}
      <AddButton onClick={() => writeCriteria({ ...question.criteria, [freeName(labels, "option")]: "" })}>
        Add option
      </AddButton>
    </div>
  );
}

function NoulCriteria({ question, onChange }: QuestionEditorProps<NoulQuestion>) {
  const yesId = useId();
  const noId = useId();

  return (
    <div className="grid gap-2 sm:grid-cols-2">
      <div className="grid gap-1.5">
        <Label htmlFor={yesId}>Yes means</Label>
        <Input
          id={yesId}
          value={question.criteria?.true ?? ""}
          placeholder="Optional"
          onChange={(event) => onChange(withNoulCriterion(question, "true", event.target.value))}
        />
      </div>
      <div className="grid gap-1.5">
        <Label htmlFor={noId}>No means</Label>
        <Input
          id={noId}
          value={question.criteria?.false ?? ""}
          placeholder="Optional"
          onChange={(event) => onChange(withNoulCriterion(question, "false", event.target.value))}
        />
      </div>
    </div>
  );
}

function ScoreLevels({ question, onChange }: QuestionEditorProps<ScoreQuestion>) {
  const writeCriteria = (criteria: string[]) => onChange({ ...question, criteria });

  return (
    <div className="grid gap-2">
      <span className="text-sm font-medium">Levels, lowest first</span>
      {question.criteria.map((level, index) => (
        <div key={index} className="flex items-center gap-2">
          <span className="w-6 shrink-0 text-right text-xs text-muted-foreground tabular-nums">{index}</span>
          <Input
            aria-label={`Level ${index}`}
            value={level}
            placeholder="What this score means"
            onChange={(event) => writeCriteria(replaceAt(question.criteria, index, event.target.value))}
          />
          <RemoveButton
            label={`Remove level ${index}`}
            onClick={() => writeCriteria(removeAt(question.criteria, index))}
          />
        </div>
      ))}
      <AddButton onClick={() => writeCriteria([...question.criteria, ""])}>Add level</AddButton>
    </div>
  );
}

function CriteriaEditor({ question, onChange }: QuestionEditorProps<FormQuestion>) {
  switch (question.type) {
    case "choice":
      return <ChoiceOptions question={question} onChange={onChange} />;
    case "noul":
      return <NoulCriteria question={question} onChange={onChange} />;
    case "score":
      return <ScoreLevels question={question} onChange={onChange} />;
  }
}

interface QuestionCardProps extends QuestionEditorProps<FormQuestion> {
  name: string;
  taken: readonly string[];
  onRename: (name: string) => void;
  onRemove: () => void;
}

function QuestionCard({ name, question, taken, onChange, onRename, onRemove }: QuestionCardProps) {
  const instructionsId = useId();

  return (
    <fieldset aria-label={`Question ${name}`} className="grid gap-3 rounded-md border p-3">
      <div className="flex flex-wrap items-start gap-2">
        <KeyInput key={name} label="Question name" value={name} taken={taken} onCommit={onRename} />
        <Select
          value={question.type}
          onValueChange={(type) => isQuestionType(type) && onChange(retype(question, type))}
        >
          <SelectTrigger className="w-36" aria-label="Answer type">
            <SelectValue>{TYPE_LABELS[question.type]}</SelectValue>
          </SelectTrigger>
          <SelectContent>
            {QUESTION_TYPES.map((type) => (
              <SelectItem key={type} value={type}>
                {TYPE_LABELS[type]}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
        <RemoveButton label={`Remove question ${name}`} onClick={onRemove} />
      </div>
      <div className="grid gap-1.5">
        <Label htmlFor={instructionsId}>Instructions</Label>
        <Textarea
          id={instructionsId}
          rows={2}
          value={question.instructions ?? ""}
          placeholder="What should the model decide?"
          onChange={(event) => onChange({ ...question, instructions: event.target.value })}
        />
      </div>
      <CriteriaEditor question={question} onChange={onChange} />
    </fieldset>
  );
}

export default function SystemOneForm({ value, onChange, validation, onOpenJson }: SystemOneFormProps) {
  const stateId = useId();
  const form = useMemo(() => readForm(value), [value]);

  if (form === undefined) {
    return (
      <FormFrame validation={validation}>
        <p className="text-sm text-muted-foreground">
          This request has JSON the form can&apos;t show, such as a syntax error or a value that isn&apos;t text.
        </p>
        <Button variant="outline" className="w-fit" onClick={onOpenJson}>
          Edit in JSON
        </Button>
      </FormFrame>
    );
  }

  const questions = form.questions ?? {};
  const names = Object.keys(questions);
  const write = (next: FormPayload) => onChange(JSON.stringify(next, null, 2));
  const writeQuestions = (next: Record<string, FormQuestion>) => write({ ...form, questions: next });

  return (
    <FormFrame validation={validation}>
      <div className="grid gap-1.5">
        <Label htmlFor={stateId}>Input</Label>
        <p className="text-xs text-muted-foreground">
          The text the model reads before it answers, such as a support ticket or a chat message.
        </p>
        <Textarea
          id={stateId}
          rows={4}
          value={form.state ?? ""}
          placeholder="Paste the text the model should decide about"
          onChange={(event) => write({ ...form, state: event.target.value })}
        />
      </div>
      {Object.entries(questions).map(([name, question], index) => (
        <QuestionCard
          key={index}
          name={name}
          question={question}
          taken={names}
          onChange={(next) => writeQuestions({ ...questions, [name]: next })}
          onRename={(next) => writeQuestions(renameKey(questions, name, next))}
          onRemove={() => writeQuestions(withoutKey(questions, name))}
        />
      ))}
      <AddButton onClick={() => writeQuestions({ ...questions, [freeName(names, "question")]: blankQuestion() })}>
        Add question
      </AddButton>
    </FormFrame>
  );
}
