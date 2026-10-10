"use client";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import { useId, useMemo } from "react";
import { AddButton, FormFrame, RemoveButton } from "./DecisionFormControls";
import { freeName, removeAt, replaceAt } from "./lib/formPayload";
import {
  readDecisionsForm,
  retypeDecisionQuestion,
  type DecisionsFormPayload,
  type DecisionsFormQuestion,
  type DecisionsQuestionType,
} from "./lib/openAIDecisionsForm";
import type { OpenAIDecisionsRequest } from "./lib/openAIDecisions";
import type { PayloadValidation } from "./lib/validatePayload";

const QUESTION_TYPES: readonly DecisionsQuestionType[] = ["predicate", "choice", "score"];
const TYPE_LABELS = { predicate: "Predicate", choice: "Choice", score: "Score" };

interface QuestionEditorProps<Q extends DecisionsFormQuestion = DecisionsFormQuestion> {
  question: Q;
  onChange: (question: DecisionsFormQuestion) => void;
}

function Choices({ question, onChange }: QuestionEditorProps<Extract<DecisionsFormQuestion, { type: "choice" }>>) {
  const choices = question.choices ?? [];
  const write = (next: typeof choices) => onChange({ ...question, choices: next });

  return (
    <div className="grid gap-2">
      <span className="text-sm font-medium">Choices</span>
      {choices.map((choice, index) => (
        <div key={index} className="grid gap-2 rounded-md border p-2">
          <div className="flex items-center gap-2">
            <Select
              value={typeof choice.value === "string" ? "text" : String(choice.value)}
              onValueChange={(type) => {
                if (type === "text" || type === "true" || type === "false") {
                  const value = type === "text" ? String(choice.value) : type === "true";
                  write(replaceAt(choices, index, { ...choice, value }));
                }
              }}
            >
              <SelectTrigger className="w-40" aria-label={`Choice ${index + 1} value type`}>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="text">Text</SelectItem>
                <SelectItem value="true">true (boolean)</SelectItem>
                <SelectItem value="false">false (boolean)</SelectItem>
              </SelectContent>
            </Select>
            {typeof choice.value === "string" && (
              <Input
                aria-label={`Choice ${index + 1} value`}
                value={choice.value}
                onChange={(event) => write(replaceAt(choices, index, { ...choice, value: event.target.value }))}
              />
            )}
            <RemoveButton label={`Remove choice ${index + 1}`} onClick={() => write(removeAt(choices, index))} />
          </div>
          <Input
            aria-label={`Choice ${index + 1} description`}
            value={choice.description ?? ""}
            placeholder="Optional description"
            onChange={(event) => write(replaceAt(choices, index, { ...choice, description: event.target.value }))}
          />
        </div>
      ))}
      <AddButton
        onClick={() =>
          write([
            ...choices,
            {
              value: freeName(
                choices.map((c) => String(c.value)),
                "option",
              ),
            },
          ])
        }
      >
        Add choice
      </AddButton>
    </div>
  );
}

function Levels({ question, onChange }: QuestionEditorProps<Extract<DecisionsFormQuestion, { type: "score" }>>) {
  const levels = question.levels ?? [];
  const write = (next: typeof levels) => onChange({ ...question, levels: next });

  return (
    <div className="grid gap-2">
      <span className="text-sm font-medium">Levels, lowest first</span>
      {levels.map((level, index) => (
        <div key={index} className="flex items-center gap-2">
          <Input
            aria-label={`Level ${index + 1} label`}
            value={level.label}
            placeholder="Label"
            onChange={(event) => write(replaceAt(levels, index, { ...level, label: event.target.value }))}
          />
          <Input
            aria-label={`Level ${index + 1} description`}
            value={level.description ?? ""}
            placeholder="Optional description"
            onChange={(event) => write(replaceAt(levels, index, { ...level, description: event.target.value }))}
          />
          <RemoveButton label={`Remove level ${index + 1}`} onClick={() => write(removeAt(levels, index))} />
        </div>
      ))}
      <AddButton onClick={() => write([...levels, { label: String(levels.length) }])}>Add level</AddButton>
    </div>
  );
}

interface QuestionCardProps extends QuestionEditorProps {
  index: number;
  onRemove: () => void;
}

function QuestionCard({ question, index, onChange, onRemove }: QuestionCardProps) {
  const instructionsId = useId();

  return (
    <fieldset aria-label={`Question ${index + 1}`} className="grid gap-3 rounded-md border p-3">
      <div className="flex flex-wrap items-center gap-2">
        <Input
          aria-label="Question name"
          className="min-w-0 flex-1"
          value={question.name ?? ""}
          placeholder="Optional question name"
          onChange={(event) => onChange({ ...question, name: event.target.value })}
        />
        <Select
          value={question.type}
          onValueChange={(type) => {
            const next = QUESTION_TYPES.find((value) => value === type);
            if (next) onChange(retypeDecisionQuestion(question, next));
          }}
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
        <RemoveButton label={`Remove question ${index + 1}`} onClick={onRemove} />
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
      {question.type === "choice" && <Choices question={question} onChange={onChange} />}
      {question.type === "score" && <Levels question={question} onChange={onChange} />}
    </fieldset>
  );
}

interface OpenAIDecisionsFormProps {
  value: string;
  onChange: (value: string) => void;
  validation: PayloadValidation<OpenAIDecisionsRequest>;
  onOpenJson: () => void;
}

export default function OpenAIDecisionsForm({ value, onChange, validation, onOpenJson }: OpenAIDecisionsFormProps) {
  const inputId = useId();
  const safetyId = useId();
  const form = useMemo(() => readDecisionsForm(value), [value]);

  if (form === undefined) {
    return (
      <FormFrame validation={validation}>
        <p className="text-sm text-muted-foreground">
          This request has JSON the form can&apos;t show. Edit it in JSON to fix its shape.
        </p>
        <Button variant="outline" className="w-fit" onClick={onOpenJson}>
          Edit in JSON
        </Button>
      </FormFrame>
    );
  }

  const questions = form.questions ?? [];
  const write = (next: DecisionsFormPayload) => onChange(JSON.stringify(next, null, 2));
  const writeQuestions = (next: DecisionsFormQuestion[]) => write({ ...form, questions: next });

  return (
    <FormFrame validation={validation}>
      <div className="grid gap-1.5">
        {Array.isArray(form.input) ? (
          <>
            <p className="text-sm text-muted-foreground">
              Input contains messages. Edit messages and image parts in JSON; form edits keep them unchanged.
            </p>
            <Button variant="outline" className="w-fit" onClick={onOpenJson}>
              Edit input in JSON
            </Button>
          </>
        ) : (
          <>
            <Label htmlFor={inputId}>Input</Label>
            <Textarea
              id={inputId}
              rows={4}
              value={form.input ?? ""}
              placeholder="Paste the text the model should decide about"
              onChange={(event) => write({ ...form, input: event.target.value })}
            />
          </>
        )}
      </div>
      {questions.map((question, index) => (
        <QuestionCard
          key={index}
          index={index}
          question={question}
          onChange={(next) => writeQuestions(replaceAt(questions, index, next))}
          onRemove={() => writeQuestions(removeAt(questions, index))}
        />
      ))}
      <AddButton
        onClick={() =>
          writeQuestions([
            ...questions,
            {
              type: "predicate",
              name: freeName(
                questions.map((q) => q.name ?? ""),
                "question",
              ),
              instructions: "",
            },
          ])
        }
      >
        Add question
      </AddButton>
      <div className="grid gap-1.5">
        <Label htmlFor={safetyId}>Safety identifier</Label>
        <Input
          id={safetyId}
          value={form.safety_identifier ?? ""}
          placeholder="Optional"
          onChange={(event) => write({ ...form, safety_identifier: event.target.value || undefined })}
        />
      </div>
    </FormFrame>
  );
}
