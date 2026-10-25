"use client";

import { useLanguage } from "@/contexts/LanguageContext";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import { useId, useMemo } from "react";
import { AddButton, FormFrame, QuestionCardHeader, RemoveButton } from "./DecisionFormControls";
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
  const { t } = useLanguage();
  const choices = question.choices ?? [];
  const write = (next: typeof choices) => onChange({ ...question, choices: next });

  return (
    <div className="grid gap-2">
      <span className="text-sm font-medium">{t("Choices")}</span>
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
              <SelectTrigger className="w-40" aria-label={t("Choice {value} value type", { value: String(index + 1) })}>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="text">{t("Text")}</SelectItem>
                <SelectItem value="true">{t("true (boolean)")}</SelectItem>
                <SelectItem value="false">{t("false (boolean)")}</SelectItem>
              </SelectContent>
            </Select>
            {typeof choice.value === "string" && (
              <Input
                aria-label={t("Choice {value} value", { value: String(index + 1) })}
                value={choice.value}
                onChange={(event) => write(replaceAt(choices, index, { ...choice, value: event.target.value }))}
              />
            )}
            <RemoveButton
              label={t("Remove choice {value}", { value: String(index + 1) })}
              onClick={() => write(removeAt(choices, index))}
            />
          </div>
          <Input
            aria-label={t("Choice {value} description", { value: String(index + 1) })}
            value={choice.description ?? ""}
            placeholder={t("Optional description")}
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
        {t("Add choice")}
      </AddButton>
    </div>
  );
}

function Levels({ question, onChange }: QuestionEditorProps<Extract<DecisionsFormQuestion, { type: "score" }>>) {
  const { t } = useLanguage();
  const levels = question.levels ?? [];
  const write = (next: typeof levels) => onChange({ ...question, levels: next });

  return (
    <div className="grid gap-2">
      <span className="text-sm font-medium">{t("Levels, lowest first")}</span>
      {levels.map((level, index) => (
        <div key={index} className="flex items-center gap-2">
          <Input
            aria-label={t("Level {value} label", { value: String(index + 1) })}
            value={level.label}
            placeholder={t("Label")}
            onChange={(event) => write(replaceAt(levels, index, { ...level, label: event.target.value }))}
          />
          <Input
            aria-label={t("Level {value} description", { value: String(index + 1) })}
            value={level.description ?? ""}
            placeholder={t("Optional description")}
            onChange={(event) => write(replaceAt(levels, index, { ...level, description: event.target.value }))}
          />
          <RemoveButton
            label={t("Remove level {value}", { value: String(index + 1) })}
            onClick={() => write(removeAt(levels, index))}
          />
        </div>
      ))}
      <AddButton onClick={() => write([...levels, { label: String(levels.length) }])}>{t("Add level")}</AddButton>
    </div>
  );
}

interface QuestionCardProps extends QuestionEditorProps {
  index: number;
  onRemove: () => void;
}

const blankDecisionsQuestion = (existing: DecisionsFormQuestion[]): DecisionsFormQuestion => ({
  type: "predicate",
  name: freeName(
    existing.map((q) => q.name ?? ""),
    "question",
  ),
  instructions: "",
});

function QuestionCard({ question, index, onChange, onRemove }: QuestionCardProps) {
  const { t } = useLanguage();
  const instructionsId = useId();

  return (
    <fieldset
      aria-label={t("Question {value}", { value: String(index + 1) })}
      className="grid gap-3 rounded-lg border bg-muted/30 p-3"
    >
      <QuestionCardHeader index={index} onRemove={onRemove} removeLabel={`Remove question ${index + 1}`} />
      <div className="grid gap-3 sm:grid-cols-[minmax(0,1fr)_10rem]">
        <div className="grid gap-1.5">
          <span className="text-sm font-medium">{t("Name")}</span>
          <Input
            aria-label={t("Question name")}
            value={question.name ?? ""}
            placeholder={t("Optional question name")}
            onChange={(event) => onChange({ ...question, name: event.target.value })}
          />
        </div>
        <div className="grid gap-1.5">
          <span className="text-sm font-medium">{t("Answer type")}</span>
          <Select
            value={question.type}
            onValueChange={(type) => {
              const next = QUESTION_TYPES.find((value) => value === type);
              if (next) onChange(retypeDecisionQuestion(question, next));
            }}
          >
            <SelectTrigger className="w-full" aria-label={t("Answer type")}>
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
        </div>
      </div>
      <div className="grid gap-1.5">
        <Label htmlFor={instructionsId}>{t("Instructions")}</Label>
        <Textarea
          id={instructionsId}
          rows={2}
          value={question.instructions ?? ""}
          placeholder={t("What should the model decide?")}
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
  onReset: () => void;
}

export default function OpenAIDecisionsForm({ value, onChange, validation, onReset }: OpenAIDecisionsFormProps) {
  const { t } = useLanguage();
  const inputId = useId();
  const safetyId = useId();
  const form = useMemo(() => readDecisionsForm(value), [value]);

  if (form === undefined) {
    return (
      <FormFrame validation={validation}>
        <p className="text-sm text-muted-foreground">{t("This request has a shape the form can't show.")}</p>
        <Button variant="outline" className="w-fit" onClick={onReset}>
          {t("Reset to example")}
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
              {t("Input contains messages. Form edits keep them unchanged.")}
            </p>
          </>
        ) : (
          <>
            <Label htmlFor={inputId}>{t("Input")}</Label>
            <Textarea
              id={inputId}
              rows={4}
              value={form.input ?? ""}
              placeholder={t("Paste the text the model should decide about")}
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
      <AddButton onClick={() => writeQuestions([...questions, blankDecisionsQuestion(questions)])}>
        {t("Add question")}
      </AddButton>
      <div className="grid gap-1.5">
        <Label htmlFor={safetyId}>{t("Safety identifier")}</Label>
        <Input
          id={safetyId}
          value={form.safety_identifier ?? ""}
          placeholder={t("Optional")}
          onChange={(event) => write({ ...form, safety_identifier: event.target.value || undefined })}
        />
      </div>
    </FormFrame>
  );
}
