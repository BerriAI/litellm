"use client";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import type { ParseKeys } from "i18next";
import { useId, useMemo } from "react";
import { useTranslation } from "react-i18next";
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
const TYPE_LABELS: Record<DecisionsQuestionType, ParseKeys> = {
  predicate: "playground.systemOne.openAIDecisionsForm.types.predicate",
  choice: "playground.systemOne.openAIDecisionsForm.types.choice",
  score: "playground.systemOne.openAIDecisionsForm.types.score",
};

interface QuestionEditorProps<Q extends DecisionsFormQuestion = DecisionsFormQuestion> {
  question: Q;
  onChange: (question: DecisionsFormQuestion) => void;
}

function Choices({ question, onChange }: QuestionEditorProps<Extract<DecisionsFormQuestion, { type: "choice" }>>) {
  const { t } = useTranslation();
  const choices = question.choices ?? [];
  const write = (next: typeof choices) => onChange({ ...question, choices: next });

  return (
    <div className="grid gap-2">
      <span className="text-sm font-medium">{t("playground.systemOne.openAIDecisionsForm.choices")}</span>
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
              <SelectTrigger
                className="w-40"
                aria-label={t("playground.systemOne.openAIDecisionsForm.choiceValueType", { number: index + 1 })}
              >
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="text">{t("playground.systemOne.openAIDecisionsForm.valueTypes.text")}</SelectItem>
                <SelectItem value="true">{t("playground.systemOne.openAIDecisionsForm.valueTypes.true")}</SelectItem>
                <SelectItem value="false">{t("playground.systemOne.openAIDecisionsForm.valueTypes.false")}</SelectItem>
              </SelectContent>
            </Select>
            {typeof choice.value === "string" && (
              <Input
                aria-label={t("playground.systemOne.openAIDecisionsForm.choiceValue", { number: index + 1 })}
                value={choice.value}
                onChange={(event) => write(replaceAt(choices, index, { ...choice, value: event.target.value }))}
              />
            )}
            <RemoveButton
              label={t("playground.systemOne.openAIDecisionsForm.removeChoice", { number: index + 1 })}
              onClick={() => write(removeAt(choices, index))}
            />
          </div>
          <Input
            aria-label={t("playground.systemOne.openAIDecisionsForm.choiceDescription", { number: index + 1 })}
            value={choice.description ?? ""}
            placeholder={t("playground.systemOne.openAIDecisionsForm.optionalDescription")}
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
        {t("playground.systemOne.openAIDecisionsForm.addChoice")}
      </AddButton>
    </div>
  );
}

function Levels({ question, onChange }: QuestionEditorProps<Extract<DecisionsFormQuestion, { type: "score" }>>) {
  const { t } = useTranslation();
  const levels = question.levels ?? [];
  const write = (next: typeof levels) => onChange({ ...question, levels: next });

  return (
    <div className="grid gap-2">
      <span className="text-sm font-medium">{t("playground.systemOne.openAIDecisionsForm.levels")}</span>
      {levels.map((level, index) => (
        <div key={index} className="flex items-center gap-2">
          <Input
            aria-label={t("playground.systemOne.openAIDecisionsForm.levelLabel", { number: index + 1 })}
            value={level.label}
            placeholder={t("playground.systemOne.openAIDecisionsForm.labelPlaceholder")}
            onChange={(event) => write(replaceAt(levels, index, { ...level, label: event.target.value }))}
          />
          <Input
            aria-label={t("playground.systemOne.openAIDecisionsForm.levelDescription", { number: index + 1 })}
            value={level.description ?? ""}
            placeholder={t("playground.systemOne.openAIDecisionsForm.optionalDescription")}
            onChange={(event) => write(replaceAt(levels, index, { ...level, description: event.target.value }))}
          />
          <RemoveButton
            label={t("playground.systemOne.openAIDecisionsForm.removeLevel", { number: index + 1 })}
            onClick={() => write(removeAt(levels, index))}
          />
        </div>
      ))}
      <AddButton onClick={() => write([...levels, { label: String(levels.length) }])}>
        {t("playground.systemOne.openAIDecisionsForm.addLevel")}
      </AddButton>
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
  const { t } = useTranslation();
  const instructionsId = useId();

  return (
    <fieldset
      aria-label={t("playground.systemOne.openAIDecisionsForm.question", { number: index + 1 })}
      className="grid gap-3 rounded-lg border bg-muted/30 p-3"
    >
      <QuestionCardHeader
        index={index}
        onRemove={onRemove}
        removeLabel={t("playground.systemOne.openAIDecisionsForm.removeQuestion", { number: index + 1 })}
      />
      <div className="grid gap-3 sm:grid-cols-[minmax(0,1fr)_10rem]">
        <div className="grid gap-1.5">
          <span className="text-sm font-medium">{t("playground.systemOne.openAIDecisionsForm.name")}</span>
          <Input
            aria-label={t("playground.systemOne.openAIDecisionsForm.questionName")}
            value={question.name ?? ""}
            placeholder={t("playground.systemOne.openAIDecisionsForm.questionNamePlaceholder")}
            onChange={(event) => onChange({ ...question, name: event.target.value })}
          />
        </div>
        <div className="grid gap-1.5">
          <span className="text-sm font-medium">{t("playground.systemOne.openAIDecisionsForm.answerType")}</span>
          <Select
            value={question.type}
            onValueChange={(type) => {
              const next = QUESTION_TYPES.find((value) => value === type);
              if (next) onChange(retypeDecisionQuestion(question, next));
            }}
          >
            <SelectTrigger className="w-full" aria-label={t("playground.systemOne.openAIDecisionsForm.answerType")}>
              <SelectValue>{t(TYPE_LABELS[question.type])}</SelectValue>
            </SelectTrigger>
            <SelectContent>
              {QUESTION_TYPES.map((type) => (
                <SelectItem key={type} value={type}>
                  {t(TYPE_LABELS[type])}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      </div>
      <div className="grid gap-1.5">
        <Label htmlFor={instructionsId}>{t("playground.systemOne.openAIDecisionsForm.instructions")}</Label>
        <Textarea
          id={instructionsId}
          rows={2}
          value={question.instructions ?? ""}
          placeholder={t("playground.systemOne.openAIDecisionsForm.instructionsPlaceholder")}
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
  const { t } = useTranslation();
  const inputId = useId();
  const safetyId = useId();
  const form = useMemo(() => readDecisionsForm(value), [value]);

  if (form === undefined) {
    return (
      <FormFrame validation={validation}>
        <p className="text-sm text-muted-foreground">{t("playground.systemOne.openAIDecisionsForm.unreadable")}</p>
        <Button variant="outline" className="w-fit" onClick={onReset}>
          {t("playground.systemOne.openAIDecisionsForm.resetToExample")}
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
              {t("playground.systemOne.openAIDecisionsForm.inputMessages")}
            </p>
          </>
        ) : (
          <>
            <Label htmlFor={inputId}>{t("playground.systemOne.openAIDecisionsForm.input")}</Label>
            <Textarea
              id={inputId}
              rows={4}
              value={form.input ?? ""}
              placeholder={t("playground.systemOne.openAIDecisionsForm.inputPlaceholder")}
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
        {t("playground.systemOne.openAIDecisionsForm.addQuestion")}
      </AddButton>
      <div className="grid gap-1.5">
        <Label htmlFor={safetyId}>{t("playground.systemOne.openAIDecisionsForm.safetyIdentifier")}</Label>
        <Input
          id={safetyId}
          value={form.safety_identifier ?? ""}
          placeholder={t("playground.systemOne.openAIDecisionsForm.safetyIdentifierPlaceholder")}
          onChange={(event) => write({ ...form, safety_identifier: event.target.value || undefined })}
        />
      </div>
    </FormFrame>
  );
}
