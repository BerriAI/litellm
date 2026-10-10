import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import { CircleAlert, Plus, X } from "lucide-react";
import { useId } from "react";
import {
  blankChoice,
  blankLevel,
  isNativeEndpoint,
  newQuestion,
  withQuestionType,
  type DecisionsForm as DecisionsFormValue,
  type FormQuestion,
  type FormQuestionType,
} from "./lib/form";
import type { DecisionEndpoint } from "./lib/schemas";

interface DecisionsFormProps {
  form: DecisionsFormValue;
  endpoint: DecisionEndpoint;
  issues: string[];
  onChange: (form: DecisionsFormValue) => void;
}

const QUESTION_TYPES: FormQuestionType[] = ["choice", "yes_no", "score"];
const typeLabel = (type: FormQuestionType, native: boolean): string => {
  switch (type) {
    case "choice":
      return "Choice";
    case "yes_no":
      return native ? "Noul (yes / no)" : "Predicate (yes / no)";
    case "score":
      return "Score";
  }
};
const isQuestionType = (value: string): value is FormQuestionType => QUESTION_TYPES.some((type) => type === value);
const replaceAt = <T,>(items: T[], index: number, item: T): T[] =>
  items.map((current, i) => (i === index ? item : current));
const removeAt = <T,>(items: T[], index: number): T[] => items.filter((_, i) => i !== index);

export default function DecisionsForm({ form, endpoint, issues, onChange }: DecisionsFormProps) {
  const native = isNativeEndpoint(endpoint);
  const contextId = useId();
  const contextLabel = native ? "State" : "Input";

  function updateQuestion(index: number, question: FormQuestion) {
    onChange({ ...form, questions: replaceAt(form.questions, index, question) });
  }

  return (
    <div className="flex min-h-0 flex-1 flex-col rounded-md border">
      <div className="grid min-h-0 flex-1 content-start gap-4 overflow-auto p-4">
        <div className="grid gap-2">
          <div className="flex items-baseline justify-between">
            <Label htmlFor={contextId}>{contextLabel}</Label>
            <span className="text-xs text-muted-foreground">what the questions are about</span>
          </div>
          <Textarea
            id={contextId}
            value={form.context}
            onChange={(event) => onChange({ ...form, context: event.target.value })}
            placeholder={
              native ? "Paste the state the model should decide on" : "Paste the text the model should decide on"
            }
            className="min-h-24"
          />
        </div>
        {form.questions.map((question, index) => (
          <QuestionCard
            key={index}
            index={index}
            question={question}
            native={native}
            onChange={(next) => updateQuestion(index, next)}
            onRemove={() => onChange({ ...form, questions: removeAt(form.questions, index) })}
          />
        ))}
        <Button
          variant="outline"
          className="w-full border-dashed"
          onClick={() => onChange({ ...form, questions: [...form.questions, newQuestion("choice")] })}
        >
          <Plus />
          Add question
        </Button>
      </div>
      {issues.length > 0 && (
        <ul aria-label="Form validation issues" className="grid max-h-36 gap-1 overflow-auto border-t px-3 py-2">
          {issues.map((issue) => (
            <li key={issue} className="flex items-start gap-1.5 text-xs">
              <CircleAlert className="mt-0.5 size-3.5 shrink-0 text-destructive" />
              <span>{issue}</span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

interface QuestionCardProps {
  index: number;
  question: FormQuestion;
  native: boolean;
  onChange: (question: FormQuestion) => void;
  onRemove: () => void;
}

function QuestionCard({ index, question, native, onChange, onRemove }: QuestionCardProps) {
  const id = useId();
  const n = index + 1;
  return (
    <fieldset className="grid gap-3 rounded-md border p-4" aria-label={`Question ${n}`}>
      <div className="grid grid-cols-[12rem_1fr_auto] items-end gap-3">
        <div className="grid gap-2">
          <Label htmlFor={`${id}-type`}>Type</Label>
          <Select
            value={question.type}
            onValueChange={(value) => {
              if (value !== null && isQuestionType(value)) {
                onChange(withQuestionType(question, value));
              }
            }}
          >
            <SelectTrigger id={`${id}-type`} aria-label={`Question ${n} type`} className="w-full">
              <SelectValue>{typeLabel(question.type, native)}</SelectValue>
            </SelectTrigger>
            <SelectContent>
              {QUESTION_TYPES.map((type) => (
                <SelectItem key={type} value={type}>
                  {typeLabel(type, native)}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
        <div className="grid gap-2">
          <div className="flex items-baseline justify-between">
            <Label htmlFor={`${id}-name`}>Name</Label>
            <span className="text-xs text-muted-foreground">{native ? "required" : "optional"}</span>
          </div>
          <Input
            id={`${id}-name`}
            aria-label={`Question ${n} name`}
            className="font-mono"
            value={question.name}
            onChange={(event) => onChange({ ...question, name: event.target.value })}
            placeholder={native ? `question_${n}` : "optional"}
          />
        </div>
        <Button variant="ghost" size="icon" aria-label={`Remove question ${n}`} onClick={onRemove}>
          <X />
        </Button>
      </div>
      <div className="grid gap-2">
        <Label htmlFor={`${id}-instructions`}>Instructions</Label>
        <Input
          id={`${id}-instructions`}
          aria-label={`Question ${n} instructions`}
          value={question.instructions}
          onChange={(event) => onChange({ ...question, instructions: event.target.value })}
          placeholder="What should the model decide?"
        />
      </div>
      {question.type === "choice" && (
        <div className="grid gap-2">
          <div className="flex items-baseline justify-between">
            <span className="text-sm font-medium">Choices</span>
            <span className="text-xs text-muted-foreground">value · description</span>
          </div>
          {question.choices.map((choice, i) => (
            <div key={i} className="grid grid-cols-[12rem_1fr_auto] items-start gap-2">
              <Input
                aria-label={`Question ${n} choice ${i + 1} value`}
                className="font-mono"
                value={choice.value}
                onChange={(event) =>
                  onChange({
                    ...question,
                    choices: replaceAt(question.choices, i, { ...choice, value: event.target.value }),
                  })
                }
                placeholder="value"
              />
              <Textarea
                aria-label={`Question ${n} choice ${i + 1} description`}
                rows={1}
                className="min-h-9 py-1.5"
                value={choice.description}
                onChange={(event) =>
                  onChange({
                    ...question,
                    choices: replaceAt(question.choices, i, { ...choice, description: event.target.value }),
                  })
                }
                placeholder="description (optional)"
              />
              <Button
                variant="ghost"
                size="icon"
                aria-label={`Remove choice ${i + 1} from question ${n}`}
                onClick={() => onChange({ ...question, choices: removeAt(question.choices, i) })}
              >
                <X />
              </Button>
            </div>
          ))}
          <Button
            variant="link"
            className="w-fit px-0"
            onClick={() => onChange({ ...question, choices: [...question.choices, blankChoice()] })}
          >
            <Plus />
            Add choice
          </Button>
        </div>
      )}
      {question.type === "score" && (
        <div className="grid gap-2">
          <div className="flex items-baseline justify-between">
            <span className="text-sm font-medium">Levels</span>
            <span className="text-xs text-muted-foreground">
              {native ? "lowest first · label" : "lowest first · label · description"}
            </span>
          </div>
          {question.levels.map((level, i) => (
            <div
              key={i}
              className={
                native
                  ? "grid grid-cols-[2rem_1fr_auto] items-start gap-2"
                  : "grid grid-cols-[2rem_1fr_1fr_auto] items-start gap-2"
              }
            >
              <span className="flex h-9 items-center justify-center rounded-md bg-muted text-sm text-muted-foreground">
                {i}
              </span>
              <Input
                aria-label={`Question ${n} level ${i + 1} label`}
                value={level.label}
                onChange={(event) =>
                  onChange({
                    ...question,
                    levels: replaceAt(question.levels, i, { ...level, label: event.target.value }),
                  })
                }
                placeholder="label"
              />
              {!native && (
                <Textarea
                  aria-label={`Question ${n} level ${i + 1} description`}
                  rows={1}
                  className="min-h-9 py-1.5"
                  value={level.description}
                  onChange={(event) =>
                    onChange({
                      ...question,
                      levels: replaceAt(question.levels, i, { ...level, description: event.target.value }),
                    })
                  }
                  placeholder="description (optional)"
                />
              )}
              <Button
                variant="ghost"
                size="icon"
                aria-label={`Remove level ${i + 1} from question ${n}`}
                onClick={() => onChange({ ...question, levels: removeAt(question.levels, i) })}
              >
                <X />
              </Button>
            </div>
          ))}
          <Button
            variant="link"
            className="w-fit px-0"
            onClick={() => onChange({ ...question, levels: [...question.levels, blankLevel()] })}
          >
            <Plus />
            Add level
          </Button>
        </div>
      )}
      {question.type === "yes_no" && !native && (
        <p className="text-xs text-muted-foreground">Predicate questions answer yes or no, nothing else to fill in</p>
      )}
      {question.type === "yes_no" && native && (
        <div className="grid grid-cols-2 gap-2">
          <div className="grid gap-2">
            <Label htmlFor={`${id}-yes`}>Yes means</Label>
            <Input
              id={`${id}-yes`}
              value={question.yes}
              onChange={(event) => onChange({ ...question, yes: event.target.value })}
              placeholder="optional"
            />
          </div>
          <div className="grid gap-2">
            <Label htmlFor={`${id}-no`}>No means</Label>
            <Input
              id={`${id}-no`}
              value={question.no}
              onChange={(event) => onChange({ ...question, no: event.target.value })}
              placeholder="optional"
            />
          </div>
        </div>
      )}
    </fieldset>
  );
}
