import { createContext, useContext, useEffect, useMemo } from "react";
import { get, type Validate } from "react-hook-form";
import { z } from "zod";

import type { AgentFormFieldValue, AgentFormValues } from "./AgentFormKit";

export interface AgentFieldRules {
  required?: string | false;
  pattern?: { value: RegExp; message: string };
  validate?: Validate<AgentFormFieldValue, AgentFormValues>;
}

type RegisterRules = (name: string, rules: AgentFieldRules) => () => void;
export const AgentFieldRulesContext = createContext<RegisterRules>(() => () => undefined);

export const useAgentFieldRules = (name: string, rules: AgentFieldRules | undefined) => {
  const registerRules = useContext(AgentFieldRulesContext);
  useEffect(() => (rules ? registerRules(name, rules) : undefined), [name, registerRules, rules]);
};

const validateField = (rules: AgentFieldRules, value: AgentFormFieldValue, values: AgentFormValues) => {
  const empty = value == null || value === "" || (Array.isArray(value) && value.length === 0);
  if (rules.required && (empty || value === false)) return rules.required;
  if (rules.pattern && typeof value === "string" && value !== "" && !value.match(rules.pattern.value)) {
    return rules.pattern.message;
  }
  return rules.validate?.(value, values);
};

export const useAgentFormSchema = () =>
  useMemo(() => {
    const fields = new Map<string, AgentFieldRules>();
    const registerRules: RegisterRules = (name, rules) => {
      fields.set(name, rules);
      return () => {
        if (fields.get(name) === rules) fields.delete(name);
      };
    };
    const schema = z.custom<AgentFormValues>().superRefine(async (values, context) => {
      for (const [name, rules] of Array.from(fields)) {
        const value = get(values, name) as AgentFormFieldValue;
        const message = await validateField(rules, value, values);
        if (typeof message === "string") {
          context.addIssue({ code: "custom", path: name.split("."), message });
        } else if (message === false || Array.isArray(message)) {
          context.addIssue({ code: "custom", path: name.split("."), message: "" });
        }
      }
    });
    return { schema, registerRules };
  }, []);
