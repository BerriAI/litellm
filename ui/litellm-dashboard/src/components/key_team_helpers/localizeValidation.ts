import type { FieldValues, Validate } from "react-hook-form";
import type { Translate } from "@/i18n/language";

// 仅供本地受控校验规则使用，服务端错误和业务内容不经过此层
export const localizeValidation = <TValue, TFormValues extends FieldValues>(
  rules: Readonly<Record<string, Validate<TValue, TFormValues>>>,
  t: Translate,
): Record<string, Validate<TValue, TFormValues>> =>
  Object.fromEntries(
    Object.entries(rules).map(([name, validate]) => [
      name,
      async (value: TValue, values: TFormValues) => {
        const result = await validate(value, values);
        if (typeof result === "string") return t(result);
        if (Array.isArray(result)) return result.map((message) => t(message));
        return result;
      },
    ]),
  );
