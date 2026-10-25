import { z } from "zod";
import { englishTranslate, type Translate } from "@/i18n/language";

const scalar = z.union([z.string(), z.number(), z.null()]).optional();
const isBlank = (value: string | number | null | undefined): boolean => value == null || String(value).trim() === "";

// 定价换算和校验独立于表单布局；纯函数导出保留英文回退兼容既有调用。
export const perMillionSchema = (t: Translate) =>
  scalar
    .refine(
      (perMillion) => isBlank(perMillion) || (Number.isFinite(Number(perMillion)) && Number(perMillion) >= 0),
      t("Enter a price of 0 or more"),
    )
    .transform((perMillion) => (isBlank(perMillion) ? null : Number(perMillion) / 1_000_000));

export const perMillionToPerToken = perMillionSchema(englishTranslate);
