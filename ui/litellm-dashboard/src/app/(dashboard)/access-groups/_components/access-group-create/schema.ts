import { z } from "zod";
import { englishTranslate, type Translate } from "@/i18n/language";

// 校验消息随界面语言创建，提交字段和值保持不变。
export const buildAccessGroupCreateSchema = (t: Translate = englishTranslate) =>
  z.object({
    name: z.string().refine((value) => value.trim() !== "", t("Please enter the access group name")),
    description: z.string(),
    modelIds: z.array(z.string()),
    mcpServerIds: z.array(z.string()),
    mcpToolPermissions: z.record(z.string(), z.array(z.string())),
    agentIds: z.array(z.string()),
    skillNames: z.array(z.string()),
  });

export const accessGroupCreateSchema = buildAccessGroupCreateSchema();

export type AccessGroupCreateFormValues = z.output<typeof accessGroupCreateSchema>;
