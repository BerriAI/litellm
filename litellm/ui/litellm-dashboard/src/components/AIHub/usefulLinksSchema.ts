import { z } from "zod";

const linkDraft = z.object({
  displayName: z.string(),
  url: z.string(),
});

const editDraft = linkDraft.extend({
  id: z.string(),
  url: z.unknown(),
  index: z.unknown().optional(),
});

export type UsefulLink = z.output<typeof editDraft>;

export const createUsefulLinksSchemas = (links: readonly Pick<UsefulLink, "displayName" | "id">[]) => {
  const linkError = ({ displayName, url }: Pick<UsefulLink, "displayName" | "url">, editingId?: string) => {
    try {
      new URL(url as string);
    } catch {
      return "invalidUrl";
    }
    if (links.some((link) => link.id !== editingId && link.displayName === displayName)) {
      return "duplicateName";
    }
    return undefined;
  };

  const addSchema = linkDraft.superRefine((draft, ctx) => {
    const message = !draft.url || !draft.displayName ? "missingFields" : linkError(draft);
    if (message) ctx.addIssue({ code: "custom", message });
  });
  const editSchema = editDraft.superRefine((draft, ctx) => {
    const message = linkError(draft, draft.id);
    if (message) ctx.addIssue({ code: "custom", message });
  });

  return { addSchema, editSchema: z.object({ link: editSchema.nullable() }) };
};
