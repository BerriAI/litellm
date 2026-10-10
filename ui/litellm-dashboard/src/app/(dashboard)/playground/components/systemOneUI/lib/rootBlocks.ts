export type RootBlockKey = "state" | "questions";

export const ROOT_BLOCK_STYLES: Readonly<Record<RootBlockKey, { accent: string }>> = {
  state: { accent: "border-l-green-500/60" },
  questions: { accent: "border-l-orange-500/60" },
};
