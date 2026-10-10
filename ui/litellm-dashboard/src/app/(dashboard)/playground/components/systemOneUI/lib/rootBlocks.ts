export type RootBlockKey = "model" | "state" | "questions";

export interface RootBlock {
  readonly key: RootBlockKey;
  readonly startLine: number;
  readonly endLine: number;
}

export const ROOT_BLOCK_STYLES: Readonly<Record<RootBlockKey, { band: string; accent: string }>> = {
  model: { band: "bg-blue-500/10 before:bg-blue-500/60", accent: "border-l-blue-500/60" },
  state: { band: "bg-green-500/10 before:bg-green-500/60", accent: "border-l-green-500/60" },
  questions: { band: "bg-orange-500/10 before:bg-orange-500/60", accent: "border-l-orange-500/60" },
};
