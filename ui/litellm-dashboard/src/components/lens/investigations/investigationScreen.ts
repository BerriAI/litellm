import type { Lens, LensList, Settings } from "../model/types";
import type { LensDialog } from "../route";

export type SetupMode = "new" | "edit" | "duplicate";

export interface ScreenInput {
  readonly list: { readonly data: LensList | undefined; readonly error: Error | null };
  readonly lensId: string | null;
  readonly dialog: LensDialog | null;
  readonly target: string | null;
}

export type SetupScreen =
  | { readonly kind: "setup"; readonly mode: "new" }
  | { readonly kind: "setup"; readonly mode: "edit" | "duplicate"; readonly lens: Lens; readonly initial: Settings };

export type Screen =
  | { readonly kind: "loading" }
  | { readonly kind: "failed"; readonly error: Error }
  | { readonly kind: "welcome" }
  | { readonly kind: "list"; readonly lenses: readonly Lens[]; readonly lens?: Lens }
  | { readonly kind: "missing" }
  | SetupScreen;

const isSetupMode = (dialog: LensDialog | null): dialog is SetupMode =>
  dialog === "new" || dialog === "edit" || dialog === "duplicate";

function setupScreen(mode: SetupMode, lens: Lens | undefined): SetupScreen | undefined {
  if (mode === "new") return { kind: "setup", mode };
  if (!lens) return undefined;
  const initial =
    mode === "duplicate" ? { ...lens.settings, name: `${lens.settings.name} copy`, enabled: false } : lens.settings;
  return { kind: "setup", mode, lens, initial };
}

export function investigationScreen({ list, lensId, dialog, target }: ScreenInput): Screen {
  const lenses = list.data?.lenses ?? [];
  const selected = lenses.find((lens) => lens.id === lensId);
  const setup = isSetupMode(dialog)
    ? setupScreen(dialog, target ? lenses.find((lens) => lens.id === target) : selected)
    : undefined;
  if (setup) return setup;
  if (!list.data) return list.error ? { kind: "failed", error: list.error } : { kind: "loading" };
  if (lensId && !selected) return { kind: "missing" };
  if (lenses.length === 0) return { kind: "welcome" };
  return {
    kind: "list",
    lenses: [...lenses].sort((a, b) => Date.parse(b.created_at) - Date.parse(a.created_at)),
    ...(selected ? { lens: selected } : {}),
  };
}
