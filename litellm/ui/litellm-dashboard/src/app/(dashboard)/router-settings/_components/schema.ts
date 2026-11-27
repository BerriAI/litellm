import { z } from "zod";
import type { generalSettingsItem } from "./general_settings";

export class GeneralSettingsRows {
  // Preserve the raw values and row order through React Hook Form's object cloning and comparison.
  readonly #rows: generalSettingsItem[];

  constructor(rows: generalSettingsItem[]) {
    this.#rows = rows;
  }

  get items(): generalSettingsItem[] {
    return this.#rows;
  }
}

export const generalSettingsSchema = z.object({ rows: z.instanceof(GeneralSettingsRows) });
