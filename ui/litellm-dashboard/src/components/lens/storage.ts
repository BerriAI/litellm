import { z } from "zod";
import { storageKey } from "@/lib/storage";

export const FINDING_PANEL_WIDTH_KEY = "litellm.lens.findingPanelWidth";
export const LENS_INTRO_DISMISSED = storageKey("local", "lens.intro.dismissed", z.boolean(), false);
export const LENS_INTRO_SEEN = storageKey("session", "lens.intro.seen", z.boolean(), false);
