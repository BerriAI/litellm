import Papa from "papaparse";
import type { PluginDownloadEvent } from "./types";

export const buildSkillDownloadsCsv = (downloads: readonly PluginDownloadEvent[]): string =>
  Papa.unparse(
    {
      fields: ["download_id", "downloaded_at", "user_email", "user_id", "team", "key", "version"],
      data: downloads.map((event) => [
        event.id,
        new Date(event.downloaded_at).toISOString(),
        event.user_email,
        event.user_id,
        event.team_alias || event.team_id,
        event.key_alias,
        event.version,
      ]),
    },
    { escapeFormulae: true },
  );
