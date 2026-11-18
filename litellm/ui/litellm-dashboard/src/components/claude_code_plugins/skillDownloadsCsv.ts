import Papa from "papaparse";
import type { PluginDownloadEvent } from "./types";

export const buildSkillDownloadsCsv = (downloads: readonly PluginDownloadEvent[]): string =>
  Papa.unparse(
    {
      fields: ["downloaded_at", "user_id", "user_email", "key", "team", "version", "authenticated", "download_id"],
      data: downloads.map((event) => [
        new Date(event.downloaded_at).toISOString(),
        event.user_id,
        event.user_email,
        event.key_alias,
        event.team_alias || event.team_id,
        event.version,
        event.authenticated,
        event.id,
      ]),
    },
    { escapeFormulae: true },
  );
