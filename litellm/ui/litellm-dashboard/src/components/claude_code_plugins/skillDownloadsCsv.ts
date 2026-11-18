import Papa from "papaparse";
import type { PluginDownloadEvent } from "./types";

export const buildSkillDownloadsCsv = (downloads: readonly PluginDownloadEvent[]): string =>
  Papa.unparse(
    {
      fields: [
        "downloaded_at",
        "downloader",
        "key",
        "team",
        "version",
        "authenticated",
        "user_id",
        "user_email",
        "team_id",
        "id",
        "digest",
      ],
      data: downloads.map((event) => [
        new Date(event.downloaded_at).toISOString(),
        event.authenticated ? event.user_email || event.user_id || "" : "",
        event.key_alias,
        event.team_alias || event.team_id,
        event.version,
        event.authenticated,
        event.user_id,
        event.user_email,
        event.team_id,
        event.id,
        event.digest,
      ]),
    },
    { escapeFormulae: true },
  );
