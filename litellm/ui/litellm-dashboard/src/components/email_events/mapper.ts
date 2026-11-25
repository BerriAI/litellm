import type { EmailEventSettingsFields } from "./schema";
import type { EmailEventSettingsResponse, EmailEventSettingsUpdateRequest } from "./types";

export const toEmailEventSettingsFields = ({ settings }: EmailEventSettingsResponse): EmailEventSettingsFields => ({
  settings,
});

export const toEmailEventSettingsUpdate = ({
  settings,
}: EmailEventSettingsFields): EmailEventSettingsUpdateRequest => ({
  settings,
});
