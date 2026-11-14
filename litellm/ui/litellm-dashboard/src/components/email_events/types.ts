import { EmailEvent } from "../../types";

export interface EmailEventSetting {
  event: EmailEvent;
  enabled: boolean;
}

export interface EmailEventSettingsUpdateRequest {
  settings: EmailEventSetting[];
}

export interface EmailEventSettingsResponse {
  settings: EmailEventSetting[];
}
