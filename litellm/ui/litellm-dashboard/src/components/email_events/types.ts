import { EmailEvent } from "../../types";

export interface EmailEventSetting {
  event: EmailEvent;
  enabled: boolean;
}

export interface EmailEventSettingsUpdateRequest {
  settings: EmailEventSetting[];
  model_deprecation_email_recipient?: string | null;
}

export interface EmailEventSettingsResponse {
  settings: EmailEventSetting[];
  model_deprecation_email_recipient?: string | null;
}
