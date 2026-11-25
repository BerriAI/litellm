import type { TelemetryGroup, TelemetrySettingsUpdate } from "@/app/(dashboard)/hooks/telemetry/useTelemetrySettings";
import { telemetryUpdateSchema } from "./schema";

export const toTelemetryUpdate = (groups: Iterable<TelemetryGroup>, destination: unknown): TelemetrySettingsUpdate =>
  telemetryUpdateSchema.parse({ groups: [...groups], destination });
