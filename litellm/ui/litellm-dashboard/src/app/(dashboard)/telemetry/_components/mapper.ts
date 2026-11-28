import type { z } from "zod";
import type { telemetryUpdateSchema } from "./schema";

type TelemetryUpdate = z.infer<typeof telemetryUpdateSchema>;

export const toTelemetryUpdate = (
  groups: Iterable<TelemetryUpdate["groups"][number]>,
  destination: TelemetryUpdate["destination"],
): TelemetryUpdate => ({ groups: [...groups], destination });
