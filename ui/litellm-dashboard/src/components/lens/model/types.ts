import type { components } from "@/lib/http/schema";

export type Lens = components["schemas"]["Lens"];

export type Settings = components["schemas"]["LensSettings"];

export type LensList = components["schemas"]["LensList"];

export type Finding = components["schemas"]["Finding"];

export type Sample = components["schemas"]["Sample"];

export type WorkerCreated = components["schemas"]["WorkerCreated"];

export type Job = components["schemas"]["Job"];

export type IssueBrief = NonNullable<Finding["brief"]>;

export type ActivitySelection = Pick<Settings, "source"> &
  Partial<
    Pick<
      Settings,
      | "service"
      | "agent_name"
      | "filters"
      | "lookback_hours"
      | "sample_percent"
      | "sample_size"
      | "team_id"
      | "execution_ids"
    >
  >;

export type Worker = LensList["workers"][number];
