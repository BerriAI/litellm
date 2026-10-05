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

export type Review = components["schemas"]["Review"];

export type InFlight = components["schemas"]["InFlight"];

export type ReviewVerdict = components["schemas"]["ReviewVerdict"];

export interface RunWindow {
  agent_name?: string;
  start?: string;
  end?: string;
  lookback_hours?: number;
}

export interface AnalysisModelInfo {
  model_group: string;
  providers: string[];
  mode?: string | null;
  supported_openai_params?: string[] | null;
}
