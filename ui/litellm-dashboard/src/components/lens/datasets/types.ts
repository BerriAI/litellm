import type { paths } from "@/lib/http/schema";

type Json<T> = T extends { content: { "application/json": infer Body } } ? Body : never;

export type Dataset = Json<paths["/lens/datasets/{dataset_id}"]["get"]["responses"][200]>;
export type DatasetSummary = Json<paths["/lens/datasets"]["get"]["responses"][200]>[number];
export type DatasetCreate = Json<paths["/lens/datasets"]["post"]["requestBody"]>;
export type BuildRequest = Json<paths["/lens/datasets/build"]["post"]["requestBody"]>;
export type BuildResult = Json<paths["/lens/datasets/build"]["post"]["responses"][200]>;
export type BuildSource = BuildRequest["sources"][number];
export type RevisionSave = Json<paths["/lens/datasets/{dataset_id}/revisions"]["post"]["requestBody"]>;
export type EvalCases = Json<paths["/lens/datasets/{dataset_id}/revisions/{revision}/cases"]["get"]["responses"][200]>;
export type DatasetCase = Dataset["cases"][number];
export type DatasetMessage = DatasetCase["messages"][number];
export type DatasetToolCall = DatasetCase["tool_calls"][number];
export type SkippedCase = BuildResult["skipped"][number];
export type SkipReason = SkippedCase["reason"];
export type CaseSource = DatasetCase["source"];
