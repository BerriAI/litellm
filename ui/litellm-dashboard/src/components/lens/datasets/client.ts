import type { Client } from "openapi-fetch";

import type { ApiClient } from "@/lib/http/client";
import { getAuthHeaderName } from "@/lib/http/runtime";
import type { paths } from "@/lib/http/schema";

import type {
  BuildRequest,
  BuildResult,
  Dataset,
  DatasetCreate,
  DatasetSummary,
  EvalCases,
  RevisionSave,
} from "./types";

export interface DatasetsApi {
  list(): Promise<readonly DatasetSummary[]>;
  get(datasetId: string, revision?: number): Promise<Dataset>;
  create(body: DatasetCreate): Promise<Dataset>;
  build(body: BuildRequest): Promise<BuildResult>;
  saveRevision(datasetId: string, body: RevisionSave): Promise<Dataset>;
  exportJsonl(datasetId: string, revision?: number): Promise<Blob>;
  evalCases(datasetId: string, revision: number): Promise<EvalCases>;
}

async function required<T>(request: Promise<{ data?: T }>): Promise<T> {
  const { data } = await request;
  if (data === undefined) throw new Error("The proxy returned an empty response");
  return data;
}

export const datasetExportPath = (datasetId: string): string =>
  `/lens/datasets/${encodeURIComponent(datasetId)}/export`;

export function liveDatasetsApi(client: Client<paths>, apiClient: ApiClient, accessToken: string): DatasetsApi {
  const headers = { [getAuthHeaderName()]: `Bearer ${accessToken}` };
  return {
    list: () => required(client.GET("/lens/datasets", { headers })),
    get: (datasetId, revision) =>
      required(
        client.GET("/lens/datasets/{dataset_id}", {
          headers,
          params: { path: { dataset_id: datasetId }, query: { revision } },
        }),
      ),
    create: (body) => required(client.POST("/lens/datasets", { headers, body })),
    build: (body) => required(client.POST("/lens/datasets/build", { headers, body })),
    saveRevision: (datasetId, body) =>
      required(
        client.POST("/lens/datasets/{dataset_id}/revisions", {
          headers,
          params: { path: { dataset_id: datasetId } },
          body,
        }),
      ),
    exportJsonl: (datasetId, revision) =>
      apiClient.getBlob(datasetExportPath(datasetId), { accessToken, query: { revision } }),
    evalCases: (datasetId, revision) =>
      required(
        client.GET("/lens/datasets/{dataset_id}/revisions/{revision}/cases", {
          headers,
          params: { path: { dataset_id: datasetId, revision } },
        }),
      ),
  };
}
