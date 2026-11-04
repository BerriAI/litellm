import {
  dailyActivityAggregatedCall,
  dailyActivityExportCall,
  dailyActivityKeyPageCall,
  dailyActivityKeySearchCall,
  dailyActivityModelTopKeysCall,
  dailyActivityTagPageCall,
} from "@/components/networking";
import type { EntityType } from "@/components/EntityUsageExport/types";
import type {
  DailyActivityAggregatedResponse,
  DailyActivityEntity,
  DailyActivityKeySearchResponse,
  DailyActivityKeyPageResponse,
  DailyActivityRequest,
  DailyActivityTagPageRequest,
  DailyActivityTagPageResponse,
  ExportFormat,
  ExportType,
  ModelTopKeysResponse,
} from "@/components/UsagePage/dailyActivityApi";

export interface EntityApi {
  aggregated(req: DailyActivityRequest): Promise<DailyActivityAggregatedResponse>;
  keyPage(req: DailyActivityRequest, offset: number, limit: number): Promise<DailyActivityKeyPageResponse>;
  searchKeys(req: DailyActivityRequest, search: string, limit?: number): Promise<DailyActivityKeySearchResponse>;
  modelTopKeys(
    req: DailyActivityRequest,
    model: string,
    byModelGroup: boolean,
    limit?: number,
  ): Promise<ModelTopKeysResponse>;
  exportRows(req: DailyActivityRequest, exportType: ExportType, format: ExportFormat): Promise<Blob>;
}

export interface TeamEntityApi extends EntityApi {
  tagPage(req: DailyActivityTagPageRequest): Promise<DailyActivityTagPageResponse>;
}

const entityApi = (
  entity: DailyActivityEntity,
  defaults?: Pick<DailyActivityRequest, "excludeEntityIds">,
): EntityApi => ({
  aggregated: (req) => dailyActivityAggregatedCall(entity, { ...defaults, ...req }),
  keyPage: (req, offset, limit) => dailyActivityKeyPageCall(entity, { ...defaults, ...req }, offset, limit),
  searchKeys: (req, search, limit) => dailyActivityKeySearchCall(entity, { ...defaults, ...req }, search, limit),
  modelTopKeys: (req, model, byModelGroup, limit) =>
    dailyActivityModelTopKeysCall(entity, { ...defaults, ...req }, model, byModelGroup, limit),
  exportRows: (req, exportType, format) => dailyActivityExportCall(entity, { ...defaults, ...req }, exportType, format),
});

export const ENTITY_API: Record<EntityType, EntityApi> & { team: TeamEntityApi } = {
  tag: entityApi("tag"),
  team: {
    ...entityApi("team", { excludeEntityIds: ["litellm-dashboard"] }),
    tagPage: dailyActivityTagPageCall,
  },
  organization: entityApi("organization"),
  customer: entityApi("customer"),
  agent: entityApi("agent"),
  user: entityApi("user"),
};
