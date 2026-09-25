import type { EntityType } from "@/components/EntityUsageExport/types";
import {
  agentDailyActivityAggregatedCall,
  agentDailyActivityCall,
  agentDailyActivityExportCall,
  agentDailyActivityKeySearchCall,
  agentDailyActivityModelTopKeysCall,
  customerDailyActivityAggregatedCall,
  customerDailyActivityCall,
  customerDailyActivityExportCall,
  customerDailyActivityKeySearchCall,
  customerDailyActivityModelTopKeysCall,
  organizationDailyActivityAggregatedCall,
  organizationDailyActivityCall,
  organizationDailyActivityExportCall,
  organizationDailyActivityKeySearchCall,
  organizationDailyActivityModelTopKeysCall,
  tagDailyActivityAggregatedCall,
  tagDailyActivityCall,
  tagDailyActivityExportCall,
  tagDailyActivityKeySearchCall,
  tagDailyActivityModelTopKeysCall,
  teamDailyActivityAggregatedCall,
  teamDailyActivityCall,
  teamDailyActivityExportCall,
  teamDailyActivityKeySearchCall,
  teamDailyActivityModelTopKeysCall,
  userDailyActivityAggregatedCall,
  userDailyActivityCall,
  userDailyActivityExportCall,
  userDailyActivityKeySearchCall,
  userDailyActivityModelTopKeysCall,
  type ModelTopApiKeysGroupBy,
  type ModelTopApiKeysResponse,
} from "@/components/networking";

export const ENTITY_FETCH_FNS: Record<EntityType, (...args: any[]) => Promise<any>> = {
  tag: tagDailyActivityCall,
  team: teamDailyActivityCall,
  organization: organizationDailyActivityCall,
  customer: customerDailyActivityCall,
  agent: agentDailyActivityCall,
  user: userDailyActivityCall,
};

export const ENTITY_AGGREGATED_FETCH_FNS: Record<EntityType, (...args: any[]) => Promise<any>> = {
  tag: tagDailyActivityAggregatedCall,
  team: teamDailyActivityAggregatedCall,
  organization: organizationDailyActivityAggregatedCall,
  customer: customerDailyActivityAggregatedCall,
  agent: agentDailyActivityAggregatedCall,
  user: userDailyActivityAggregatedCall,
};

type EntityKeySearchCall = (
  accessToken: string,
  startTime: Date,
  endTime: Date,
  ...options: [search: string, entityIds?: string[] | null]
) => Promise<any>;

export const ENTITY_KEY_SEARCH_FNS: Partial<Record<EntityType, EntityKeySearchCall>> = {
  tag: tagDailyActivityKeySearchCall,
  team: teamDailyActivityKeySearchCall,
  organization: organizationDailyActivityKeySearchCall,
  customer: customerDailyActivityKeySearchCall,
  agent: agentDailyActivityKeySearchCall,
  user: (accessToken, startTime, endTime, ...options) =>
    userDailyActivityKeySearchCall(accessToken, startTime, endTime, options[0], options[1]?.[0] ?? null),
};

export type EntityExportCall = (options: {
  accessToken: string;
  startTime: Date;
  endTime: Date;
  entityIds: string[] | null;
  exportType: Parameters<typeof teamDailyActivityExportCall>[0]["exportType"];
  format: Parameters<typeof teamDailyActivityExportCall>[0]["format"];
}) => Promise<Blob>;

const teamExportCall: EntityExportCall = ({ entityIds, ...rest }) =>
  teamDailyActivityExportCall({ ...rest, teamIds: entityIds });

export const ENTITY_EXPORT_FNS: Partial<Record<EntityType, EntityExportCall>> = {
  tag: tagDailyActivityExportCall,
  team: teamExportCall,
  organization: organizationDailyActivityExportCall,
  customer: customerDailyActivityExportCall,
  agent: agentDailyActivityExportCall,
  user: ({ entityIds, ...rest }) => userDailyActivityExportCall({ ...rest, userId: entityIds?.[0] ?? null }),
};

export type EntityModelTopKeysCall = (
  accessToken: string,
  startTime: Date,
  endTime: Date,
  ...options: [model: string, groupBy: ModelTopApiKeysGroupBy, entityIds?: string[] | null]
) => Promise<ModelTopApiKeysResponse>;

export const ENTITY_MODEL_TOP_KEYS_FNS: Record<EntityType, EntityModelTopKeysCall> = {
  tag: tagDailyActivityModelTopKeysCall,
  team: teamDailyActivityModelTopKeysCall,
  organization: organizationDailyActivityModelTopKeysCall,
  customer: customerDailyActivityModelTopKeysCall,
  agent: agentDailyActivityModelTopKeysCall,
  user: (accessToken, startTime, endTime, ...options) =>
    userDailyActivityModelTopKeysCall(accessToken, startTime, endTime, options[0], options[1], options[2]?.[0] ?? null),
};
