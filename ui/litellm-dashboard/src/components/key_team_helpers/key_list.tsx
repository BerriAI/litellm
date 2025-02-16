import { useState, useEffect } from 'react';
import { keyListCall, Organization } from '../networking';

export interface Team {
    team_id: string;
    team_alias: string;
    models: string[];
    max_budget: number | null;
    budget_duration: string | null;
    tpm_limit: number | null;
    rpm_limit: number | null;
    organization_id: string;
    created_at: string;
}

export interface KeyResponse {
    token: string;
    key_name: string;
    key_alias: string;
    spend: number;
    max_budget: number;
    expires: string;
    models: string[];
    aliases: Record<string, unknown>;
    config: Record<string, unknown>;
    user_id: string;
    team_id: string | null;
    max_parallel_requests: number;
    metadata: Record<string, unknown>;
    tpm_limit: number;
    rpm_limit: number;
    duration: string;
    budget_duration: string;
    budget_reset_at: string;
    allowed_cache_controls: string[];
    permissions: Record<string, unknown>;
    model_spend: Record<string, number>;
    model_max_budget: Record<string, number>;
    soft_budget_cooldown: boolean;
    blocked: boolean;
    litellm_budget_table: Record<string, unknown>;
    organization_id: string | null;
    created_at: string;
    updated_at: string;
    team_spend: number;
    team_alias: string;
    team_tpm_limit: number;
    team_rpm_limit: number;
    team_max_budget: number;
    team_models: string[];
    team_blocked: boolean;
    soft_budget: number;
    team_model_aliases: Record<string, string>;
    team_member_spend: number;
    team_member?: {
        user_id: string;
        user_email: string;
        role: 'admin' | 'user';
    };
    team_metadata: Record<string, unknown>;
    end_user_id: string;
    end_user_tpm_limit: number;
    end_user_rpm_limit: number;
    end_user_max_budget: number;
    last_refreshed_at: number;
    api_key: string;
    user_role: 'proxy_admin' | 'user';
    allowed_model_region?: 'eu' | 'us' | string;
    parent_otel_span?: string;
    rpm_limit_per_model: Record<string, number>;
    tpm_limit_per_model: Record<string, number>;
    user_tpm_limit: number;
    user_rpm_limit: number;
    user_email: string;
}

interface KeyListResponse {
keys: KeyResponse[];
total_count: number;
current_page: number;
total_pages: number;
}

interface UseKeyListProps {
selectedTeam?: Team;
currentOrg: Organization | null;
accessToken: string;
currentPage?: number;
}

interface PaginationData {
currentPage: number;
totalPages: number;
totalCount: number;
}

interface UseKeyListReturn {
keys: KeyResponse[];
isLoading: boolean;
error: Error | null;
pagination: PaginationData;
refresh: (params?: Record<string, unknown>) => Promise<void>;
}

const useKeyList = ({
    selectedTeam,
    currentOrg,
    accessToken,
    currentPage = 1,

}: UseKeyListProps): UseKeyListReturn => {
    const [keyData, setKeyData] = useState<KeyListResponse>({ 
        keys: [], 
        total_count: 0, 
        current_page: 1, 
        total_pages: 0 
    });
    const [isLoading, setIsLoading] = useState<boolean>(true);
    const [error, setError] = useState<Error | null>(null);

    const fetchKeys = async (params: Record<string, unknown> = {}): Promise<void> => {
        try {
            console.log("calling fetchKeys");
            if (!accessToken) {
                console.log("accessToken", accessToken);
                return;
            }
            setIsLoading(true);

            const data = await keyListCall(
                accessToken,
                currentOrg?.organization_id || null,
                selectedTeam?.team_id || "",
                params.page as number || 1,
                50
            );
            console.log("data", data);
            setKeyData(data);
            setError(null);
        } catch (err) {
            setError(err instanceof Error ? err : new Error("An error occurred"));
        } finally {
            setIsLoading(false);
        }
    };

    useEffect(() => {
        fetchKeys();
        console.log("selectedTeam", selectedTeam, "currentOrg", currentOrg, "accessToken", accessToken);
    }, [selectedTeam, currentOrg, accessToken]);

    return {
        keys: keyData.keys,
        isLoading,
        error,
        pagination: {
        currentPage: keyData.current_page,
        totalPages: keyData.total_pages,
        totalCount: keyData.total_count
        },
        refresh: fetchKeys
    };
};

export default useKeyList;