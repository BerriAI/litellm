import { useState, useEffect } from 'react';
import { keyListCall, Organization } from '../networking';
interface Team {
team_id: string;
team_alias: string;
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
budget_duration: string;
budget_reset_at: string;
allowed_cache_controls: string[];
permissions: Record<string, unknown>;
model_spend: Record<string, number>;
model_max_budget: Record<string, number>;
soft_budget_cooldown: boolean;
blocked: boolean;
litellm_budget_table: Record<string, unknown>;
org_id: string | null;
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

const useKeyList = ({ selectedTeam, currentOrg, accessToken }: UseKeyListProps): UseKeyListReturn => {
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
            if (!currentOrg || !selectedTeam || !accessToken) {
                console.log("currentOrg", currentOrg);
                console.log("selectedTeam", selectedTeam);
                console.log("accessToken", accessToken);
                return
            }
            setIsLoading(true);

            const data = await keyListCall(accessToken, currentOrg.organization_id, selectedTeam.team_id);
            console.log("data", data);
            setKeyData(data);
            setError(null);
        } catch (err) {
            setError(err instanceof Error ? err : new Error('An error occurred'));
        } finally {
            setIsLoading(false);
        }
    };

    useEffect(() => {
        fetchKeys();
    }, [selectedTeam, currentOrg]);

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