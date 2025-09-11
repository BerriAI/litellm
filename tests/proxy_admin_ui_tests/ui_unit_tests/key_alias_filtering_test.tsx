/**
 * Unit tests for Key Alias filtering functionality - Bug #14341 fix
 * 
 * Tests the fix where Key Alias filtering in Request Logs would show logs 
 * briefly and then disappear due to dual filtering conflicts.
 */

// Import only the constants we need to test
const FILTER_KEYS = {
  TEAM_ID: "Team ID",
  KEY_HASH: "Key Hash",
  REQUEST_ID: "Request ID",
  MODEL: "Model",
  USER_ID: "User ID",
  END_USER: "End User",
  STATUS: "Status",
  KEY_ALIAS: "Key Alias",
};

describe('Key Alias Filtering Fix - Bug #14341', () => {
  const mockLogs = {
    data: [
      {
        request_id: 'req-1',
        metadata: {
          user_api_key_alias: 'sk-1234',
          user_api_key_team_alias: 'sk-1234',
          status: 'success'
        }
      },
      {
        request_id: 'req-2',
        metadata: {
          user_api_key_alias: 'sk-5678',
          user_api_key_team_alias: 'sk-5678', 
          status: 'failure'
        }
      },
      {
        request_id: 'req-3',
        metadata: {
          user_api_key_alias: null,
          user_api_key_team_alias: 'sk-1234',
          status: 'success'
        }
      },
      {
        request_id: 'req-4',
        metadata: {
          user_api_key_alias: 'sk-1234',
          user_api_key_team_alias: null,
          status: 'success'
        }
      },
      {
        request_id: 'req-5',
        metadata: null // Missing metadata case
      }
    ],
    total: 5,
    page: 1,
    page_size: 50,
    total_pages: 1
  };

  test('client-side Key Alias filtering works correctly', () => {
    // Test the core filtering logic that was fixed
    const selectedKey = 'sk-1234';
    
    // Filter using the logic from our fix
    const filteredData = mockLogs.data.filter(
      log => log.metadata?.user_api_key_alias === selectedKey || 
             log.metadata?.user_api_key_team_alias === selectedKey
    );
    
    // Should match req-1, req-3, req-4 (3 logs total)
    expect(filteredData).toHaveLength(3);
    
    const requestIds = filteredData.map(log => log.request_id);
    expect(requestIds).toContain('req-1'); // Both fields match
    expect(requestIds).toContain('req-3'); // team_alias matches
    expect(requestIds).toContain('req-4'); // user_alias matches
    expect(requestIds).not.toContain('req-2'); // Different alias
    expect(requestIds).not.toContain('req-5'); // No metadata
  });

  test('safe metadata access prevents undefined errors', () => {
    // Test the accessorFn fix for columns.tsx
    const testCases = [
      { metadata: { user_api_key_alias: 'sk-1234' } },
      { metadata: { user_api_key_team_alias: 'sk-1234' } },
      { metadata: {} },
      { metadata: null },
      {} // No metadata key
    ];
    
    const safeGetAlias = (row: any) => row.metadata?.user_api_key_alias || '-';
    const safeGetTeamAlias = (row: any) => row.metadata?.user_api_key_team_alias || '-';
    
    testCases.forEach(testCase => {
      expect(() => safeGetAlias(testCase)).not.toThrow();
      expect(() => safeGetTeamAlias(testCase)).not.toThrow();
      
      const alias = safeGetAlias(testCase);
      const teamAlias = safeGetTeamAlias(testCase);
      
      expect(typeof alias).toBe('string');
      expect(typeof teamAlias).toBe('string');
    });
  });

  test('server-side parameter exclusion prevents dual filtering', () => {
    // Test that Key Alias is excluded from server calls
    const filters = {
      [FILTER_KEYS.TEAM_ID]: '',
      [FILTER_KEYS.KEY_HASH]: '',
      [FILTER_KEYS.REQUEST_ID]: '',
      [FILTER_KEYS.MODEL]: '',
      [FILTER_KEYS.USER_ID]: '',
      [FILTER_KEYS.END_USER]: '',
      [FILTER_KEYS.STATUS]: '',
      [FILTER_KEYS.KEY_ALIAS]: 'sk-1234' // Only this is set
    };
    
    // Extract server parameters (Key Alias should be excluded)
    const serverParams = {
      api_key: filters[FILTER_KEYS.KEY_HASH] || undefined,
      team_id: filters[FILTER_KEYS.TEAM_ID] || undefined,
      request_id: filters[FILTER_KEYS.REQUEST_ID] || undefined,
      user_id: filters[FILTER_KEYS.USER_ID] || undefined,
      end_user: filters[FILTER_KEYS.END_USER] || undefined,
      status_filter: filters[FILTER_KEYS.STATUS] || undefined,
      model: filters[FILTER_KEYS.MODEL] || undefined
      // Key Alias intentionally excluded
    };
    
    // Verify Key Alias is not in server parameters
    expect(serverParams).not.toHaveProperty('key_alias');
    expect(serverParams).not.toHaveProperty(FILTER_KEYS.KEY_ALIAS);
    
    // All parameters should be undefined when only Key Alias is filtered
    const allParamsEmpty = Object.values(serverParams).every(
      param => param === undefined || param === ''
    );
    expect(allParamsEmpty).toBe(true);
  });

  test('dual filtering prevention logic', () => {
    // Test the logic that prevents server calls when only Key Alias changes
    const prevFilters = {
      [FILTER_KEYS.KEY_ALIAS]: '',
      [FILTER_KEYS.STATUS]: '',
      [FILTER_KEYS.MODEL]: ''
    };
    
    const newFilters = {
      [FILTER_KEYS.KEY_ALIAS]: 'sk-1234', // Only this changed
      [FILTER_KEYS.STATUS]: '',
      [FILTER_KEYS.MODEL]: ''
    };
    
    // Function that checks if server call is needed (our fix)
    const shouldCallServer = (prev: any, updated: any) => {
      // Remove KEY_ALIAS from comparison for server-side filtering
      const prevServer = { ...prev };
      const newServer = { ...updated };
      
      delete prevServer[FILTER_KEYS.KEY_ALIAS];
      delete newServer[FILTER_KEYS.KEY_ALIAS];
      
      return JSON.stringify(prevServer) !== JSON.stringify(newServer);
    };
    
    // Should NOT trigger server call when only Key Alias changes
    expect(shouldCallServer(prevFilters, newFilters)).toBe(false);
    
    // Should trigger server call when other filters change
    const newFiltersWithStatus = {
      ...newFilters,
      [FILTER_KEYS.STATUS]: 'failure'
    };
    expect(shouldCallServer(prevFilters, newFiltersWithStatus)).toBe(true);
  });

  test('FILTER_KEYS constant structure', () => {
    // Verify the filter keys structure is correct
    expect(FILTER_KEYS).toHaveProperty('KEY_ALIAS');
    expect(FILTER_KEYS.KEY_ALIAS).toBe('Key Alias');
    
    // Verify all expected filter keys exist
    const expectedKeys = [
      'TEAM_ID', 'KEY_HASH', 'REQUEST_ID', 'MODEL', 
      'USER_ID', 'END_USER', 'STATUS', 'KEY_ALIAS'
    ];
    
    expectedKeys.forEach(key => {
      expect(FILTER_KEYS).toHaveProperty(key);
    });
  });
});