// Test for the core logic of team creation model preservation
// This test focuses on the filtering logic without rendering the full component

// Helper function to get organization models (copied from the component)
const getOrganizationModels = (organization: any, userModels: string[]) => {
  let tempModelsToPick = [];
  if (organization) {
    if (organization.models.length > 0) {
      tempModelsToPick = organization.models;
    } else {
      tempModelsToPick = userModels;
    }
  } else {
    tempModelsToPick = userModels;
  }
  return tempModelsToPick;
};

describe('Team Creation Modal Model Preservation', () => {
  const defaultProps = {
    teams: [],
    searchParams: {},
    accessToken: 'test-token',
    setTeams: jest.fn(),
    userID: 'test-user',
    userRole: 'Admin',
    organizations: [
      {
        organization_id: 'org1',
        organization_alias: 'Organization 1',
        models: ['model1', 'model2']
      },
      {
        organization_id: 'org2', 
        organization_alias: 'Organization 2',
        models: ['model2', 'model3', 'model4']
      }
    ],
    premiumUser: false,
  };

  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('should preserve models when switching to organization that includes them', () => {
    const userModels = ['model1', 'model2', 'model3', 'model4'];
    const selectedModels = ['model1', 'model2'];
    
    // Organization 1 has models: ['model1', 'model2']
    const org1 = defaultProps.organizations![0];
    const availableModels1 = getOrganizationModels(org1, userModels);
    
    // Simulate current selected models
    const stillAvailableModels = selectedModels.filter((selectedModel: string) => 
      selectedModel === "all-proxy-models" || availableModels1.includes(selectedModel)
    );
    
    // Both selected models should still be available
    expect(stillAvailableModels).toEqual(['model1', 'model2']);
    expect(stillAvailableModels.length).toBe(selectedModels.length);
  });

  it('should remove only unavailable models when switching organization', () => {
    const userModels = ['model1', 'model2', 'model3', 'model4'];
    const selectedModels = ['model1', 'model2', 'model3'];
    
    // Organization 1 has models: ['model1', 'model2'] 
    const org1 = defaultProps.organizations![0];
    const availableModels1 = getOrganizationModels(org1, userModels);
    
    const stillAvailableModels = selectedModels.filter((selectedModel: string) => 
      selectedModel === "all-proxy-models" || availableModels1.includes(selectedModel)
    );
    
    // Only model1 and model2 should remain, model3 should be filtered out
    expect(stillAvailableModels).toEqual(['model1', 'model2']);
    expect(stillAvailableModels.length).toBe(2);
    expect(stillAvailableModels.length).toBeLessThan(selectedModels.length);
  });

  it('should preserve all-proxy-models selection regardless of organization', () => {
    const userModels = ['model1', 'model2', 'model3', 'model4'];
    const selectedModels = ['all-proxy-models', 'model1'];
    
    // Organization 1 has models: ['model1', 'model2']
    const org1 = defaultProps.organizations![0];
    const availableModels1 = getOrganizationModels(org1, userModels);
    
    const stillAvailableModels = selectedModels.filter((selectedModel: string) => 
      selectedModel === "all-proxy-models" || availableModels1.includes(selectedModel)
    );
    
    // Both all-proxy-models and model1 should be preserved
    expect(stillAvailableModels).toEqual(['all-proxy-models', 'model1']);
    expect(stillAvailableModels.length).toBe(selectedModels.length);
  });

  it('should handle empty model selection without errors', () => {
    const userModels = ['model1', 'model2', 'model3', 'model4'];
    const selectedModels: string[] = [];
    
    const org1 = defaultProps.organizations![0];
    const availableModels1 = getOrganizationModels(org1, userModels);
    
    const stillAvailableModels = selectedModels.filter((selectedModel: string) => 
      selectedModel === "all-proxy-models" || availableModels1.includes(selectedModel)
    );
    
    expect(stillAvailableModels).toEqual([]);
    expect(stillAvailableModels.length).toBe(0);
    expect(stillAvailableModels.length).toBe(selectedModels.length);
  });

  it('should handle organization with no models specified (should use all user models)', () => {
    const userModels = ['model1', 'model2', 'model3', 'model4'];
    const selectedModels = ['model1', 'model3'];
    
    // Organization with empty models array should use all user models
    const orgWithNoModels = {
      organization_id: 'org3',
      organization_alias: 'Organization 3',
      models: []
    };
    
    const availableModels = getOrganizationModels(orgWithNoModels, userModels);
    expect(availableModels).toEqual(userModels);
    
    const stillAvailableModels = selectedModels.filter((selectedModel: string) => 
      selectedModel === "all-proxy-models" || availableModels.includes(selectedModel)
    );
    
    // All selected models should be preserved since org uses all user models
    expect(stillAvailableModels).toEqual(['model1', 'model3']);
    expect(stillAvailableModels.length).toBe(selectedModels.length);
  });
});