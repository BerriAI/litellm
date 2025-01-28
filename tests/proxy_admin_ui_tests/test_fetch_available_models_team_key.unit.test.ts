/**
 * Unit Tests for fetch_available_models_team_key.tsx
 *
 * How to run:
 * 1. Install Jest (and ts-jest) if not already installed:
 *    npm install --save-dev jest ts-jest @types/jest
 * 2. Ensure you have a jest config. For a quick setup, you can create "jest.config.js" or "jest.config.ts".
 * 3. Run the tests:
 *    npx jest tests/proxy_admin_ui_tests/test_fetch_available_models_team_key.unit.test.ts
 *
 * or add a script to your package.json:
 *   "scripts": {
 *     "test:unit": "jest"
 *   }
 * and then run:
 *   npm run test:unit
 */

import { describe, it, expect, jest } from '@jest/globals';

// If "modelAvailableCall" comes from ../networking, mock it:
jest.mock('../../ui/litellm-dashboard/src/components/networking', () => {
  return {
    modelAvailableCall: jest.fn(),
  };
});

import {
    fetchAvailableModelsForTeamOrKey,
    getModelDisplayName,
    unfurlWildcardModelsInList,
  } from '../../ui/litellm-dashboard/src/components/key_team_helpers/fetch_available_models_team_key';
  import { modelAvailableCall } from '../../ui/litellm-dashboard/src/components/networking';
  
// Define the type for the modelAvailableCall response
type ModelAvailableResponse = {
  data: Array<{ id: string }>;
};

// Update the mock typing
const mockModelAvailableCall = modelAvailableCall as unknown as jest.MockedFunction<
  (accessToken: String, userID: String, userRole: String, return_wildcard_routes?: boolean) => Promise<ModelAvailableResponse>
>;

describe('fetchAvailableModelsForTeamOrKey', () => {
  it('returns undefined if userID or userRole is null', async () => {
    const result1 = await fetchAvailableModelsForTeamOrKey(null as any, 'someRole', 'accessToken');
    const result2 = await fetchAvailableModelsForTeamOrKey('someUser', null as any, 'accessToken');
    expect(result1).toBeUndefined();
    expect(result2).toBeUndefined();
  });

  it('returns sorted providerModels followed by specificModels', async () => {
    mockModelAvailableCall.mockResolvedValue({
      data: [{ id: 'openai/*' }, { id: 'openai/gpt-3.5' }, { id: 'anthropic/*' }, { id: 'openai/gpt-4' }],
    });

    const result = await fetchAvailableModelsForTeamOrKey('someUser', 'someRole', 'accessToken');
    // We expect all provider wildcards sorted first, then specific
    // The function doesn't re-sort among wildcards but does place them before specifics
    // So the result should have "openai/*" and "anthropic/*" first, then "openai/gpt-3.5" and "openai/gpt-4"
    expect(result).toEqual(['openai/*', 'anthropic/*', 'openai/gpt-3.5', 'openai/gpt-4']);
  });

  it('handles errors gracefully and logs to console', async () => {
    const consoleSpy = jest.spyOn(console, 'error').mockImplementation(() => {});
    mockModelAvailableCall.mockRejectedValue(new Error('Test Error'));

    const result = await fetchAvailableModelsForTeamOrKey('someUser', 'someRole', 'accessToken');
    expect(result).toBeUndefined();
    expect(consoleSpy).toHaveBeenCalledWith('Error fetching user models:', expect.any(Error));

    consoleSpy.mockRestore();
  });
});

describe('getModelDisplayName', () => {
  it('returns "All <provider> models" for wildcard patterns', () => {
    const wildcardName = getModelDisplayName('openai/*');
    expect(wildcardName).toBe('All openai models');
  });

  it('returns the original model string if not a wildcard', () => {
    const normalName = getModelDisplayName('openai/gpt-4');
    expect(normalName).toBe('openai/gpt-4');
  });
});

describe('unfurlWildcardModelsInList', () => {
  it('expands wildcard models correctly', () => {
    const teamModels = ['openai/*', 'anthropic/claude-2'];
    const allModels = ['openai/gpt-3.5', 'openai/gpt-4', 'anthropic/claude-2', 'anthropic/*'];

    const result = unfurlWildcardModelsInList(teamModels, allModels);
    // The openai/* should expand to openai/gpt-3.5 & openai/gpt-4,
    // that wildcard is pushed to top, then specific models, in the order described by the function
    // So the returned array should have wildcard ("openai/*"), then expansions, then any direct listing
    expect(result).toEqual([
      'openai/*',
      'openai/gpt-3.5',
      'openai/gpt-4',
      'anthropic/claude-2', // direct mention
    ]);
  });

  it('returns teamModels unchanged if no wildcard is present', () => {
    const teamModels = ['openai/gpt-3.5', 'anthropic/claude-2'];
    const allModels = ['openai/gpt-3.5', 'openai/gpt-4', 'anthropic/claude-2'];

    const result = unfurlWildcardModelsInList(teamModels, allModels);
    // No wildcard to expand, so we just return the same array
    expect(result).toEqual(['openai/gpt-3.5', 'anthropic/claude-2']);
  });
}); 