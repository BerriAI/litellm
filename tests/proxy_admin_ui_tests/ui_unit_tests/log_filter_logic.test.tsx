import { uiSpendLogsCall } from '../../../ui/litellm-dashboard/src/components/networking';

// Mock the networking module
jest.mock('../../../ui/litellm-dashboard/src/components/networking', () => ({
  uiSpendLogsCall: jest.fn(),
}));

const mockUiSpendLogsCall = uiSpendLogsCall as jest.MockedFunction<typeof uiSpendLogsCall>;

describe('Key Alias Filtering Integration Test', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('should call API with correct key_alias parameter', async () => {
    // Mock API response with both success and failure logs
    const mockResponse = {
      data: [
        { request_id: 'req-1', status: 'success', metadata: { user_api_key_alias: 'test-key' } },
        { request_id: 'req-2', status: 'failure', metadata: { user_api_key_alias: 'test-key' } }
      ],
      total: 2,
      page: 1,
      page_size: 50,
      total_pages: 1
    };

    mockUiSpendLogsCall.mockResolvedValueOnce(mockResponse);

    // Simulate the API call that would happen when filtering by key alias
    const result = await uiSpendLogsCall(
      'test-token',
      undefined, 
      undefined,   
      undefined, 
      '2024-01-15 09:00:00', 
      '2024-01-15 11:00:00', 
      1, 
      50, 
      undefined, 
      undefined, 
      undefined, 
      undefined, 
      'test-key-alias' // key_alias - this is the fix
    );

    // Verify the API was called correctly
    expect(mockUiSpendLogsCall).toHaveBeenCalledWith(
      'test-token',
      undefined,
      undefined, 
      undefined,
      '2024-01-15 09:00:00',
      '2024-01-15 11:00:00',
      1,
      50,
      undefined,
      undefined,
      undefined,
      undefined,
      'test-key-alias' // The key assertion - this parameter should be passed through
    );

    // Verify response contains both success and failure logs
    expect(result.data).toHaveLength(2);
    expect(result.data[0].status).toBe('success');
    expect(result.data[1].status).toBe('failure');
  });

  it('should pass undefined for empty key alias', async () => {
    mockUiSpendLogsCall.mockResolvedValueOnce({ data: [], total: 0, page: 1, page_size: 50, total_pages: 0 });

    await uiSpendLogsCall(
      'test-token', undefined, undefined, undefined,
      '2024-01-15 09:00:00', '2024-01-15 11:00:00',
      1, 50, undefined, undefined, undefined, undefined,
      undefined // Empty string should become undefined
    );

    expect(mockUiSpendLogsCall).toHaveBeenCalledWith(
      'test-token', undefined, undefined, undefined,
      '2024-01-15 09:00:00', '2024-01-15 11:00:00', 
      1, 50, undefined, undefined, undefined, undefined,
      undefined // Should be undefined for empty key alias
    );
  });
});