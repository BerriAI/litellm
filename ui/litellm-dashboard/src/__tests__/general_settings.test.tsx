import { act } from '@testing-library/react';
import { setCallbacksCall } from '../components/networking';

// Mock the networking module
jest.mock('../components/networking', () => ({
  setCallbacksCall: jest.fn(),
}));

// Mock antd message
jest.mock('antd', () => ({
  message: {
    success: jest.fn(),
    error: jest.fn(),
  },
}));

describe('deleteFallbacks', () => {
  let mockSetRouterSettings: jest.Mock;
  let deleteFallbacks: (key: string) => Promise<void>;

  beforeEach(() => {
    mockSetRouterSettings = jest.fn();
    
    // Reset mocks
    (setCallbacksCall as jest.Mock).mockReset();
    
    // Initialize the deleteFallbacks function with required context
    deleteFallbacks = async (key: string) => {
      const routerSettings = {
        fallbacks: [
          { "gpt-4": ["gpt-3.5-turbo", "claude-2"] },
          { "gpt-3.5-turbo": ["claude-2"] }
        ],
        // other settings...
      };

      const updatedFallbacks = routerSettings.fallbacks
        .map((dict: { [key: string]: any }) => {
          if (key in dict) {
            delete dict[key];
          }
          return dict;
        })
        .filter((dict: { [key: string]: any }) => Object.keys(dict).length > 0);

      const updatedSettings = {
        ...routerSettings,
        fallbacks: updatedFallbacks
      };

      try {
        await setCallbacksCall("mock-token", { router_settings: updatedSettings });
        mockSetRouterSettings(updatedSettings);
      } catch (error) {
        throw error;
      }
    };
  });

  it('should remove the fallback and filter out empty dictionaries', async () => {
    // Act
    await act(async () => {
      await deleteFallbacks('gpt-4');
    });

    // Assert
    expect(mockSetRouterSettings).toHaveBeenCalledWith(
      expect.objectContaining({
        fallbacks: [
          { "gpt-3.5-turbo": ["claude-2"] } // Only non-empty dictionary remains
        ]
      })
    );
  });

  it('should handle deleting non-existent key', async () => {
    // Act
    await act(async () => {
      await deleteFallbacks('non-existent-model');
    });

    // Assert
    expect(mockSetRouterSettings).toHaveBeenCalledWith(
      expect.objectContaining({
        fallbacks: [
          { "gpt-4": ["gpt-3.5-turbo", "claude-2"] },
          { "gpt-3.5-turbo": ["claude-2"] }
        ]
      })
    );
  });

  it('should handle network errors', async () => {
    // Setup
    (setCallbacksCall as jest.Mock).mockRejectedValue(new Error('Network error'));

    // Act & Assert
    await expect(deleteFallbacks('gpt-4')).rejects.toThrow('Network error');
  });
}); 