import React from 'react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import userEvent from '@testing-library/user-event';
import { renderWithProviders, screen, waitFor } from "../../tests/test-utils";
import UIAccessControlForm from './UIAccessControlForm';
import * as networking from './networking';

// Mock the networking module
vi.mock('./networking', () => ({
  getSSOSettings: vi.fn(),
  updateSSOSettings: vi.fn(),
}));

// Mock NotificationManager
vi.mock('./molecules/notifications_manager', () => ({
  default: {
    fromBackend: vi.fn(),
  },
}));

describe('UIAccessControlForm', () => {
  const mockAccessToken = 'test-access-token';
  const mockOnSuccess = vi.fn();

  beforeEach(() => {
    vi.resetAllMocks();
    
    // Default mock for getSSOSettings
    vi.mocked(networking.getSSOSettings).mockResolvedValue({
      values: {
        ui_access_mode: {
          type: 'restricted_sso_group',
          restricted_sso_group: 'admin-group',
          sso_group_jwt_field: 'groups',
        }
      }
    });

    // Default mock for updateSSOSettings
    vi.mocked(networking.updateSSOSettings).mockResolvedValue({});
  });

  it('renders the form with all expected fields', async () => {
    renderWithProviders(
      <UIAccessControlForm 
        accessToken={mockAccessToken} 
        onSuccess={mockOnSuccess} 
      />
    );

    // Wait for the form to load
    await waitFor(() => {
      expect(screen.getByText('UI Access Mode')).toBeInTheDocument();
    });

    expect(screen.getByText('Configure who can access the UI interface and how group information is extracted from JWT tokens.')).toBeInTheDocument();
    expect(screen.getByText('SSO Group JWT Field')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Update UI Access Control' })).toBeInTheDocument();
  });

  it('sends ui_access_mode="none" when ui_access_mode_type is "all_authenticated_users"', async () => {
    const user = userEvent.setup();
    
    renderWithProviders(
      <UIAccessControlForm 
        accessToken={mockAccessToken} 
        onSuccess={mockOnSuccess} 
      />
    );

    // Wait for form to load
    await waitFor(() => {
      expect(screen.getByText('UI Access Mode')).toBeInTheDocument();
    });

    // Select "All Authenticated Users" from the dropdown
    const accessModeSelect = screen.getByRole('combobox');
    await user.click(accessModeSelect);
    
    const allUsersOption = screen.getByText('All Authenticated Users');
    await user.click(allUsersOption);

    // Fill in the JWT field (optional but good for completeness)
    const jwtField = screen.getByPlaceholderText('groups');
    await user.clear(jwtField);
    await user.type(jwtField, 'user_groups');

    // Submit the form
    const submitButton = screen.getByRole('button', { name: 'Update UI Access Control' });
    await user.click(submitButton);

    // Verify that updateSSOSettings was called with the correct payload
    await waitFor(() => {
      expect(networking.updateSSOSettings).toHaveBeenCalledWith(
        mockAccessToken,
        {
          ui_access_mode: "none"
        }
      );
    });

    expect(mockOnSuccess).toHaveBeenCalled();
  });

  it('sends object structure when ui_access_mode_type is "restricted_sso_group"', async () => {
    const user = userEvent.setup();
    
    renderWithProviders(
      <UIAccessControlForm 
        accessToken={mockAccessToken} 
        onSuccess={mockOnSuccess} 
      />
    );

    // Wait for form to load
    await waitFor(() => {
      expect(screen.getByText('UI Access Mode')).toBeInTheDocument();
    });

    // Select "Restricted SSO Group" from the dropdown
    const accessModeSelect = screen.getByRole('combobox');
    await user.click(accessModeSelect);
    
    const restrictedOption = screen.getByText('Restricted SSO Group');
    await user.click(restrictedOption);

    // Fill in the restricted SSO group field (should appear after selection)
    await waitFor(() => {
      expect(screen.getByText('Restricted SSO Group')).toBeInTheDocument();
    });
    
    const restrictedGroupInput = screen.getByPlaceholderText('ui-access-group');
    await user.clear(restrictedGroupInput);
    await user.type(restrictedGroupInput, 'admin-users');

    // Fill in the JWT field
    const jwtField = screen.getByPlaceholderText('groups');
    await user.clear(jwtField);
    await user.type(jwtField, 'team_groups');

    // Submit the form
    const submitButton = screen.getByRole('button', { name: 'Update UI Access Control' });
    await user.click(submitButton);

    // Verify that updateSSOSettings was called with the correct object structure
    await waitFor(() => {
      expect(networking.updateSSOSettings).toHaveBeenCalledWith(
        mockAccessToken,
        {
          ui_access_mode: {
            type: 'restricted_sso_group',
            restricted_sso_group: 'admin-users',
            sso_group_jwt_field: 'team_groups',
          }
        }
      );
    });

    expect(mockOnSuccess).toHaveBeenCalled();
  });

  it('shows restricted SSO group field only when restricted_sso_group is selected', async () => {
    const user = userEvent.setup();
    
    renderWithProviders(
      <UIAccessControlForm 
        accessToken={mockAccessToken} 
        onSuccess={mockOnSuccess} 
      />
    );

    // Wait for form to load
    await waitFor(() => {
      expect(screen.getByText('UI Access Mode')).toBeInTheDocument();
    });

    // Initially, with "restricted_sso_group" selected, the field should be visible
    expect(screen.getByText('Restricted SSO Group')).toBeInTheDocument();

    // Switch to "All Authenticated Users"
    const accessModeSelect = screen.getByRole('combobox');
    await user.click(accessModeSelect);
    
    const allUsersOption = screen.getByText('All Authenticated Users');
    await user.click(allUsersOption);

    // The restricted SSO group field should disappear
    await waitFor(() => {
      expect(screen.queryByText('Restricted SSO Group')).not.toBeInTheDocument();
    });
  });

  it('handles API errors gracefully', async () => {
    const user = userEvent.setup();
    const consoleSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
    
    // Mock updateSSOSettings to reject
    vi.mocked(networking.updateSSOSettings).mockRejectedValue(new Error('API Error'));
    
    renderWithProviders(
      <UIAccessControlForm 
        accessToken={mockAccessToken} 
        onSuccess={mockOnSuccess} 
      />
    );

    // Wait for form to load
    await waitFor(() => {
      expect(screen.getByText('UI Access Mode')).toBeInTheDocument();
    });

    // Submit the form
    const submitButton = screen.getByRole('button', { name: 'Update UI Access Control' });
    await user.click(submitButton);

    // Verify error handling
    await waitFor(() => {
      expect(consoleSpy).toHaveBeenCalledWith('Failed to save UI access settings:', expect.any(Error));
    });

    // onSuccess should not be called on error
    expect(mockOnSuccess).not.toHaveBeenCalled();

    consoleSpy.mockRestore();
  });
});
