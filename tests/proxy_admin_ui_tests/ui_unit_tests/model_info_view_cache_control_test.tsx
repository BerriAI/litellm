import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import ModelInfoView from '../../../ui/litellm-dashboard/src/components/model_info_view';

// Mock all the dependencies that ModelInfoView uses
const mockModelPatchUpdateCall = jest.fn().mockResolvedValue({ data: 'success' });
const mockCredentialGetCall = jest.fn().mockResolvedValue({
  credential_name: 'test-credential',
  credential_values: {},
  credential_info: {}
});
const mockModelInfoV1Call = jest.fn().mockResolvedValue({
  data: [
    {
      model_name: 'test-model',
      litellm_model_name: 'gpt-4',
      litellm_params: {
        cache_control_injection_points: []
      },
      model_info: {
        id: 'test-model-id'
      }
    }
  ]
});
const mockGetGuardrailsList = jest.fn().mockResolvedValue({
  guardrails: []
});

jest.mock('../../../ui/litellm-dashboard/src/components/networking', () => ({
  modelPatchUpdateCall: mockModelPatchUpdateCall,
  credentialGetCall: mockCredentialGetCall,
  modelInfoV1Call: mockModelInfoV1Call,
  getGuardrailsList: mockGetGuardrailsList
}));

// Mock Tremor components
jest.mock('@tremor/react', () => ({
  Card: ({ children }: { children: React.ReactNode }) => <div data-testid="card">{children}</div>,
  Title: ({ children }: { children: React.ReactNode }) => <h1>{children}</h1>,
  Text: ({ children }: { children: React.ReactNode }) => <span>{children}</span>,
  Tab: ({ children }: { children: React.ReactNode }) => <div role="tab">{children}</div>,
  TabList: ({ children }: { children: React.ReactNode }) => <div role="tablist">{children}</div>,
  TabGroup: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  TabPanel: ({ children }: { children: React.ReactNode }) => <div role="tabpanel">{children}</div>,
  TabPanels: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  Grid: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  Badge: ({ children }: { children: React.ReactNode }) => <span>{children}</span>,
  Button: ({ children, onClick, icon: Icon }: { children: React.ReactNode, onClick?: () => void, icon?: any }) => 
    <button onClick={onClick}>{Icon && <Icon />}{children}</button>,
  TextInput: ({ placeholder, value, onChange }: { placeholder?: string, value?: string, onChange?: (e: any) => void }) => 
    <input placeholder={placeholder} value={value} onChange={onChange} />
}));

// Mock Ant Design components
jest.mock('antd', () => ({
  Form: {
    useForm: () => [{ 
      resetFields: jest.fn(), 
      submit: jest.fn(),
      getFieldsValue: () => ({
        cache_control: false,
        cache_control_injection_points: []
      })
    }],
    Item: ({ children, name }: { children: React.ReactNode, name?: string }) => 
      <div data-testid={`form-item-${name}`}>{children}</div>
  },
  Button: ({ children, onClick, type, danger }: { 
    children: React.ReactNode, 
    onClick?: () => void, 
    type?: string,
    danger?: boolean 
  }) => (
    <button 
      onClick={onClick} 
      className={danger ? 'danger' : type || ''}
      role="button"
    >
      {children}
    </button>
  ),
  Input: {
    TextArea: ({ value, onChange }: { value?: string, onChange?: (e: any) => void }) => 
      <textarea value={value} onChange={onChange} />
  },
  InputNumber: ({ value, onChange }: { value?: number, onChange?: (value: number) => void }) => 
    <input type="number" value={value} onChange={(e) => onChange?.(parseInt(e.target.value))} />,
  message: {
    error: jest.fn(),
    success: jest.fn()
  },
  Select: ({ children, mode, value, onChange }: { 
    children?: React.ReactNode, 
    mode?: string, 
    value?: any, 
    onChange?: (value: any) => void 
  }) => (
    <select multiple={mode === 'tags'} value={value} onChange={(e) => onChange?.(e.target.value)}>
      {children}
    </select>
  ),
  Modal: ({ children, open, onCancel }: { 
    children: React.ReactNode, 
    open?: boolean, 
    onCancel?: () => void 
  }) => 
    open ? <div onClick={onCancel}>{children}</div> : null,
  Tooltip: ({ children }: { children: React.ReactNode }) => <div>{children}</div>
}));

// Mock other dependencies
jest.mock('../../../ui/litellm-dashboard/src/components/shared/numerical_input', () => 
  ({ value, onChange }: { value?: number, onChange?: (e: any) => void }) => 
    <input type="number" value={value} onChange={onChange} />
);

jest.mock('../../../ui/litellm-dashboard/src/components/add_model/cache_control_settings', () => 
  ({ form, showCacheControl, onCacheControlChange }: { 
    form: any, 
    showCacheControl: boolean, 
    onCacheControlChange: (checked: boolean) => void 
  }) => (
    <div data-testid="cache-control-settings">
      <input 
        type="checkbox" 
        checked={showCacheControl} 
        onChange={(e) => onCacheControlChange(e.target.checked)}
        data-testid="cache-control-checkbox"
      />
      Cache Control Settings
    </div>
  )
);

// Mock other component dependencies
jest.mock('../../../ui/litellm-dashboard/src/components/provider_info_helpers', () => ({
  getProviderLogoAndName: () => ({ logo: 'test-logo.png' })
}));

jest.mock('../../../ui/litellm-dashboard/src/components/view_model/model_name_display', () => ({
  getDisplayModelName: () => 'Test Model'
}));

jest.mock('../../../ui/litellm-dashboard/src/utils/dataUtils', () => ({
  copyToClipboard: jest.fn().mockResolvedValue(true)
}));

jest.mock('../../../ui/litellm-dashboard/src/components/molecules/notifications_manager', () => ({
  success: jest.fn(),
  error: jest.fn(),
  info: jest.fn(),
  fromBackend: jest.fn()
}));

// Mock heroicons
jest.mock('@heroicons/react/outline', () => ({
  ArrowLeftIcon: () => <div>ArrowLeft</div>,
  TrashIcon: () => <div>Trash</div>,
  KeyIcon: () => <div>Key</div>
}));

jest.mock('lucide-react', () => ({
  CheckIcon: () => <div>Check</div>,
  CopyIcon: () => <div>Copy</div>
}));

describe('ModelInfoView Cache Control', () => {
  const mockOnModelUpdate = jest.fn();
  const mockOnClose = jest.fn();

  const defaultProps = {
    modelId: 'test-model-id',
    onClose: mockOnClose,
    modelData: {
      model_name: 'test-model',
      litellm_model_name: 'gpt-4',
      provider: 'openai',
      input_cost: 0.001,
      output_cost: 0.002,
      litellm_params: {
        cache_control_injection_points: [],
        api_base: 'https://api.openai.com/v1',
        custom_llm_provider: 'openai'
      },
      model_info: {
        id: 'test-model-id',
        created_at: '2024-01-01',
        created_by: 'test-user'
      }
    },
    accessToken: 'test-access-token',
    userID: 'test-user-id',
    userRole: 'Admin',
    editModel: false,
    setEditModalVisible: jest.fn(),
    setSelectedModel: jest.fn(),
    onModelUpdate: mockOnModelUpdate,
    modelAccessGroups: []
  };

  beforeEach(() => {
    jest.clearAllMocks();
    mockModelPatchUpdateCall.mockClear();
    mockOnModelUpdate.mockClear();
  });

  it('should display "Disabled" when cache_control_injection_points is empty', async () => {
    render(<ModelInfoView {...defaultProps} />);
    
    await waitFor(() => {
      expect(screen.getByText('Disabled')).toBeInTheDocument();
    });
  });

  it('should display "Enabled" when cache_control_injection_points has values', async () => {
    const propsWithCacheControl = {
      ...defaultProps,
      modelData: {
        ...defaultProps.modelData,
        litellm_params: {
          ...defaultProps.modelData.litellm_params,
          cache_control_injection_points: [{ location: 'message', role: 'user', index: 0 }]
        }
      }
    };

    mockModelInfoV1Call.mockResolvedValueOnce({
      data: [
        {
          ...propsWithCacheControl.modelData,
          litellm_params: {
            ...propsWithCacheControl.modelData.litellm_params,
            cache_control_injection_points: [{ location: 'message', role: 'user', index: 0 }]
          }
        }
      ]
    });

    render(<ModelInfoView {...propsWithCacheControl} />);
    
    await waitFor(() => {
      expect(screen.getByText('Enabled')).toBeInTheDocument();
    });
  });

  it('should call onModelUpdate with empty cache_control_injection_points when saving with cache control disabled', async () => {
    render(<ModelInfoView {...defaultProps} />);
    
    await waitFor(() => {
      expect(screen.getByText('Disabled')).toBeInTheDocument();
    });

    // Click Edit Model button
    const editButton = screen.getByText('Edit Model');
    fireEvent.click(editButton);

    await waitFor(() => {
      // Click Save Changes button
      const saveButton = screen.getByText('Save Changes');
      fireEvent.click(saveButton);
    });

    await waitFor(() => {
      expect(mockModelPatchUpdateCall).toHaveBeenCalledWith(
        'test-access-token',
        expect.objectContaining({
          litellm_params: expect.objectContaining({
            cache_control_injection_points: []
          })
        }),
        'test-model-id'
      );
    });

    await waitFor(() => {
      expect(mockOnModelUpdate).toHaveBeenCalledWith(
        expect.objectContaining({
          litellm_params: expect.objectContaining({
            cache_control_injection_points: []
          })
        })
      );
    });
  });
});