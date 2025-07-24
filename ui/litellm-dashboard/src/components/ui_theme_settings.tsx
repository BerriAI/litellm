import React, { useState, useEffect } from "react";
import {
  Card,
  Title,
  Text,
  Grid,
  Button,
  TextInput,
  Col,
} from "@tremor/react";
import {
  Form,
  Upload,
  Button as AntButton,
  message,
  ColorPicker,
  Space,
} from "antd";
import { UploadOutlined, EyeOutlined } from "@ant-design/icons";
import type { UploadProps } from 'antd';
import { useTheme } from "@/contexts/ThemeContext";
import { getProxyBaseUrl } from "@/components/networking";

interface UIThemeSettingsProps {
  userID: string | null;
  userRole: string | null;
  accessToken: string | null;
}

interface UIThemeConfig {
  brand_color_primary?: string;
  brand_color_muted?: string;
  brand_color_subtle?: string;
  brand_color_faint?: string;
  brand_color_emphasis?: string;
  logo_url?: string | null;
}

const UIThemeSettings: React.FC<UIThemeSettingsProps> = ({
  userID,
  userRole,
  accessToken,
}) => {
  const { colors, updateColors, triggerLogoUpdate, setLogoUrl, logoUrl: contextLogoUrl } = useTheme();
  const [form] = Form.useForm();
  const [loading, setLoading] = useState(false);
  const [themeConfig, setThemeConfig] = useState<UIThemeConfig>({});
  const [logoPreview, setLogoPreview] = useState<string | null>(null);
  // Initialize logoUrlInput from context if available
  const [logoUrlInput, setLogoUrlInput] = useState<string>(() => {
    // Try to get from context first, then from localStorage as fallback
    const contextValue = contextLogoUrl || "";
    const localStorageValue = typeof window !== 'undefined' ? localStorage.getItem('litellm-logo-url') || "" : "";
    return contextValue || localStorageValue;
  });
  
  // Add debug logging whenever logoUrlInput changes
  useEffect(() => {
    console.log('[UI Theme Settings] logoUrlInput changed to:', logoUrlInput);
  }, [logoUrlInput]);

  // Load current theme settings when component mounts or accessToken changes
  useEffect(() => {
    if (accessToken) {
      console.log('[Component Mount] Starting with contextLogoUrl:', contextLogoUrl);
      // Always fetch theme settings to get the latest
      fetchThemeSettings();
    }
  }, [accessToken]);
  
  // Update input when context logo URL changes
  useEffect(() => {
    if (contextLogoUrl && contextLogoUrl !== logoUrlInput) {
      console.log('[Context Update] Updating logoUrlInput from context:', contextLogoUrl);
      setLogoUrlInput(contextLogoUrl);
      form.setFieldsValue({ logo_url: contextLogoUrl });
    }
  }, [contextLogoUrl]);

  // Sync form with theme context when colors change
  useEffect(() => {
    form.setFieldsValue(colors);
  }, [colors, form]);

  // Remove Form.useWatch as we're now using logoUrlInput as the primary source of truth
  // The form value is updated when logoUrlInput changes, not the other way around

  const fetchCurrentLogoUrl = async () => {
    try {
      const proxyBaseUrl = getProxyBaseUrl();
      const url = proxyBaseUrl ? `${proxyBaseUrl}/get_logo_url` : "/get_logo_url";
      const response = await fetch(url);
      if (response.ok) {
        const data = await response.json();
        return data.logo_url || '';
      }
    } catch (error) {
      console.error('Error fetching current logo URL:', error);
    }
    return '';
  };

  const fetchThemeSettings = async () => {
    // Always fetch the current logo URL from environment as the source of truth
    const currentLogoUrl = await fetchCurrentLogoUrl();
    console.log('Current logo URL from environment:', currentLogoUrl);
    
    try {
      const proxyBaseUrl = getProxyBaseUrl();
      const url = proxyBaseUrl ? `${proxyBaseUrl}/get/ui_theme_settings` : "/get/ui_theme_settings";
      const response = await fetch(url, {
        method: "GET",
        headers: {
          Authorization: `Bearer ${accessToken}`,
          "Content-Type": "application/json",
        },
      });

      if (response.ok) {
        const data = await response.json();
        console.log('Fetched theme settings:', data);
        
        const themeValues = data.values || {};
        setThemeConfig(themeValues);
        
        // Update theme context with server values
        updateColors(themeValues);
        
        // Always use the current environment logo URL as the source of truth
        // This ensures we show what's actually being displayed
        const logoUrl = currentLogoUrl || themeValues.logo_url || '';
        console.log('[fetchThemeSettings] Final logo URL:', logoUrl, 'Sources:', {
          currentLogoUrl,
          themeLogoUrl: themeValues.logo_url,
          currentInputValue: logoUrlInput
        });
        
        // Force update the input state
        setLogoUrlInput(logoUrl);
        setLogoUrl(logoUrl || null);
        
        // Also save to localStorage for persistence
        if (logoUrl && typeof window !== 'undefined') {
          localStorage.setItem('litellm-logo-url', logoUrl);
        }
        
        // Double-check it was set
        setTimeout(() => {
          console.log('[fetchThemeSettings] After set - logoUrlInput should be:', logoUrl);
        }, 0);
        
        // Clear preview for URLs, keep for data URLs
        if (logoUrl && !logoUrl.startsWith('data:')) {
          setLogoPreview(null);
        } else if (logoUrl && logoUrl.startsWith('data:')) {
          setLogoPreview(logoUrl);
        }
        
        // Set all form values including logo_url
        form.setFieldsValue({
          ...themeValues,
          logo_url: logoUrl
        });
      } else if (currentLogoUrl) {
        // If theme settings fetch failed but we have a current logo URL, use it
        console.log('Using current logo URL as fallback:', currentLogoUrl);
        setLogoUrlInput(currentLogoUrl);
        setLogoUrl(currentLogoUrl);
        form.setFieldsValue({ logo_url: currentLogoUrl });
      } else {
        console.error('Failed to fetch theme settings:', response.status, response.statusText);
        // Still set the current logo URL even if theme settings fail
        if (currentLogoUrl) {
          setLogoUrlInput(currentLogoUrl);
          setLogoUrl(currentLogoUrl);
          form.setFieldsValue({ logo_url: currentLogoUrl });
        }
      }
    } catch (error) {
      console.error("Error fetching theme settings:", error);
      // Still set the current logo URL even if theme settings fail
      if (currentLogoUrl) {
        setLogoUrlInput(currentLogoUrl);
        setLogoUrl(currentLogoUrl);
        form.setFieldsValue({ logo_url: currentLogoUrl });
      }
    }
  };

  const handleColorChange = (field: string, color: any) => {
    const hexColor = color.toHexString();
    // Update theme context for real-time preview
    updateColors({ [field]: hexColor });
    form.setFieldsValue({ [field]: hexColor });
  };

  const handleLogoUpload: UploadProps['customRequest'] = async (options) => {
    const { file, onSuccess, onError } = options;
    
    try {
      // Convert file to data URL for preview
      const reader = new FileReader();
      reader.onload = (e) => {
        const dataUrl = e.target?.result as string;
        setLogoPreview(dataUrl);
        form.setFieldsValue({ logo_url: dataUrl });
        setLogoUrlInput(dataUrl);
        // Update logo URL in real-time for preview
        setLogoUrl(dataUrl);
      };
      reader.readAsDataURL(file as File);
      
      // For production scaling, you would typically:
      // 1. Upload to a CDN (e.g., AWS S3, Cloudinary) and get a public URL
      // 2. Store only the URL in the database
      // 3. Use the CDN URL for display
      // For now, we'll use the data URL for preview
      message.success('Logo loaded successfully');
      onSuccess?.({ url: 'data-url' });
    } catch (error) {
      console.error('Error loading logo:', error);
      message.error('Failed to load logo');
      onError?.(error as Error);
    }
  };

  const handleSave = async (values: UIThemeConfig) => {
    setLoading(true);
    try {
      const proxyBaseUrl = getProxyBaseUrl();
      const url = proxyBaseUrl ? `${proxyBaseUrl}/update/ui_theme_settings` : "/update/ui_theme_settings";
      const response = await fetch(url, {
        method: "PATCH",
        headers: {
          Authorization: `Bearer ${accessToken}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify(values),
      });

      if (response.ok) {
        const result = await response.json();
        console.log('Theme settings saved successfully:', result);
        message.success("UI theme settings updated successfully!");
        setThemeConfig(values);
        updateColors(values);
        
        // Trigger logo update if logo was changed
        if (values.logo_url !== undefined) {
          console.log('Logo URL changed, triggering update:', values.logo_url);
          setLogoUrl(values.logo_url || null);
          // Small delay to ensure backend has processed the update
          setTimeout(() => {
            triggerLogoUpdate();
            // Force refetch theme settings to ensure sync
            fetchThemeSettings();
          }, 500);
        }
      } else {
        throw new Error("Failed to update settings");
      }
    } catch (error) {
      console.error("Error updating theme settings:", error);
      message.error("Failed to update theme settings");
    } finally {
      setLoading(false);
    }
  };

  const resetToDefaults = async () => {
    const defaults = {
      brand_color_primary: "#6366f1",
      brand_color_muted: "#8688ef",
      brand_color_subtle: "#8e91eb",
      brand_color_faint: "#c7d2fe",
      brand_color_emphasis: "#5558eb",
      logo_url: null, // Explicitly set to null to trigger removal
    };
    
    form.setFieldsValue({ ...defaults, logo_url: "" });
    setLogoPreview(null);
    setLogoUrlInput("");
    setLogoUrl(null); // Clear logo URL immediately
    
    // Clear from localStorage
    if (typeof window !== 'undefined') {
      localStorage.removeItem('litellm-logo-url');
    }
    
    // Save the defaults to trigger logo update
    await handleSave(defaults);
  };



  if (!accessToken) {
    return null;
  }

  return (
    <div className="w-full mx-auto max-w-6xl px-6 py-8">
      <div className="mb-8">
        <Title className="text-2xl font-bold mb-2">UI Theme Customization</Title>
        <Text className="text-gray-600">
          Customize the appearance of your LiteLLM admin dashboard with custom colors and logo.
        </Text>
      </div>

      {/* Logo Upload Section */}
      <Card className="mb-6 shadow-sm p-6">
        <div className="mb-6">
          <Title className="text-xl font-semibold mb-2">Custom Logo</Title>
          <Text className="text-gray-600">
            Upload a custom logo or provide a URL to personalize your dashboard.
          </Text>
        </div>
        
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          <div>
            <Text className="text-sm font-medium text-gray-700 mb-3 block">Upload Options</Text>
            <div className="space-y-3">
              <Upload
                customRequest={handleLogoUpload}
                accept=".png,.jpg,.jpeg,.svg"
                maxCount={1}
                showUploadList={false}
              >
                <AntButton 
                  icon={<UploadOutlined />} 
                  size="large"
                  className="w-full"
                  style={{ width: '100%' }}
                >
                  Upload Logo
                </AntButton>
              </Upload>
              
              <div>
                <Text className="text-xs text-gray-500 mb-1 block">Or enter logo URL:</Text>
                <TextInput
                  placeholder="https://example.com/logo.png"
                  value={logoUrlInput}
                  onValueChange={(value) => {
                    console.log('[TextInput] User changing value to:', value);
                    setLogoUrlInput(value);
                    form.setFieldsValue({ logo_url: value });
                    // Clear file preview when URL is entered
                    if (value && logoPreview) {
                      setLogoPreview(null);
                    }
                    // Update logo URL in real-time for preview
                    if (value) {
                      setLogoUrl(value);
                    } else {
                      setLogoUrl(null);
                    }
                  }}
                  className="w-full"
                />
              </div>
            </div>
          </div>
          
          {/* Logo Preview */}
          <div>
            <Text className="text-sm font-medium text-gray-700 mb-3 block">Preview</Text>
            {(logoPreview || logoUrlInput) ? (
              <div className="bg-gray-50 rounded-lg p-6 min-h-[120px] flex items-center justify-center">
                <img 
                  src={logoPreview || logoUrlInput} 
                  alt="Logo preview"
                  className="max-w-full max-h-24 object-contain"
                  onError={(e) => {
                    (e.target as HTMLImageElement).style.display = 'none';
                    if (logoPreview) {
                      setLogoPreview(null);
                    }
                  }}
                />
              </div>
            ) : (
              <div className="border-2 border-dashed border-gray-300 rounded-lg p-6 min-h-[120px] flex items-center justify-center text-center bg-gray-50">
                <Text className="text-gray-500 text-sm">Logo preview will appear here</Text>
              </div>
            )}
          </div>
        </div>
      </Card>

      <Grid numItems={1} numItemsLg={2} className="gap-6">
        {/* Settings Form */}
        <Card className="shadow-sm p-6">
          <div className="mb-6">
            <Title className="text-xl font-semibold mb-2">Theme Settings</Title>
            <Text className="text-gray-600">Configure your brand colors</Text>
          </div>
          
          <Form
            form={form}
            layout="vertical"
            onFinish={handleSave}
            initialValues={themeConfig}
          >
            {/* Hidden Form.Item for logo_url to ensure it's included in form submission */}
            <Form.Item name="logo_url" hidden>
              <input type="hidden" />
            </Form.Item>
            
            <div className="space-y-6">
              <div className="border-b border-gray-200 pb-4 mb-4">
                <Text className="text-lg font-medium text-gray-800">Brand Colors</Text>
              </div>
              
              <div className="space-y-5">
                <Form.Item 
                  label={
                    <span className="text-sm font-medium text-gray-700">
                      Primary Color
                      <Text className="text-xs text-gray-500 font-normal ml-2">
                        Main brand color for buttons and accents
                      </Text>
                    </span>
                  }
                  name="brand_color_primary"
                  className="mb-0"
                >
                  <ColorPicker
                    size="large"
                    value={colors.brand_color_primary}
                    onChange={(color) => handleColorChange('brand_color_primary', color)}
                    showText
                    format="hex"
                    style={{ width: '100%' }}
                  />
                </Form.Item>

                <Form.Item 
                  label={
                    <span className="text-sm font-medium text-gray-700">
                      Muted Color
                      <Text className="text-xs text-gray-500 font-normal ml-2">
                        Softer version of primary
                      </Text>
                    </span>
                  }
                  name="brand_color_muted"
                  className="mb-0"
                >
                  <ColorPicker
                    size="large"
                    value={colors.brand_color_muted}
                    onChange={(color) => handleColorChange('brand_color_muted', color)}
                    showText
                    format="hex"
                    style={{ width: '100%' }}
                  />
                </Form.Item>

                <Form.Item 
                  label={
                    <span className="text-sm font-medium text-gray-700">
                      Subtle Color
                      <Text className="text-xs text-gray-500 font-normal ml-2">
                        Light accent color
                      </Text>
                    </span>
                  }
                  name="brand_color_subtle"
                  className="mb-0"
                >
                  <ColorPicker
                    size="large"
                    value={colors.brand_color_subtle}
                    onChange={(color) => handleColorChange('brand_color_subtle', color)}
                    showText
                    format="hex"
                    style={{ width: '100%' }}
                  />
                </Form.Item>

                <Form.Item 
                  label={
                    <span className="text-sm font-medium text-gray-700">
                      Faint Color
                      <Text className="text-xs text-gray-500 font-normal ml-2">
                        Very light background color
                      </Text>
                    </span>
                  }
                  name="brand_color_faint"
                  className="mb-0"
                >
                  <ColorPicker
                    size="large"
                    value={colors.brand_color_faint}
                    onChange={(color) => handleColorChange('brand_color_faint', color)}
                    showText
                    format="hex"
                    style={{ width: '100%' }}
                  />
                </Form.Item>

                <Form.Item 
                  label={
                    <span className="text-sm font-medium text-gray-700">
                      Emphasis Color
                      <Text className="text-xs text-gray-500 font-normal ml-2">
                        Strong emphasis version
                      </Text>
                    </span>
                  }
                  name="brand_color_emphasis"
                  className="mb-0"
                >
                  <ColorPicker
                    size="large"
                    value={colors.brand_color_emphasis}
                    onChange={(color) => handleColorChange('brand_color_emphasis', color)}
                    showText
                    format="hex"
                    style={{ width: '100%' }}
                  />
                </Form.Item>
              </div>
            </div>

            <div className="flex gap-3 mt-8 pt-6 border-t border-gray-200">
              <AntButton
                type="primary"
                htmlType="submit"
                loading={loading}
                size="large"
                className="px-6"
                style={{ backgroundColor: '#1890ff', borderColor: '#1890ff' }}
              >
                Save Changes
              </AntButton>
              
              <AntButton 
                onClick={resetToDefaults} 
                size="large"
                className="px-6"
              >
                Reset to Defaults
              </AntButton>
            </div>
          </Form>
        </Card>

        {/* Live Theme Demo */}
        <Card className="shadow-sm p-6">
          <div className="mb-6">
            <div className="flex items-center gap-2 mb-2">
              <EyeOutlined className="text-gray-700" />
              <Title className="text-xl font-semibold">Live Theme Demo</Title>
            </div>
            <Text className="text-gray-600">
              See your brand colors in action across various UI components. Changes are applied in real-time!
            </Text>
          </div>
          
          <div className="space-y-6">
            {/* Buttons Section */}
            <div className="bg-gray-50 rounded-lg p-4">
              <Text className="text-sm font-semibold text-gray-800 mb-3 block">Buttons</Text>
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                <button className="px-4 py-2 bg-brand-primary text-white rounded-md text-sm font-medium hover:opacity-90 transition-opacity shadow-sm">
                  Primary Button
                </button>
                <button className="px-4 py-2 bg-brand-muted text-white rounded-md text-sm font-medium hover:opacity-90 transition-opacity shadow-sm">
                  Secondary Button
                </button>
                <button className="px-4 py-2 border-2 border-brand-primary text-brand-primary rounded-md text-sm font-medium hover:bg-brand-faint transition-colors">
                  Outline Button
                </button>
              </div>
            </div>
            
            {/* Navigation Elements */}
            <div className="bg-gray-50 rounded-lg p-4">
              <Text className="text-sm font-semibold text-gray-800 mb-3 block">Navigation Elements</Text>
              <div className="space-y-2">
                <div className="bg-brand-primary text-white px-4 py-2 rounded-md text-sm font-medium shadow-sm">
                  Active Navigation Item
                </div>
                <div className="text-brand-primary px-4 py-2 rounded-md text-sm font-medium border border-brand-subtle hover:bg-brand-faint transition-colors cursor-pointer">
                  Navigation Link
                </div>
                <div className="bg-brand-faint text-brand-emphasis px-4 py-2 rounded-md text-sm font-medium">
                  Highlighted Item
                </div>
              </div>
            </div>
            
            {/* Cards & Accents */}
            <div className="bg-gray-50 rounded-lg p-4">
              <Text className="text-sm font-semibold text-gray-800 mb-3 block">Cards & Accents</Text>
              <div className="space-y-3">
                <div className="border-l-4 border-brand-primary bg-white p-4 rounded-md shadow-sm">
                  <div className="text-brand-emphasis font-semibold text-sm mb-1">Featured Card</div>
                  <div className="text-gray-600 text-sm">This card uses your brand colors for accents</div>
                </div>
                <div className="bg-brand-subtle text-white p-4 rounded-md shadow-sm">
                  <div className="font-semibold text-sm mb-1">Branded Section</div>
                  <div className="text-sm opacity-90">Content with brand background</div>
                </div>
              </div>
            </div>
          </div>
          
          {/* Real-time Preview Indicator */}
          <div className="mt-6 p-4 bg-brand-faint border-l-4 border-brand-emphasis rounded-md">
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 bg-brand-primary rounded-full animate-pulse"></div>
              <Text className="text-brand-emphasis font-semibold text-sm">
                Real-time Preview Active
              </Text>
            </div>
            <Text className="text-gray-600 text-sm mt-1">
              Changes are immediately reflected in this demo and throughout the dashboard.
            </Text>
          </div>
        </Card>
      </Grid>
    </div>
  );
};

export default UIThemeSettings; 