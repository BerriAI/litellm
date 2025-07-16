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
  Divider,
} from "antd";
import { UploadOutlined, EyeOutlined, ReloadOutlined } from "@ant-design/icons";
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
  logo_url?: string;
}

const UIThemeSettings: React.FC<UIThemeSettingsProps> = ({
  userID,
  userRole,
  accessToken,
}) => {
  const { colors, updateColors, resetColors } = useTheme();
  const [form] = Form.useForm();
  const [loading, setLoading] = useState(false);
  const [themeConfig, setThemeConfig] = useState<UIThemeConfig>({});
  const [serverSaveLoading, setServerSaveLoading] = useState(false);

  // Load current theme settings
  useEffect(() => {
    if (accessToken) {
      fetchThemeSettings();
    }
  }, [accessToken]);

  // Sync form with theme context when colors change
  useEffect(() => {
    form.setFieldsValue(colors);
  }, [colors, form]);

  const fetchThemeSettings = async () => {
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
        setThemeConfig(data.values || {});
        // Update theme context with server values
        if (data.values) {
          updateColors(data.values);
        }
        form.setFieldsValue(data.values || {});
      }
    } catch (error) {
      console.error("Error fetching theme settings:", error);
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
      const formData = new FormData();
      formData.append('file', file as File);
      
      const proxyBaseUrl = getProxyBaseUrl();
      const url = proxyBaseUrl ? `${proxyBaseUrl}/upload/logo` : "/upload/logo";
      const response = await fetch(url, {
        method: 'POST',
        headers: {
          Authorization: `Bearer ${accessToken}`,
        },
        body: formData,
      });
      
      if (response.ok) {
        const result = await response.json();
        form.setFieldsValue({ logo_url: result.file_path });
        message.success('Logo uploaded successfully');
        onSuccess?.(result);
      } else {
        throw new Error('Upload failed');
      }
    } catch (error) {
      console.error('Error uploading logo:', error);
      message.error('Failed to upload logo');
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
        message.success("UI theme settings updated successfully!");
        setThemeConfig(values);
        updateColors(values);
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

  const resetToDefaults = () => {
    const defaults = {
      brand_color_primary: "#6366f1",
      brand_color_muted: "#8688ef",
      brand_color_subtle: "#8e91eb",
      brand_color_faint: "#c7d2fe",
      brand_color_emphasis: "#5558eb",
      logo_url: "",
    };
    
    form.setFieldsValue(defaults);
    updateColors(defaults);
  };



  if (!accessToken) {
    return null;
  }

  return (
    <div className="w-full mx-4 p-4">
      <Title>UI Theme Customization</Title>
      <Text className="text-gray-600 mb-4">
        Customize the appearance of your LiteLLM admin dashboard with custom colors and logo.
      </Text>

      {/* Logo Upload Section */}
      <Card className="mb-4">
        <Title className="text-lg mb-4">Custom Logo</Title>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          <div>
            <Text className="text-gray-600 mb-3">
              Upload a custom logo or provide a URL to personalize your dashboard.
            </Text>
            <div className="space-y-3">
              <Upload
                customRequest={handleLogoUpload}
                accept=".png,.jpg,.jpeg,.svg"
                maxCount={1}
                showUploadList={false}
              >
                <AntButton icon={<UploadOutlined />} size="large">
                  Upload Logo
                </AntButton>
              </Upload>
              
              <TextInput
                placeholder="Or enter logo URL"
                value={form.getFieldValue('logo_url')}
                onValueChange={(value) => form.setFieldsValue({ logo_url: value })}
              />
            </div>
          </div>
          
          {/* Logo Preview */}
          <div>
            {form.getFieldValue('logo_url') ? (
              <div>
                <Text className="block mb-2 font-medium">Logo Preview:</Text>
                <img 
                  src={form.getFieldValue('logo_url')} 
                  alt="Logo preview"
                  className="max-w-full max-h-24 object-contain border border-gray-200 rounded p-2 bg-white"
                  onError={(e) => {
                    (e.target as HTMLImageElement).style.display = 'none';
                  }}
                />
              </div>
            ) : (
              <div className="border-2 border-dashed border-gray-300 rounded-lg p-6 text-center">
                <Text className="text-gray-500">Logo preview will appear here</Text>
              </div>
            )}
          </div>
        </div>
      </Card>

      <Grid numItems={1} numItemsLg={2} className="gap-4">
        {/* Settings Form */}
        <Card>
          <Title className="text-lg mb-4">Theme Settings</Title>
          
          <Form
            form={form}
            layout="vertical"
            onFinish={handleSave}
            initialValues={themeConfig}
          >
            <Divider>Brand Colors</Divider>
            
            <Form.Item 
              label="Primary Color"
              name="brand_color_primary"
              tooltip="Main brand color used for buttons and accents"
            >
              <ColorPicker
                size="large"
                value={colors.brand_color_primary}
                onChange={(color) => handleColorChange('brand_color_primary', color)}
                showText
                format="hex"
              />
            </Form.Item>

            <Form.Item 
              label="Muted Color" 
              name="brand_color_muted"
              tooltip="Muted version of the primary color"
            >
              <ColorPicker
                size="large"
                value={colors.brand_color_muted}
                onChange={(color) => handleColorChange('brand_color_muted', color)}
                showText
                format="hex"
              />
            </Form.Item>

            <Form.Item 
              label="Subtle Color" 
              name="brand_color_subtle"
              tooltip="Subtle version of the primary color"
            >
              <ColorPicker
                size="large"
                value={colors.brand_color_subtle}
                onChange={(color) => handleColorChange('brand_color_subtle', color)}
                showText
                format="hex"
              />
            </Form.Item>

            <Form.Item 
              label="Faint Color" 
              name="brand_color_faint"
              tooltip="Very light version of the primary color"
            >
              <ColorPicker
                size="large"
                value={colors.brand_color_faint}
                onChange={(color) => handleColorChange('brand_color_faint', color)}
                showText
                format="hex"
              />
            </Form.Item>

            <Form.Item 
              label="Emphasis Color" 
              name="brand_color_emphasis"
              tooltip="Strong emphasis version of the primary color"
            >
              <ColorPicker
                size="large"
                value={colors.brand_color_emphasis}
                onChange={(color) => handleColorChange('brand_color_emphasis', color)}
                showText
                format="hex"
              />
            </Form.Item>



            <div className="flex space-x-4">
              <AntButton
                type="primary"
                htmlType="submit"
                loading={loading}
                size="large"
              >
                Save Changes
              </AntButton>
              
              <AntButton onClick={resetToDefaults} size="large">
                Reset to Defaults
              </AntButton>
            </div>
          </Form>
        </Card>

        {/* Live Theme Demo */}
        <Card>
          <Title className="text-lg mb-4">
            <EyeOutlined className="mr-2" />
            Live Theme Demo
          </Title>
          <Text className="text-gray-600 mb-3">
            See your brand colors in action across various UI components. Changes are applied in real-time!
          </Text>
          
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {/* Demo Buttons & Navigation */}
            <div className="space-y-4">
              <div>
                <Text className="font-medium text-gray-800 mb-2">Buttons</Text>
                <div className="space-y-2">
                  <button className="w-full px-3 py-1.5 bg-brand-primary text-white rounded text-sm font-medium hover:opacity-90 transition-opacity">
                    Primary Button
                  </button>
                  <button className="w-full px-3 py-1.5 bg-brand-muted text-white rounded text-sm font-medium hover:opacity-90 transition-opacity">
                    Secondary Button
                  </button>
                  <button className="w-full px-3 py-1.5 border border-brand-primary text-brand-primary rounded text-sm font-medium hover:bg-brand-faint transition-colors">
                    Outline Button
                  </button>
                </div>
              </div>
              
              <div>
                <Text className="font-medium text-gray-800 mb-2">Navigation Elements</Text>
                <div className="space-y-1">
                  <div className="bg-brand-primary text-white px-3 py-1.5 rounded text-sm">
                    Active Navigation Item
                  </div>
                  <div className="text-brand-primary px-3 py-1.5 rounded text-sm border border-brand-subtle">
                    Navigation Link
                  </div>
                  <div className="bg-brand-faint text-brand-emphasis px-3 py-1.5 rounded text-sm">
                    Highlighted Item
                  </div>
                </div>
              </div>
            </div>
            
            {/* Demo Cards */}
            <div className="space-y-3">
              <Text className="font-medium text-gray-800">Cards & Accents</Text>
              <div className="border-l-4 border-brand-primary bg-brand-faint p-3 rounded-md">
                <div className="text-brand-emphasis font-medium text-sm">Featured Card</div>
                <div className="text-gray-600 text-xs mt-1">This card uses your brand colors for accents</div>
              </div>
              <div className="bg-brand-subtle text-white p-3 rounded-md">
                <div className="font-medium text-sm">Branded Section</div>
                <div className="text-xs opacity-90">Content with brand background</div>
              </div>
            </div>
          </div>


          
          <div className="mt-4 p-3 bg-brand-faint border-l-4 border-brand-emphasis rounded-md">
            <div className="flex items-center">
              <div className="w-2 h-2 bg-brand-primary rounded-full mr-2"></div>
              <Text className="text-brand-emphasis font-medium text-sm">
                Real-time Preview Active
              </Text>
            </div>
            <Text className="text-gray-600 text-xs mt-1">
              Changes are immediately reflected in this demo and throughout the dashboard.
            </Text>
          </div>
        </Card>
      </Grid>
    </div>
  );
};

export default UIThemeSettings; 