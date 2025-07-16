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
import { UploadOutlined, EyeOutlined } from "@ant-design/icons";
import type { UploadProps } from 'antd';

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
  const [form] = Form.useForm();
  const [loading, setLoading] = useState(false);
  const [themeConfig, setThemeConfig] = useState<UIThemeConfig>({});
  const [previewColors, setPreviewColors] = useState<UIThemeConfig>({});

  // Load current theme settings
  useEffect(() => {
    if (accessToken) {
      fetchThemeSettings();
    }
  }, [accessToken]);

  const fetchThemeSettings = async () => {
    try {
      const response = await fetch("/get/ui_theme_settings", {
        method: "GET",
        headers: {
          Authorization: `Bearer ${accessToken}`,
          "Content-Type": "application/json",
        },
      });

      if (response.ok) {
        const data = await response.json();
        setThemeConfig(data.values || {});
        setPreviewColors(data.values || {});
        form.setFieldsValue(data.values || {});
      }
    } catch (error) {
      console.error("Error fetching theme settings:", error);
    }
  };

  const handleColorChange = (field: string, color: any) => {
    const hexColor = color.toHexString();
    setPreviewColors(prev => ({
      ...prev,
      [field]: hexColor
    }));
    form.setFieldsValue({ [field]: hexColor });
  };

  const handleLogoUpload: UploadProps['customRequest'] = async (options) => {
    const { file, onSuccess, onError } = options;
    
    try {
      const formData = new FormData();
      formData.append('file', file as File);
      
      const response = await fetch('/upload/logo', {
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
      const response = await fetch("/update/ui_theme_settings", {
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
        setPreviewColors(values);
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
    setPreviewColors(defaults);
  };

  const ColorPreview: React.FC<{ color?: string; label: string }> = ({ color, label }) => (
    <div className="flex items-center space-x-2">
      <div 
        className="w-8 h-8 rounded border border-gray-200"
        style={{ backgroundColor: color || "#ffffff" }}
      />
      <Text className="text-sm">{label}</Text>
    </div>
  );

  if (!accessToken) {
    return null;
  }

  return (
    <div className="w-full mx-4 p-6">
      <Title>UI Theme Customization</Title>
      <Text className="text-gray-600 mb-6">
        Customize the appearance of your LiteLLM admin dashboard with custom colors and logo.
      </Text>

      <Grid numItems={1} numItemsLg={2} className="gap-6">
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
                onChange={(color) => handleColorChange('brand_color_emphasis', color)}
                showText
                format="hex"
              />
            </Form.Item>

            <Divider>Logo</Divider>

            <Form.Item 
              label="Custom Logo"
              name="logo_url"
              tooltip="Upload a custom logo or provide a URL"
            >
              <Space direction="vertical" className="w-full">
                <Upload
                  customRequest={handleLogoUpload}
                  accept=".png,.jpg,.jpeg,.svg"
                  maxCount={1}
                  showUploadList={false}
                >
                  <AntButton icon={<UploadOutlined />}>
                    Upload Logo
                  </AntButton>
                </Upload>
                
                <TextInput
                  placeholder="Or enter logo URL"
                  value={form.getFieldValue('logo_url')}
                  onValueChange={(value) => form.setFieldsValue({ logo_url: value })}
                />
              </Space>
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

        {/* Preview */}
        <Card>
          <Title className="text-lg mb-4">
            <EyeOutlined className="mr-2" />
            Color Preview
          </Title>
          
          <div className="space-y-4">
            <ColorPreview 
              color={previewColors.brand_color_primary} 
              label="Primary Color" 
            />
            <ColorPreview 
              color={previewColors.brand_color_muted} 
              label="Muted Color" 
            />
            <ColorPreview 
              color={previewColors.brand_color_subtle} 
              label="Subtle Color" 
            />
            <ColorPreview 
              color={previewColors.brand_color_faint} 
              label="Faint Color" 
            />
            <ColorPreview 
              color={previewColors.brand_color_emphasis} 
              label="Emphasis Color" 
            />
          </div>

          {form.getFieldValue('logo_url') && (
            <div className="mt-6">
              <Text className="block mb-2 font-medium">Logo Preview:</Text>
              <img 
                src={form.getFieldValue('logo_url')} 
                alt="Logo preview"
                className="max-w-full max-h-32 object-contain border border-gray-200 rounded"
                onError={(e) => {
                  (e.target as HTMLImageElement).style.display = 'none';
                }}
              />
            </div>
          )}

          <div className="mt-6 p-4 bg-gray-50 rounded">
            <Text className="text-sm text-gray-600">
              Note: Theme changes will be reflected after the UI is rebuilt. 
              Changes to colors may require a page refresh to see full effects.
            </Text>
          </div>
        </Card>
      </Grid>
    </div>
  );
};

export default UIThemeSettings; 