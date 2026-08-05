import { theme, ThemeConfig } from "antd";

export const darkModeColors = {
  bgBase: "#0e0e0e",
  bgElevated: "#151515",
  bgSurface: "#1a1a1a",
  bgHover: "#1f1f1f",

  border: "#1e1e1e",
  borderHover: "#282828",

  textPrimary: "#e5e7eb",
  textSecondary: "#a1a1aa",
  textMuted: "#71717a",

  accentBlue: "#3b82f6",
  accentError: "#ef4444",
} as const;

export const getAntdTheme = (isDarkMode: boolean): ThemeConfig => ({
  algorithm: isDarkMode ? theme.darkAlgorithm : theme.defaultAlgorithm,
  token: isDarkMode
    ? {
        colorBgContainer: darkModeColors.bgElevated,
        colorBgElevated: darkModeColors.bgElevated,
        colorBorder: darkModeColors.border,
        colorText: darkModeColors.textPrimary,
        colorTextSecondary: darkModeColors.textSecondary,
      }
    : {},
  components: isDarkMode
    ? {
        Menu: {
          iconSize: 18,
          fontSize: 14,
          itemBg: darkModeColors.bgBase,
          itemColor: darkModeColors.textSecondary,
          itemHoverColor: darkModeColors.textPrimary,
          itemHoverBg: darkModeColors.bgElevated,
          itemSelectedColor: darkModeColors.textPrimary,
          itemSelectedBg: darkModeColors.bgElevated,
          subMenuItemBg: darkModeColors.bgBase,
        },
        Card: {
          colorBgContainer: darkModeColors.bgElevated,
          colorBorderSecondary: darkModeColors.border,
        },
        Table: {
          colorBgContainer: darkModeColors.bgElevated,
          headerBg: darkModeColors.bgBase,
          rowHoverBg: darkModeColors.bgHover,
          borderColor: darkModeColors.border,
        },
        Input: {
          colorBgContainer: darkModeColors.bgBase,
          colorBorder: darkModeColors.border,
          hoverBorderColor: darkModeColors.borderHover,
          activeBorderColor: darkModeColors.accentBlue,
        },
        InputNumber: {
          colorBgContainer: darkModeColors.bgBase,
          colorBorder: darkModeColors.border,
          hoverBorderColor: darkModeColors.borderHover,
          handleBg: darkModeColors.bgHover,
        },
        Select: {
          colorBgContainer: darkModeColors.bgBase,
          colorBorder: darkModeColors.border,
          optionSelectedBg: darkModeColors.bgHover,
          optionActiveBg: darkModeColors.bgHover,
          selectorBg: darkModeColors.bgBase,
        },
        DatePicker: {
          colorBgContainer: darkModeColors.bgBase,
          colorBorder: darkModeColors.border,
        },
        Modal: {
          contentBg: darkModeColors.bgElevated,
          headerBg: darkModeColors.bgElevated,
          footerBg: darkModeColors.bgElevated,
        },
        Dropdown: {
          colorBgElevated: darkModeColors.bgElevated,
          controlItemBgHover: darkModeColors.bgHover,
        },
        Button: {
          defaultBg: darkModeColors.bgElevated,
          defaultBorderColor: darkModeColors.border,
        },
        Tabs: {
          inkBarColor: darkModeColors.accentBlue,
        },
        Form: {
          labelColor: darkModeColors.textPrimary,
        },
      }
    : {
        Menu: {
          iconSize: 18,
          fontSize: 14,
        },
      },
});
