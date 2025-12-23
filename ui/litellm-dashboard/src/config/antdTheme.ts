import { theme, ThemeConfig } from "antd";

/**
 * Shared dark mode color tokens used throughout the application.
 * These values provide consistency for dark mode styling.
 *
 * Color palette designed with subtle contrast to avoid jarring transitions:
 * - bgBase (#141414) - Main page background
 * - bgElevated (#1a1a1a) - Sidebar, navbar, cards
 * - bgSurface (#1e1e1e) - Interior content areas, panels
 * - bgHover (#252525) - Hover states, active items
 */
export const darkModeColors = {
  // Backgrounds (subtle gradient for less harsh contrast)
  bgBase: "#141414",     // Main background
  bgElevated: "#1a1a1a", // Cards, sidebar, navbar
  bgSurface: "#1e1e1e",  // Interior content areas
  bgHover: "#252525",    // Hover states

  // Borders (more visible for better definition)
  border: "#2a2a2a",
  borderHover: "#3a3a3a",

  // Text colors
  textPrimary: "#e5e7eb",
  textSecondary: "#a1a1aa",
  textMuted: "#71717a",

  // Accent colors
  accentBlue: "#3b82f6",
  accentError: "#ef4444",
} as const;

/**
 * Creates an Ant Design theme configuration based on the current dark mode state.
 * Use this with ConfigProvider to ensure consistent theming across the app.
 *
 * @param isDarkMode - Whether dark mode is currently active
 * @returns ThemeConfig object for Ant Design ConfigProvider
 */
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
