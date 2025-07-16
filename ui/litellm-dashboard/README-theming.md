# Real-Time UI Theme Customization

This document explains how the real-time UI theming system works in the LiteLLM dashboard.

## Overview

The LiteLLM dashboard now supports real-time theme customization, allowing administrators to change brand colors and see the changes immediately across the entire interface without requiring a page refresh.

## Architecture

### 1. Theme Context (`src/contexts/ThemeContext.tsx`)
- **Purpose**: Manages theme state globally using React Context
- **Features**:
  - Real-time color updates via CSS variables
  - Persistent storage in localStorage
  - Automatic color palette generation
  - RGB conversion for Tremor compatibility

### 2. CSS Variables (`src/app/globals.css`)
- **Custom Properties**: `--brand-primary`, `--brand-muted`, etc.
- **Utility Classes**: `.bg-brand-primary`, `.text-brand-primary`, etc.
- **Tremor Integration**: RGB format variables for Tremor components

### 3. UI Theme Settings (`src/components/ui_theme_settings.tsx`)
- **Color Pickers**: Real-time color selection with live preview
- **Logo Upload**: Custom logo functionality
- **Live Demo**: Interactive preview showing theme changes
- **Server Sync**: Save/load settings from backend

## Usage

### For Developers

#### Using Theme Colors in Components
```tsx
// CSS classes (recommended)
<div className="bg-brand-primary text-white">
  <button className="border-brand-emphasis">Primary Button</button>
</div>

// CSS variables (for custom styles)
<div style={{ backgroundColor: 'var(--brand-primary)' }}>
  Custom styled element
</div>

// Theme context (for dynamic logic)
import { useTheme } from '@/contexts/ThemeContext';

const MyComponent = () => {
  const { colors, updateColors } = useTheme();
  return <div style={{ color: colors.brand_color_primary }}>...</div>;
};
```

#### Available CSS Variables
- `--brand-primary`: Main brand color
- `--brand-muted`: Softer variant
- `--brand-subtle`: Subtle variant  
- `--brand-faint`: Very light variant
- `--brand-emphasis`: High contrast variant
- `--brand-50` through `--brand-900`: Auto-generated palette

#### Available CSS Classes
- Background: `.bg-brand-primary`, `.bg-brand-muted`, etc.
- Text: `.text-brand-primary`, `.text-brand-muted`, etc.
- Border: `.border-brand-primary`, `.border-brand-muted`, etc.

### For Administrators

1. **Access**: Navigate to Settings → UI Theme in the admin dashboard
2. **Color Selection**: Use color pickers to select brand colors
3. **Live Preview**: See changes immediately in the demo section
4. **Save**: Click "Save Theme Settings" to persist changes
5. **Reset**: Use "Reset to Defaults" to restore original colors

## Technical Details

### Color Processing
1. **Input**: Hex colors from color pickers
2. **CSS Variables**: Injected into document root
3. **Palette Generation**: Automatic creation of 50-900 color variants
4. **RGB Conversion**: For Tremor component compatibility

### Persistence
- **Client-side**: localStorage for immediate UI updates
- **Server-side**: Backend API for permanent storage
- **Sync**: Form syncs with theme context on load

### Performance
- **CSS Variables**: Efficient browser-native theme switching
- **Debounced Updates**: Optimized for real-time color picker changes
- **Minimal Re-renders**: Context optimization prevents unnecessary updates

## Troubleshooting

### Colors Not Updating
1. Check browser console for JavaScript errors
2. Verify CSS variables are being set in DevTools
3. Clear localStorage: `localStorage.removeItem('litellm-theme-colors')`

### Tremor Components Not Themed
- Ensure Tailwind config includes CSS variable fallbacks
- Check that Tremor classes are using the updated color definitions

### Server Sync Issues
- Verify backend endpoints are accessible
- Check network tab for API call failures
- Ensure proper authentication tokens

## Browser Support
- **CSS Variables**: Modern browsers (IE11+ with polyfill)
- **Local Storage**: All modern browsers
- **Color Input**: Modern browsers with color picker support

## Future Enhancements
- Dark mode support
- Multiple theme presets
- Advanced typography customization
- Component-specific theming
- Theme import/export functionality 