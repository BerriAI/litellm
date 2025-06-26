# Docusaurus Versioning & Version Selector Research

## Summary

**Yes, Docusaurus absolutely supports version selectors** and offers robust versioning capabilities that allow users to discover new features before they're available in stable releases. This is exactly what you're looking for based on the Argo CD docs example.

## Key Features

### 1. Built-in Version Dropdown (`docsVersionDropdown`)

Docusaurus provides a built-in navbar component called `docsVersionDropdown` that creates a version selector dropdown:

```javascript
export default {
  themeConfig: {
    navbar: {
      items: [
        {
          type: 'docsVersionDropdown',
          position: 'left',
        },
      ],
    },
  },
};
```

### 2. Flexible Version Configuration

You can customize which versions appear and their labels:

```javascript
export default {
  themeConfig: {
    navbar: {
      items: [
        {
          type: 'docsVersionDropdown',
          versions: ['current', '3.0', '2.0'],
        },
      ],
    },
  },
};
```

Or with custom labels:

```javascript
export default {
  themeConfig: {
    navbar: {
      items: [
        {
          type: 'docsVersionDropdown',
          versions: {
            current: {label: 'Version 4.0'},
            '3.0': {label: 'Version 3.0'},
            '2.0': {label: 'Version 2.0'},
          },
        },
      ],
    },
  },
};
```

### 3. Version Structure Support

Docusaurus supports three types of versions:

- **Current Version**: The latest development version (usually labeled "Next")
- **Latest Version**: The stable production version (default route)
- **Past Versions**: Previous stable releases

### 4. Automatic Version Management

- Easy CLI command to create versions: `npm run docusaurus docs:version 1.1.0`
- Automatic directory structure creation
- URL routing handled automatically

## Real-World Examples

Many major projects use Docusaurus versioning with version selectors:

- **React Native** - Uses versioning for different React Native releases
- **Jest** - Maintains multiple documentation versions
- **Ionic** - Version selector for different Ionic versions
- **Strapi** - Has both v3 and v4 documentation with easy switching
- **MikroORM** - Version dropdown for different releases
- **React Navigation** - Multiple version support

## Configuration Options

### Version Behavior Control

```javascript
export default {
  presets: [
    '@docusaurus/preset-classic',
    docs: {
      // Include/exclude current version
      includeCurrentVersion: true,
      
      // Set which version is "latest"
      lastVersion: 'current',
      
      // Limit versions for dev/preview
      onlyIncludeVersions: ['current', '2.0', '1.0'],
      
      // Version-specific configuration
      versions: {
        current: {
          label: 'Next',
          path: 'next',
          banner: 'unreleased',
        },
        '2.0': {
          label: '2.0.x',
          path: '2.0',
          banner: 'none',
        },
        '1.0': {
          label: '1.0.x',
          path: '1.0',
          banner: 'unmaintained',
        },
      },
    },
  ],
};
```

### Version Banners

Docusaurus automatically adds banners to indicate version status:
- **"unreleased"** for versions above latest
- **"unmaintained"** for versions below latest
- **"none"** for the latest stable version

## Advanced Features

### Multiple Documentation Sets
- Support for multiple plugin instances
- Different versioning per documentation set
- Context-aware version switching

### Archived Versions
- Ability to link to archived versions on external CDNs
- Keeps build size manageable while preserving historical access

### Custom URL Patterns
- Flexible URL structure (`/docs/1.0.0/page` vs `/docs/v1/page`)
- SEO-friendly version routing

## Implementation Best Practices

1. **Start Simple**: Begin with basic versioning, add complexity as needed
2. **Keep Version Count Low**: Recommended to maintain <10 active versions
3. **Use Semantic Versioning**: Align with your project's release strategy
4. **Archive Old Versions**: Use external hosting for very old versions
5. **Test Version Navigation**: Ensure smooth user experience between versions

## Getting Started

1. **Enable Versioning**: Add the `docsVersionDropdown` to your navbar
2. **Create First Version**: Run `docusaurus docs:version 1.0.0`
3. **Configure Behavior**: Set up version-specific options
4. **Customize Appearance**: Adjust labels and styling as needed

## Comparison to Other Solutions

Unlike static site generators or custom solutions, Docusaurus provides:
- **Zero Configuration**: Works out of the box
- **SEO Optimized**: Proper meta tags and URL structure
- **User Experience**: Smooth navigation between versions
- **Developer Experience**: Simple CLI commands and configuration
- **Performance**: Optimized builds and lazy loading

This makes Docusaurus an excellent choice for documentation that needs version selection capabilities, exactly like what you saw in the Argo CD documentation.