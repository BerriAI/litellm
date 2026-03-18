# LiteLLM Adopters

This directory contains data for organizations that use LiteLLM in production.

## Adding Your Organization

We've made it super easy to add your organization! Just follow the steps below.

### Quick Add (Recommended)

**[Edit adopters.json on GitHub →](https://github.com/BerriAI/litellm/edit/main/docs/my-website/src/data/adopters/adopters.json)**

This will open the GitHub editor in your browser where you can:

1. Add your organization's entry to the JSON array
2. Commit your changes
3. GitHub will automatically create a pull request for you!

No need to clone the repository or set up a development environment.

### JSON Format

Add your organization to the array in `adopters.json`:

```json
{
  "name": "Your Organization Name",
  "logoUrl": "https://yoursite.com/logo.svg",
  "url": "https://yourcompany.com",
  "description": "Brief description of how you use LiteLLM (shown on hover)"
}
```

### Fields

- **`name`** (required): Your organization's display name
- **`logoUrl`** (required): URL to your logo - can be either:
  - External URL: `https://yoursite.com/logo.svg` (easiest!)
  - Local path: `/img/adopters/your-logo.svg` (requires uploading logo file)
- **`url`** (optional): Your organization's website (makes the logo clickable)
- **`description`** (optional): Brief description shown when users hover over your logo

### Logo Options

#### Option 1: External URL (Easiest)

Simply provide a direct link to your logo hosted anywhere:

```json
"logoUrl": "https://yourcompany.com/assets/logo.svg"
```

#### Option 2: Local Logo (Better Performance)

If you prefer to host the logo locally:

1. Add your logo to `docs/my-website/static/img/adopters/your-company.svg`
2. Reference it as: `"logoUrl": "/img/adopters/your-company.svg"`

**Logo Specifications:**

- **Format**: SVG preferred (PNG also acceptable)
- **Dimensions**: 240x160px or similar 3:2 ratio recommended
- **Background**: Transparent or white background works best

### Example

```json
{
  "name": "Acme Corporation",
  "logoUrl": "https://acme.com/logo.svg",
  "url": "https://acme.com",
  "description": "Using LiteLLM to route requests across 50+ LLM providers"
}
```

### Display Order

Adopters are displayed alphabetically by organization name, so your position will be determined automatically.

### Need Help?

If you have questions about adding your organization:

- Ask in [GitHub Discussions](https://github.com/BerriAI/litellm/discussions)
- Join our [Discord community](https://discord.com/invite/wuPM9dRgDw)

Thank you for supporting LiteLLM! 🚅
