# Contributing to LiteLLM Docs

Thanks for contributing to the LiteLLM documentation! This guide will help you run the docs site locally, make changes, and verify them before opening a PR.

## 1. Clone the docs repo

```bash
git clone https://github.com/BerriAI/litellm-docs.git
cd litellm-docs
```

## 2. Install dependencies

```bash
npm install
```

## 3. Start the docs site locally

```bash
npm start
```

Open http://localhost:3000.

The site uses Docusaurus 3, so most docs and blog changes reload automatically while the dev server is running.

## 4. Make your changes

Most documentation pages live in `docs/`.

Blog posts live in `blog/`.

Custom standalone pages live in `src/pages/`.

If you add, remove, or move docs pages, check whether `sidebars.js` needs to be updated.

## 5. Verify your changes

Before opening a PR, run:

```bash
npm run build
```

This catches broken links, invalid MDX, and other Docusaurus build issues.

## 6. Submit a PR

Create a branch:

```bash
git checkout -b docs/your-change-name
```

Commit your changes:

```bash
git add .
git commit -m "docs: update contributing guide"
```

Push your branch and open a PR against `BerriAI/litellm-docs`.
