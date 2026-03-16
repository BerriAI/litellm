---
title: Guides
sidebar_label: Overview
---

import NavigationCards from '@site/src/components/NavigationCards';

**Guides** are focused references for specific LiteLLM SDK features and proxy configuration options. Each guide is self-contained — jump to any topic without reading the others first.

> Looking for step-by-step integration walkthroughs? See [Tutorials →](/docs/tutorials)

---

## Model Configuration

<NavigationCards
columns={2}
items={[
{
icon: "🎛️",
title: "Fine-tuned Models",
description: "Call fine-tuned OpenAI, Azure, and Vertex AI models using LiteLLM with custom model names.",
to: "/docs/guides/finetuned_models",
},
]}
/>

---

## Security & Networking

<NavigationCards
columns={2}
items={[
{
icon: "🔒",
title: "Security Settings",
description: "Configure SSL certificates and HTTP proxy settings for secure LiteLLM deployments.",
to: "/docs/guides/security_settings",
},
]}
/>

---

## AI Capabilities

<NavigationCards
columns={2}
items={[
{
icon: "🖥️",
title: "Code Interpreter",
description: "Use OpenAI's Code Interpreter tool to execute Python code in a secure, sandboxed environment.",
to: "/docs/guides/code_interpreter",
},
]}
/>
