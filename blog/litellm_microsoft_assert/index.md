---
slug: litellm-microsoft-assert
title: "Announcing LiteLLM x Microsoft ASSERT"
date: 2026-06-03T10:00:00
authors:
  - mubashir
  - krrish
description: "LiteLLM now integrates with Microsoft ASSERT for policy-driven agent evaluation — catch safety and quality defects before they reach production."
hide_table_of_contents: false
---

Today we're excited to officially launch **LiteLLM x Microsoft ASSERT** — bringing policy-driven agent evaluation to every model running through the LiteLLM AI Gateway.

{/* truncate */}

## What is ASSERT?

ASSERT is Microsoft's open-source framework for policy-driven agent evaluation, built on a proven Microsoft Research approach. ASSERT takes your organizational policies and requirements as input, systematically generates targeted evaluation scenarios, and surfaces safety and quality defects before they reach production.

## Why this matters

As teams ship agents into production, the gap between "it works in a demo" and "it behaves under our policies" is where real risk lives. ASSERT closes that gap by turning your written policies into concrete, testable evaluation scenarios — and now those evaluations run against any of the 100+ LLM providers LiteLLM supports, through a single unified interface.

## How it works with LiteLLM

- **Bring your policies** — ASSERT ingests your organizational policies and requirements.
- **Generate scenarios** — ASSERT systematically produces targeted evaluation scenarios.
- **Run through LiteLLM** — evaluate any model behind the LiteLLM Gateway with consistent auth, logging, and cost tracking.
- **Surface defects early** — catch safety and quality issues before they reach production.

## Get started

ASSERT is open source. Point it at your LiteLLM Gateway endpoint and start evaluating your agents against your own policies today.

- [Set up the LiteLLM Gateway](/docs/proxy/quick_start) — get a gateway endpoint running in minutes.
- [Microsoft ASSERT on GitHub](https://github.com/microsoft/assert) — install ASSERT and run it against your gateway.
