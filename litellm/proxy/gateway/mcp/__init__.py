"""MCP Gateway v2 — from-scratch rewrite.

Flat-concern layout; direction (inbound -> pipeline -> outbound -> leaves) is
enforced by the import-linter contract in .importlinter, not by nesting.
See CLAUDE.md for the folder map and per-module conventions.
"""
