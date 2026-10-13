# ADR 001: Only type requests inside Rust

Status: Accepted

## Context

The python SDK had typed interfaces for requests living in the bridge in Python. However, there were multiple ways for requests to get there, and so the typing is very shallow and ultimately is just a shim placed on top of untyped args and kwargs.

## Decision

For the time being, avoid trying to restrict or type requests and their possible arguments inside python. Just pass the raw object over into rust, and have a translation layer at the boundary which handles it.

## Consequences

Some python interfaces are a lot less clear about what they expect as input. The typing of the rust module is entirely internal to it, and must be inferred from other channels.