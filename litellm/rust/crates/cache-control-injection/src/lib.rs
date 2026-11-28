//! `cache_control_injection_points`: the prompt cache breakpoints LiteLLM adds to a call where
//! the caller sent none. Python's `litellm/integrations/anthropic_cache_control_hook.py`.
//!
//! [`CacheControlInjectionPoint`] is the setting as a config spells it, the same for every API
//! format. Each format a route serves has a module that reads the points and places them on a
//! request of that format, as [`messages`] does for a Messages request.

pub mod messages;
mod point;

pub use point::CacheControlInjectionPoint;
