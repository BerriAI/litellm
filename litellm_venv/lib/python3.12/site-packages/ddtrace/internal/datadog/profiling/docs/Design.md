Design and Justification
========================

Why does this code exist?
Why is it like this?
This document should help answer those questions.
This is not a deep-dive into the architecture and internal concepts.
Just a superficial overview.


Component Overview
------------------

The native code here is comprised of a few components.

* *dd_wrapper*:  ships C interfaces to libdatadog resources
* *ddup*:  Python interfaces to `dd_wrapper`
* *stack_v2*:  wraps echion, providing a shim layer to conform its concepts to those used in this repo
* *crashtracker*:  Python interfaces for `crashtracker`


All of the other components rely on `dd_wrapper`, since they rely on libdatadog.
It's problematic to do it any other way because this repo has strict size requirements, and double-shipping a large amount of native code would be quite wasteful.
Thus, the cmake definitions for all the other components end up building `dd_wrapper` anyway.

`ddup` and `crashtracker` provide Python interfaces, which are defined via cython.
Ordinarily we'd just build these in `setup.py`, but the resulting artifacts need some link-time settings, such as making sure RPATH is set in a way that allows the discovery of libdatadog.
These settings are cumbersome to propagate from the normal Pythonic infrastructure, so we just do it here.


Why?
----

### Temporary Strings

When Python calls into `dd_wrapper`, it may propagate strings which only have the lifetime of the call.
When libdatadog was originally included, its baseline API made some assumptions around string lifetimes.
Thus, the first problem this repo tried to solve was to ensure string lifetimes were compatible with the libdatadog API.


### Forks and Threads

As long as we use native thread, and as long as this system supports Python, it will have to contend with a multithreaded application which can `fork()` at any time.
Managing these interfaces safely is difficult to do in upstream libraries.
So we do it here.


Miscellaneous Considerations
----------------------------

It's important to realize that even though this code supports C++17, this repo ultimately needs to abide by the manylinux2014 specification.
The manylinux2014 build containers package an anachronistic (newer!) compiler, allowing it to use language features which are then linked in a way that breaks compatibility with the specification.
For instance, you can't use `std::filesystem`.
It will build--it'll even build in CI--but it will fail the auditwheel checks.

Carefully managing symbol visibility is vital.
The baseline assumption is that any library we link might be linked by another module.


Standalone Builds
-----------------

It's possible to build these components without using `setup.py`, which is useful for testing.
See `Standalone.md` for some details on this.
