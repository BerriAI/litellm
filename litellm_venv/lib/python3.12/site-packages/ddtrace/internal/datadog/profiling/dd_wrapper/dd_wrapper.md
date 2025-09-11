# dd_wrapper

This is experimental code that uses libdatadog for the collection and propagation of profiling data.
In theory this should just be a thin wrapper, but in practice handling the edge-cases (threads+forks) around even a thin wrapper requires some amount of state management.


## Historical context

This is a section on how the "python" side of the profiler used to work (and still does, until this and stack v2 become the default).
Traditionally, dd-trace-py maintained three separate sampling mechanisms for the collection of profiling data.
These are called "collectors" in dd-trace-py parlance, but we'll refer to them as "samplers" in conformance with the broader litearture.


### Stack sampler

This is a simple stack sampler that collects stack traces at a dynamic interval.
Supposedly, it's a fixed-interval profiler, but the interval is adjusted dynamically depending on the observed overhead.
This sampler lives in a Python thread.
When it wakes up, it looks at the current threads/tasks and collects stack traces for each of them.
It also looks at the current exceptions and collects similar data for them.

Although this sampler operates in its own thread, since its fundamental interaction requires taking the GIL, it never samples at a time when other threads are on-CPU (except if they're off-GIL in C code or in io or whatever, but that isn't an issue).


#### Stack Sampler v2

In a later patch, we'll add a native sampling mechanism for this.


### Lock sampler

This patches some common lock implementations to perform sampling.
As such, the sampling occurs from the active application thread.


### Memory sampler

This is very similar to the lock profiler, except for allocations.


### Potential interactions

Samplers might interrupt each other on a single thread.
Since the v1 stack sampler is written in Python, it may request allocations.
In turn, those allocations may trigger the memory sampler.
This requires the caller of the library to specify what kind of sample they are taking.


## Current state

For various reasons, we're validating profiling concepts where as much (or all!) of the work is done from native threads.
This is a bit of a departure from the traditional dd-trace-py model, but the technique is well-represented in the community.
This is comprised of two main parts right now:

* Moving the stack sampler to a native, off-GIL thread
* Moving the collection apparatus to native code

The problem here is that in Python it's fairly common to periodically call `fork()` as a worker lifecycle management technique.
When this happens, then in the child:

* Execution resumes from the same point as the parent--one way to think about this is, execution resumes from the thread that called fork
* The heap is configured to be copied-on-write

This means any intermediate heap-stored state managed by a sibling thread (i.e., a thread which did not call `fork()`) may become corrupt in the child.
We need to protect or re-initialize such state when possible.


## Data organization in this code

Roughly, the data is organized in the following way


### Samples

A Sample is a collection of data representing one observation.
All types of observations (e.g., allocations, locks, CPU) are represented by the same data structure, since the libdatadog API assumes any sample may contain one or all of the possible sample types.
When `start_sample()` is called, an opaque pointer is returned, which must then be passed to all subsequent operations on the sample.

In a given moment, in a world where samplers may be on native threads, there may be multiple samples in flight.
A sampler may also be interrupted by another sampler on the same thread.
Samplers should not be interrupted by the same type of sampler in other threads.
However, this last case hardly matters, since the interrupted context will resume using the Sample it was given.

Samples are interacted with transactionally, for example

1. `ddup_start_sample()` (every sample begins with a call to `start_sample()`)
2. adding frames, walltime, tags, etc
3. ddup_flush_sample() (every sample ends with a call to `flush_sample()`)

The act of flushing a sample stores its data in a ddog_prof_Profile object (which is wrapped by Profile in this code).
it also releases the Sample (don't reuse Samples in application code!)

There's one wrinkle here.
The navigation through frame data (unwinding) may result in temporary strings.
We need to cache these strings.
In order to minimize overhead, strings are cached (and de-duplicated) in a cache attached to the Profile rather than the Sample.


### Profile

A Profile wraps the collection of samples.
A Profile is periodically flushed to the Datadog backend during an upload operation.
A profile actually manages its own internal cache of strings, which makes it slightly unfortunate that we de-duplicate strings _twice_.
This is a little bit of a wart, but in practice we're still way under the memory overhead of the pure-Python collection system in mainline dd-trace-py.

For simplicity, the Profile object maintains two `ddog_prof_Profile`s using a red-black swap mechanism.
When one Profile is consumed by the uploader, it gets swapped with the other profile.
This probably isn't actually necessary anymore.


### Uploader

A special class for managing the upload operations.
It holds onto some state which is attached to a Profiling upload.
This is actually a little bit delicate because the underlying libdatadog fixture calls an HTTP library which may be more prone to tearing during `fork()` than other parts of the code.
Accordingly, we register `atfork()` handlers in `interface.cpp` to try and protect it.


### Builders

Conceptually, the Uploader can be reused.
However, there is some anxiety around exactly what degree of safety we can guarantee when an upload (not obvious: uploads happen in a thread controlled by a libdatadog dependency) is cut by a `fork()`, so we rebuild the uploader fresh every time.
