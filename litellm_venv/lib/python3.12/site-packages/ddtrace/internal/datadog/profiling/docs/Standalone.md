Standalone Building and Testing
===============================

See the accompanying `Design.md` for comments on the high-level design and goals of these directories.
This document discusses some aspects of building and testing the native code in a standalone fashion, apart from the normal dd-trace-py build system.


Building
--------

The primary consumer of the build system here is setup.py, so many concessions are made with that goal in mind.
A helper script in the parent directory, `build_standalone.sh` can be used to manipulate the build system in a similar manner as `setup.py`, but which leverages the tooling we've added for testing and vetting the native code.


### Why

There are a few reasons why a developer would use `build_standalone.sh`:

* make sure this code builds without having to build other parts of the repo :)
* build and test the native code with sanitizers
* build the code with static analysis tools


### But

Note that `build_standalone.sh` is not currently part of this repo's release discipline, and if/when it is it will be run in a very prescriptive way in CI.
Thus, it's likely that this tool will not have the nice interface, error handling, and attention to detail one would expect from a first-class tool.
What does this mean for you?
Only that the tool may behave in unexpected and undelightful ways.


### Notes

Since artifacts, caches, and assets for these builds are stored in a subdirectory in the source tree, they will not interfere with the normal build system.
No need to delete things.
However, you may want to delete things if you switch branches.


### How

`build_standalone.sh` has some online documentation.
Here are the most useful commands.


#### Help
```sh
./build_standalone.sh
```


#### Build everything in release mode
```sh
./build_standalone.sh -- Release all
```


#### Build using clang

Usually, `setup.py` will use `gcc`, but this can be overridden for testing.

```sh
./build_standalone.sh --clang -- all
```


#### Build with cppcheck

CPPCheck is a powerful static analysis tool.
It doesn't work very well with cython-generated code, since cython has certain opinions.
It does work pretty well for `dd_wrapper`, though.

```sh
./build_standalone.sh --cppcheck -- dd_wrapper
```


#### Tests

Some components have tests.
Ideally these tests will be integrated into the repo's `pytest` system, but sometimes it's not convenient to do so.
For now, add the `_test` suffix to a target name.

```sh
./build_standalone.sh -- -- all_test
```


#### Sanitizers

The code can be built with sanitizers.

```sh
./build_standalone.sh --safety -- all
```

It can be useful to test with sanitizers enabled.

```sh
./build_standalone.sh --safety -- dd_wrapper_test
```
