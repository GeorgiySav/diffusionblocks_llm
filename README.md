# LLM

Built on top of [nn library](../nn%20library), consumed as a sibling checkout
rather than a submodule: `CMakeLists.txt` points at `../nn library` via
`NN_LIBRARY_DIR` and pulls it in with `add_subdirectory`. Because it's a plain,
separate git repo and not nested inside this one, changes to the library are
just `cd "../nn library" && git commit` as usual -- nothing here needs to be
updated or re-pinned.

If the checkout lives somewhere else (a different machine, CI), point at it
explicitly:

```bash
cmake -S . -B build -DNN_LIBRARY_DIR=/path/to/nn-library
```

## Build

```bash
cmake -S . -B build -G Ninja -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/llm
```
