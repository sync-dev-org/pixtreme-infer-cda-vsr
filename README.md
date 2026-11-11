# pixtreme.infer.cda_vsr

The distribution `pixtreme-infer-cda-vsr` provides the import package
`pixtreme.infer.cda_vsr`: the CDA-VSR architecture module (causal recurrent
online video super-resolution) for the pixtreme inference companion. The module
reads the scale from a caller-supplied checkpoint's safetensors header, and the
caller selects the precision. The family contract does not fix a scale.

The wheel owns only the `pixtreme/infer/cda_vsr/` subtree. The namespace root
`pixtreme/__init__.py` is owned by [pixtreme](https://github.com/sync-dev-org/pixtreme)
and `pixtreme/infer/__init__.py` by
[pixtreme-infer](https://github.com/sync-dev-org/pixtreme-infer).

## Installation

Use Linux x86_64 with CPython 3.12 or later and an NVIDIA GPU and driver compatible
with CUDA 13. Install the distribution from PyPI:

```console
pip install pixtreme-infer-cda-vsr
```

For a project managed with uv:

```console
uv add pixtreme-infer-cda-vsr
```

The Linux wheel includes the compiled module; installation and inference do not
require a compiler or a TensorRT SDK. Engine construction and execution use
tensoracle's separately installed native library.

The wheel also contains the PEP 561 marker `py.typed` and `_native.pyi`, a stub
that nanobind generates from the compiled module during the build, so IDEs and
type checkers see the constructor arguments, methods and properties of `Model`.

## Inference

Construct `Model` with a safetensors checkpoint, the computation precision and
an exact input size. Construction inspects the checkpoint header without
initializing CUDA or allocating GPU memory. The immutable `model.contract`
describes the scale, precision, input size and state footprint;
`model.family_contract` declares the RGB Frame and auxiliary inputs.

```python
import pixtreme as px
from pixtreme.infer.cda_vsr import Model

model = Model(
    weights="/path/to/checkpoints/best-inference.safetensors",
    precision="fp16",
    input_hw=(270, 480),
    use_cuda_graph=False,
)
stream = px.core.Stream(device=0, non_blocking=True)
model.prepare(device=0, stream=stream.ptr)
try:
    # Make the RGB Frame and mv/residual NDArrays ready on this stream:
    # output = model(frame, mv=mv, residual=residual, reset=False, stream=stream.ptr)
    pass
finally:
    stream.synchronize()
    model.close()
```

`prepare` rechecks the checkpoint contract and opens the I and P engines.
The first request and every explicit reset select I; subsequent requests select P.
Each model owns two persistent GPU state buffers. Reset clears those buffers;
it does not allocate new history. Multiple models keep independent histories.
Calls before preparation, after failed preparation and after `close()` are
rejected. Close is idempotent. Finish pending work before closing.

A context manager prepares a newly constructed model with the default device and
stream and closes it on exit. Prepare explicitly before entering the context
when using another stream. Omitting a request's `stream` selects the legacy
default stream; it does not select CuPy's current stream.

## CUDA Graph inference

Set `use_cuda_graph=True` in the same constructor and prepare with an explicit
non-default stream. Core warms up and captures four routes: I and P for each
direction of the two state buffers. Each route has a dedicated TensorRT execution.
Preparation restores initial history and consumes no inference request.

Core validates each request before copying inputs, resetting state or replaying.
Inputs are copied into fixed GPU storage and each returned Frame owns independent
pixels. The normal path and graph path share the same state and route logic.
Later requests, resets and model cleanup do not modify returned Frames.

One model uses one shape, precision, device and stream. Create another model to
change those settings. A failed preparation releases partial resources and leaves
the model unavailable for inference. There is no automatic fallback or recapture.

## Prepare a checkpoint

The module accepts checkpoints in the safetensors format only. Use the checkpoint
producer's export to create a safetensors file with the header metadata
`cda_vsr.residual_unit=luma-range`.

Converting a `.pth` file with `tensoracle.convert` only changes the file format;
the result lacks the required residual-unit metadata and is rejected by this
module. Adding that metadata by hand does not replace the producer's export.
The producer's export must also apply any training-specific adjustments to the
weights. Manually converting a checkpoint and adding metadata does not perform
those adjustments and can change inference output.

The checkpoint producer chooses the weights for inference. Supply exactly one
weight set, under any common prefix or with no prefix; a file containing both
`params` and `params_ema` is rejected. Preserve all producer-written metadata if
copying an exported weight set to a separate safetensors file.

Inference and engine building do not import PyTorch.

## Checkpoint validation and engines

Pass the checkpoint as `Model(weights=..., precision=..., input_hw=(H, W))`.
During preparation, tensoracle opens compatible I and P engines from its cache,
builds missing engines, and manages the files. The cache follows `XDG_CACHE_HOME`
(otherwise `~/.cache`) under `tensoracle`. The graph identity includes precision,
I/P route and residual gate. Each combination has a separate cache entry.
I/P selection happens in the native model; neither engine has a conditional route
or an `is_first` input. Inputs, outputs and recurrent state remain float32 in both
fp32 and fp16 engines.

Checkpoint paths accept strings or `os.PathLike[str]` objects. The format is checked
from the file contents, regardless of filename extension. PyTorch checkpoints
(`torch.save` ZIP and legacy pickle) and engine files are rejected with instructions
to use the checkpoint producer's export described above. The constructor checks
that the safetensors header is readable; it never imports PyTorch or TensorRT,
allocates device memory, builds an engine, or writes
to the cache. It identifies weight sets by the common prefix before the documented
tensor names; names without a prefix form an unnamed set. Even a partially present
set counts. If there are multiple sets or none, construction and preparation report the discovered
names and request a checkpoint containing exactly one inference set, before reading
tensor values or creating a cached engine.

The safetensors header metadata (`__metadata__`) must contain
`cda_vsr.residual_unit=luma-range`, recording the residual unit used during training.
Construction and preparation reject absent metadata, a missing key,
or any other value with the observed value (or its absence) and the accepted value
`luma-range`, before reading tensor values or creating an engine cache. Preserve
the producer's metadata when copying an exported weight set.
Checkpoints without this record are rejected regardless of scale.

The same metadata selects the residual gate. When `cda_vsr.gate` is `sigmoid`,
the network uses `aligned * sigmoid(w)`, where `w` is
the output of the three `conv_weight_l` convolutions and their intervening
LeakyReLU activations. When the gate key is absent, the network uses
`aligned * (1 + w)`. Any other value
is rejected by construction and preparation with the observed value
and instructions to use `sigmoid` or omit the key, before reading tensor values or
creating an engine cache. The checkpoint producer records the gate used during
training. Both gate forms are computed in float32, including in fp16 engines.

The constructor reads the shape of `upconv1.weight` within that sole set from the header and
derives scale from its output channel count, `3 * scale**2`. The tensor must have
four dimensions and scale must be a positive integer; a missing tensor or an
incompatible shape is rejected with the observed and expected values. The
persistent state footprint follows the architecture and requested dimensions.
Construction does not read tensor values.

Preparation validates all 179 tensor names, shapes and float32 dtypes in the sole
set before saving an engine in the cache. Extra tensor names under that prefix
are rejected; for an unnamed set, every tensor belongs to the set. Unrelated
tensors outside a named set are ignored. The reconstruction head and base image resize use
the checkpoint's scale; the remaining network structure uses 64 features, 3 feature
extraction blocks, 24 I reconstruction blocks, 12 P reconstruction blocks and 4 DCN
groups. An incompatible checkpoint may therefore be constructed successfully and fail
when prepared. Each engine accepts the exact requested `(H, W)`
and precision (`"fp32"` or `"fp16"`). Unsupported precision and dimensions are
rejected before the checkpoint is read.

## Prepare the engine cache

Installing this module installs tensoracle and the TensorRT runtime it uses. Building
an engine on a cache miss and opening a cached engine both need a compatible NVIDIA
GPU; neither imports PyTorch. A checkpoint that fails validation does not leave an
engine in the cache.

To complete the initial build before processing frames, prepare a model and close it:

```python
from pixtreme.infer.cda_vsr import Model

model = Model("/path/to/checkpoints/best-inference.safetensors", precision="fp16", input_hw=(270, 480))
model.prepare(device=0)
model.close()
```

Preparation writes no caller-managed engine file; tensoracle owns the two cached
engine files. Later preparations with the same weights, dimensions and precision
reuse both engines. Keep checkpoints outside the repository; weights and engines
are not included in the distribution.

## Frame, state and stream requirements

Call models with an RGB float32 Frame and C-contiguous CUDA float32 `pixtreme.NDArray` inputs
`mv=(2,H,W)` and `residual=(1,H,W)` on the instance's device. Motion vectors use
channel 0=x, 1=y and `reference = current + mv` in LR luma pixels. Residual values are
signed luma residuals divided by the stream's luma black-to-white range, following
`profile=hevc-mv-residual/4`: `219 * 2^(bit depth - 8)` for limited range and
`255 * 2^(bit depth - 8)` for full range. Streams without a range indication use
limited range. Values, units, matching picture identity,
model reference structure, and absence of concurrent calls to
one instance are caller preconditions. The caller decides when to reset.

Make inputs ready on the stream passed to `prepare` and pass its numeric handle
as `stream` on every request. Wrap DLPack producers, such as CuPy arrays, with
`pixtreme.NDArray.from_dlpack` before passing auxiliary inputs:

```python
import cupy as cp
import pixtreme as px

# frame, model and stream use the same CUDA device; model is already prepared.
with cp.cuda.ExternalStream(stream.ptr):
    h, w = model.contract.input_hw
    mv_owner = cp.zeros((2, h, w), dtype=cp.float32)
    residual_owner = cp.zeros((1, h, w), dtype=cp.float32)
    mv = px.NDArray.from_dlpack(mv_owner, stream=stream.ptr)
    residual = px.NDArray.from_dlpack(residual_owner, stream=stream.ptr)
    output = model(frame, mv=mv, residual=residual, reset=False, stream=stream.ptr)
    stream.synchronize()
```

The example uses zero motion and residual values; supply the decoded picture's
values for inference. Core waits for each NDArray's recorded writer on the
inference stream and records its read. For an NDArray borrowing external memory,
keep the owner's memory alive until reads on the inference stream have completed.
The example keeps both CuPy owners until synchronization completes.
Core records reads of pixtreme-owned Frame pixels; borrowed Frame pixels retain
the caller's storage-lifetime requirements. On another stream, wait for output
writes using Frame `wait`, and record the last use with `record_stream`, or acquire
pixels through DLPack for its stream handoff. Use `cupy.from_dlpack(output)` to
access output pixels; there is no `Frame.data` property.

Graph outputs are allocated on the inference stream. Keep an external stream
alive until all returned Frames, shallow clones and DLPack views are released,
even after closing the model. Pixtreme retains its own `px.core.Stream` for those
allocations. Synchronize pending work before releasing backend resources.

Every accepted request returns one output, including first and reset requests.
The model owns its persistent history and exchanges the input/output roles of its
two state buffers after each request. It does not retain caller buffers across
calls or synchronize the host between requests.

## Use from C and C++

The wheel also contains a native SDK under `pixtreme/infer/cda_vsr/`: the shared
library `lib/libpixtreme_infer_cda_vsr.so.1`, the C header
`include/pixtreme/infer/cda_vsr.h`, the header-only C++ wrapper
`include/pixtreme/infer/cda_vsr.hpp` and the CMake package `pixtreme_infer_cda_vsr`
in `lib/cmake/pixtreme_infer_cda_vsr/`. The package finds the pixtreme-infer
package installed in the same environment; the shared library locates the
pixtreme, pixtreme-infer and tensoracle libraries of the installed wheels. The
Python `Model` is a binding of the same C++ wrapper.

```cmake
find_package(pixtreme_infer_cda_vsr CONFIG REQUIRED)
add_executable(app main.c)
target_link_libraries(app PRIVATE pixtreme_infer_cda_vsr::pixtreme_infer_cda_vsr)
```

```console
SITE=$(python -c 'import sysconfig; print(sysconfig.get_path("purelib"))')
cmake -S . -B build -Dpixtreme_infer_cda_vsr_DIR="$SITE/pixtreme/infer/cda_vsr/lib/cmake/pixtreme_infer_cda_vsr"
```

`pixtreme_infer_cda_vsr_model_create` takes the same values as the Python
constructor and returns a pixtreme-infer core model handle. Prepare, run, close
and destroy the model with the core C functions in `pixtreme/infer/core.h`.
Creation reads only the checkpoint header and does not use CUDA. Invalid values
and unusable checkpoints return `PIXTREME_INFER_INVALID_ARGUMENT` with the
diagnostic that the Python constructor reports for the same input.
Each call below returns `PIXTREME_INFER_SUCCESS` or another status; on failure,
read `pixtreme_infer_error_message(error)` and release it with
`pixtreme_infer_error_destroy(error)`.

```c
#include <pixtreme/infer/cda_vsr.h>

pixtreme_infer_error_t *error = NULL;
pixtreme_infer_cda_vsr_check_abi(PIXTREME_INFER_CDA_VSR_ABI_MAJOR, PIXTREME_INFER_CDA_VSR_ABI_MINOR, &error);

pixtreme_infer_cda_vsr_model_options_t options = PIXTREME_INFER_CDA_VSR_MODEL_OPTIONS_INIT;
options.checkpoint = "/path/to/checkpoints/best-inference.safetensors";
options.precision = "fp16";
options.input_height = 270;
options.input_width = 480;
options.use_cuda_graph = 0;
pixtreme_infer_model_t *model = NULL;
pixtreme_infer_cda_vsr_model_create(&options, &model, &error);

pixtreme_infer_prepare_options_t prepare = PIXTREME_INFER_PREPARE_OPTIONS_INIT;
prepare.device = 0;
prepare.stream = stream; /* a pixtreme_stream_t; CUDA Graph needs a non-default stream */
pixtreme_infer_model_prepare(model, &prepare, &error);

/* request.frame is the RGB float32 Frame; request.params names the "mv" (2,H,W) and
 * "residual" (1,H,W) float32 NDArrays; request.reset and request.stream as in Python. */
pixtreme_frame_t *output = NULL;
pixtreme_infer_model_run(model, &request, &output, &error);

pixtreme_infer_model_close(model, &error); /* after the stream's work has finished */
pixtreme_infer_model_destroy(model);
```

The C++ wrapper throws `pixtreme::infer::Error` and owns the model:

```cpp
#include <pixtreme/infer/cda_vsr.hpp>

pixtreme::infer::cda_vsr::Model model("/path/to/checkpoints/best-inference.safetensors", "fp16", 270, 480);
model.prepare(&prepare);
pixtreme::frame output = model.run(frame, {{"mv", mv}, {"residual", residual}}, stream);
model.close();
```

The Frame, NDArray, stream and reset requirements of the Python section above
apply unchanged. Keep the module library loaded until every model handle is
destroyed; linking the CMake target does this.

### Create by module name

With pixtreme-infer 0.22.0 or newer, C and C++ programs can create CDA-VSR by
the module name `cda_vsr`, using only the core header and linking
`pixtreme_infer::pixtreme_infer`. Install this module's wheel alongside core.
The core finds `cda_vsr/lib/libpixtreme_infer_cda_vsr_entry.so` relative to its
own library and keeps it loaded for the process lifetime. The entry library
uses the same `libpixtreme_infer_cda_vsr.so.1` as the Python binding.

```c
#include <pixtreme/infer/core.h>

pixtreme_infer_option_t weights = PIXTREME_INFER_OPTION_INIT;
pixtreme_infer_option_t precision = PIXTREME_INFER_OPTION_INIT;
pixtreme_infer_option_t height = PIXTREME_INFER_OPTION_INIT;
pixtreme_infer_option_t width = PIXTREME_INFER_OPTION_INIT;
weights.name = "weights"; weights.value = "/path/to/checkpoints/best-inference.safetensors";
precision.name = "precision"; precision.value = "fp16";
height.name = "input_height"; height.value = "270";
width.name = "input_width"; width.value = "480";
const pixtreme_infer_option_t *options[] = {&weights, &precision, &height, &width};
pixtreme_infer_model_t *model = NULL;
pixtreme_infer_error_t *error = NULL;
pixtreme_infer_status_t status = pixtreme_infer_model_create_by_name(
    "cda_vsr", options, sizeof options / sizeof options[0], &model, &error);
/* Check status, then prepare, run, close and destroy through core as above.
 * On failure model is NULL; inspect and destroy error when it is non-NULL. */
```

```cpp
#include <pixtreme/infer/core.hpp>

auto model = pixtreme::infer::Model::create_by_name("cda_vsr", {
    {"weights", "/path/to/checkpoints/best-inference.safetensors"},
    {"precision", "fp16"}, {"input_height", "270"}, {"input_width", "480"},
    {"use_cuda_graph", "true"}});
model.prepare(&prepare);
auto output = model.run(frame, {{"mv", mv}, {"residual", residual}}, stream);
model.close();
```

All option values are strings. This module accepts these keys:

| Key | Accepted value | Required/default |
| --- | --- | --- |
| `weights` | Nonempty path to a compatible CDA-VSR safetensors checkpoint; relative paths use the process working directory | Required |
| `precision` | `fp32` or `fp16` | Required |
| `input_height` | Positive decimal integer representable by `int64_t` | Required |
| `input_width` | Positive decimal integer representable by `int64_t` | Required |
| `use_cuda_graph` | `true` or `false` | Defaults to `false` |

CDA-VSR rejects `int8` and `fp8`. Unknown, duplicate, missing and invalid options
return `PIXTREME_INFER_INVALID_ARGUMENT`; diagnostics identify the key and do not
include option values. An incompatible or unreadable checkpoint is reported
under `weights`. Options are borrowed only for the creation call. Each successful
call owns an independent model. The existing module-specific C/C++ constructors
and Python `Model(checkpoint=...)` keep their arguments and diagnostics.

Preparation, inference and release can run on a thread whose current device
differs from the model's device. Core selects the model's device during the
operation and restores the caller's device and context before returning.
Input frames, arrays and the non-blocking stream still belong to the model's
device, and callers finish pending stream work before releasing resources.

### C ABI version

This module's C ABI has its own version, `PIXTREME_INFER_CDA_VSR_ABI_MAJOR` and
`PIXTREME_INFER_CDA_VSR_ABI_MINOR` in `cda_vsr.h` (currently 1.0). It is
independent of the distribution version and of pixtreme-infer's core C ABI.
`pixtreme_infer_cda_vsr_abi_version` reports the loaded library's version, and
`pixtreme_infer_cda_vsr_check_abi` returns `PIXTREME_INFER_INCOMPATIBLE_ABI`
unless the header's major equals the library's major and the header's minor is
not newer than the library's minor. Call it with the header constants before
creating models; the C++ wrapper's constructor and
`pixtreme::infer::cda_vsr::require_abi()` perform the same check. The shared
library's SONAME carries the major (`libpixtreme_infer_cda_vsr.so.1`), so a
program linked against one major does not load a library of another major.

## Dependencies

- Runtime dependencies are `pixtreme>=2.14.0`, `pixtreme-infer>=0.22.0`, `safetensors>=0.4`,
  `tensoracle>=2.1.2`, and the TensorRT runtime `tensorrt-cu13>=11,<12` that tensoracle uses.
  The module does not rely on core extras.
- Development and tests use the published `pixtreme-infer` distribution from PyPI.
  The `dev` dependency group installs `tensoracle[convert]` and PyTorch. The tests use
  PyTorch to convert the public checkpoint, to compute reference values, and to run
  TensorRT engines on the GPU. The development environment takes PyTorch from the
  PyTorch `cu132` index (CUDA 13.2); the distributed wheel does not depend on PyTorch.

## Development

The development checkout contains an independent CMake project in `cpp/` and the
Python project in `python/`. Run Python development commands from `python/`:

```console
cd python
uv sync --group dev
uv lock --check
uv run python -m pytest
uv run ruff check . && uv run ruff format --check .
uv run mypy
uv build --wheel
```

`uv sync` installs the pinned Conan CLI from the build requirements. Before CMake
configures the compiler, Conan resolves the locked CUDA runtime/CRT and TensorRT
headers. The repository recipes verify NVIDIA downloads against their SHA-256
manifests. No OS-installed CUDA toolkit or TensorRT SDK is needed. Companion
libraries and nanobind continue to come from Python wheels. Editable installation preserves the namespace initializers supplied
by pixtreme and pixtreme-infer. The build generates `_native.pyi` from the
compiled module; editable installation places it in `src/pixtreme/infer/cda_vsr/`,
where Git ignores it.

For standalone C++ tests, prepare an SDK environment with uv and pass its installed
headers and library to CMake. The `cpp/` configuration and tests do not invoke
Python or require nanobind:

```console
uv venv ../build/native-sdk
uv pip install --python ../build/native-sdk/bin/python 'tensoracle>=2.1.2' 'pixtreme-infer>=0.22.0' 'pixtreme>=2.14.0' 'nanobind>=2.4'
CDA_SITE=$(../build/native-sdk/bin/python -c 'import sysconfig; print(sysconfig.get_path("purelib"))')
CDA_INFER_CMAKE=$(../build/native-sdk/bin/python -c 'import pixtreme.infer; print(pixtreme.infer.get_cmake_dir())')
uv tool install "$(.venv/bin/python -c 'import tomllib; print(next(r for r in tomllib.load(open("pyproject.toml", "rb"))["build-system"]["requires"] if r.startswith("conan==")))')"
cmake -P ../cpp/conan/prepare-tests.cmake
cmake -S ../cpp -B ../build/cpp -G Ninja -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$CDA_SITE/tensoracle/cpp" -Dpixtreme_infer_DIR="$CDA_INFER_CMAKE"
cmake --build ../build/cpp -j2
ctest --test-dir ../build/cpp --output-on-failure -LE gpu
```

The preparation command installs the locked CUDA, TensorRT headers and Catch2
packages. C++ test configuration and Python native helper configuration use only
prepared packages (`--no-remote --build=never`) and never export recipes. Missing
dependencies fail configuration with the preparation command. GPU CTest cases
have the `gpu` label. The lifecycle case and the C entry's GPU cases
take the documented ×2 `params_ema` safetensors checkpoint through
`PIXTREME_CDA_VSR_CHECKPOINT`; the C entry's GPU cases add the residual-unit
metadata to a temporary copy and keep their engines under the build directory.

Python tests use native helpers for import compatibility, the C++ checkpoint
inspection, direct access to the module's TensorRT engines, and small GPU test
networks. Build the helpers before running pytest. Test execution does not
download code; the only code it builds is a small C and C++ consumer of the
installed CMake package, which needs CMake and C and C++ compilers and is skipped
when they are absent:

```console
cmake -S tests/native -B ../build/python-tests -G Ninja -DCMAKE_BUILD_TYPE=Release -DPython_EXECUTABLE="$PWD/../build/native-sdk/bin/python" -DCMAKE_PREFIX_PATH="$CDA_SITE/tensoracle/cpp;$CDA_SITE/nanobind/cmake" -Dpixtreme_infer_DIR="$CDA_INFER_CMAKE"
cmake --build ../build/python-tests -j2
uv run pytest tests/test_import.py
uv run pytest -m "gpu and not performance"
```

The import tests use a separate extension that registers overlapping C++ types
and check both import orders without a GPU. The import tests and the checkpoint
inspection tests skip with a build instruction when the helpers are absent. The
helpers are excluded from the wheel.

The default Conan profile is `cpp/conan/profiles/linux-gcc13`. Set the CMake
cache variable `CDA_VSR_CONAN_PROFILE` to select a matching compiler profile, and
`CDA_VSR_CONAN_CACHE` to an absolute path to select `core.cache:storage_path`.
Use the same profile, package storage and `CONAN_HOME` for preparation and tests.
For editable builds, pass these as `-Ccmake.define.CDA_VSR_CONAN_PROFILE=...` and
`-Ccmake.define.CDA_VSR_CONAN_CACHE=...` to uv. Rebuild native edits with
`uv sync --reinstall-package pixtreme-infer-cda-vsr -Cbuild.tool-args=-j8`.

The sdist includes native sources, recipes, profiles, lock and license notices;
`uv build --sdist` followed by a wheel build from the extracted archive needs no
Git checkout. Build release wheels in a manylinux
environment, inspect them with `auditwheel show` and `readelf -d`, and retain
pixtreme, pixtreme-infer and tensoracle as external wheel dependencies when applying the platform tag:

```console
auditwheel repair --plat manylinux_2_28_x86_64 --only-plat --exclude libtensoracle.so.2 --exclude libpixtreme_infer.so.3 --exclude libpixtreme.so.2 --wheel-dir checked dist/*.whl
```

Use that tag only after the ELF dependencies satisfy its ABI requirements.
The extension uses `$ORIGIN/lib` for this module's C entry library,
`$ORIGIN/../lib` for pixtreme-infer,
`$ORIGIN/../../core/lib` for pixtreme and
`$ORIGIN/../../../tensoracle/cpp/lib` for tensoracle. The C entry library in
`lib/` uses `$ORIGIN/../../lib`, `$ORIGIN/../../../core/lib` and
`$ORIGIN/../../../../tensoracle/cpp/lib` for the same libraries. Libraries supplied by those
wheels, TensorRT and the NVIDIA driver are not copied into this wheel. Its module
library uses the versioned file `libpixtreme_infer_cda_vsr.so.1`; the adjacent
`libpixtreme_infer_cda_vsr_entry.so` exports only `pixtreme_infer_module_entry`,
has RUNPATH `$ORIGIN`, and lists the module library in `NEEDED`. Inspect that entry
with `readelf -d` and `readelf --dyn-syms`, including in a wheel built from the sdist.

For the GCC 14 manylinux image, set `CDA_VSR_CONAN_PROFILE` to the absolute path
of `cpp/conan/profiles/linux-gcc14`, `CDA_VSR_CONAN_ARGS=--build=*`, and a writable
`CONAN_HOME` and `CDA_VSR_CONAN_CACHE` in the container environment. Conan is
installed by the isolated Python build; the image needs no preinstalled Conan.
When `CONAN_HOME` is unset and HOME is unwritable, configuration uses a `home/`
directory beneath the build's Conan output directory.

## Versioning and release

The version is derived by setuptools-scm through scikit-build-core from tags of the form `vX.Y.Z`
at the repository root; tags with other prefixes do not affect it.

Each version is distributed on PyPI as `pixtreme-infer-cda-vsr`. The same wheel and
its `SHA256SUMS` file are also available in GitHub Releases. Declare dependencies
on the PyPI distribution with a minimum version (`pixtreme-infer-cda-vsr>=X.Y.Z`).
Use PyPI as the dependency source; GitHub Release assets and Git references are
not dependency sources.

## License

The module is licensed under the [MIT License](LICENSE). The native wheel also
contains nanobind (BSD-3-Clause), robin-map (MIT), compiled code from TensorRT
headers (Apache-2.0), and the CUDA runtime. Their complete license notices are included in
`dist-info/licenses/`; the wheel declares `MIT AND BSD-3-Clause AND Apache-2.0`.
