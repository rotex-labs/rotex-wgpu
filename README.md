<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/rotex-labs/rotex-assets/55fa2474cb788a445c33b4ccf615eb8bbb9dbe64/brand/docs/crate-headers/dark/rotex-wgpu-header.svg">
    <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/rotex-labs/rotex-assets/55fa2474cb788a445c33b4ccf615eb8bbb9dbe64/brand/docs/crate-headers/light/rotex-wgpu-header.svg">
    <img alt="rotex-wgpu" src="https://raw.githubusercontent.com/rotex-labs/rotex-assets/55fa2474cb788a445c33b4ccf615eb8bbb9dbe64/brand/docs/crate-headers/light/rotex-wgpu-header.svg" width="1200">
  </picture>
</p>

# rotex-wgpu

Stateless wgpu translator, including headless use.

> [!WARNING]
> **Beta, pre-1.0.** Rotex is under active development. APIs, behavior, and crate
> boundaries can change in any 0.x release. Do not depend on it for production work yet.

## Overview

`rotex-wgpu` implements the same `GpuBackend` command surface as the other backends, through wgpu. It translates commands. It does not own scene state.

## Responsibilities

- `WgpuGpuPlatform` and `WgpuBridge`
- Handle maps and pipeline compilation
- Indexed indirect draw when the device can record it

## Dependencies

`rotex-core` and `rotex-types`. Tests use `rotex-conformance`.

## Status

Version 0.2.0. Barriers are emulated: the command is accepted and wgpu inserts its own hazards. Push constants are native only when the device supports immediates.

## License

Licensed under either of

* Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
* MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.
