# image-convolution

[![Crates.io](https://img.shields.io/crates/v/image-convolution.svg)](https://crates.io/crates/image-convolution)
[![Docs](https://docs.rs/image-convolution/badge.svg)](https://docs.rs/image-convolution)

Run parallel image convolution on GPU, implemented with wgpu and wgsl.

Run examples:

```
cargo run --example gaussian -- IMAGE_FILE
cargo run --release --example gradient -- IMAGE_FILE
```
