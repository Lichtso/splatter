[![Docs](https://docs.rs/splatter/badge.svg)](https://docs.rs/splatter/)
[![crates.io](https://img.shields.io/crates/v/splatter.svg)](https://crates.io/crates/splatter)

# Splatter
Inspired by [gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting) but using a somewhat different approach to rendering.

## Features
- Correctly computes the perspective projection of ellipsoids by intersecting the bounding elliptic cone with the view plane
- Uses the rasterizer instead of a tiled compute shader
- Rasterizes rotated rectangles instead of axis aligned squares
- GPU depth sorting using onesweep radix sort (except that the block sort is not WLMS because WebGPU does not support subgroup operations yet)
- CPU depth sorting as a fallback
- Frustum culling (optionally using stream compaction via indirect drawing)
- File parser and progressive loading via segmentation in chunks
- Lots of rendering configuration parameters to customize

## Dependencies

### Dependencies of the Library
- Graphics API: [wgpu](https://wgpu.rs/)
- Geometric Algebra: [geometric_algebra](https://github.com/Lichtso/geometric_algebra)

### Dependencies of the Example
- Window API: [winit](https://github.com/rust-windowing/winit)
- Logging: [log](https://github.com/rust-lang/log)

## Example
You can download some pre-trained models from the original paper [here](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/pretrained/models.zip).

```bash
cargo run --example showcase -- models/garden/point_cloud/iteration_7000/point_cloud.ply
```

### Controls
- A / D: Move left / right
- W / S: Move forward / backward
- Q / E: Move up / down
- Z / X: Roll left / right
- Mouse: Pitch and yaw
