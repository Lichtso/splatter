use crate::{
    renderer::{DepthSorting, Renderer, Uniforms},
    utils::{transmute_slice, transmute_slice_mut},
};
use geometric_algebra::{ppga3d::Rotor, Signum};
use std::{
    convert::TryInto,
    fs::File,
    io::{BufRead, BufReader, Read, Seek, SeekFrom},
};

// Because of numerical precision issues in the shader we have to limit the excentricity of ellipsoids
const MAX_SIZE_VARIANCE: f32 = 5.0;

#[derive(Clone)]
#[repr(C)]
struct SerializedSplat {
    center: [f32; 3],
    n: [f32; 3],
    color: [f32; 3 * 16],
    alpha: f32,
    scale: [f32; 3],
    rotation: [f32; 4],
}

impl Default for SerializedSplat {
    fn default() -> Self {
        unsafe { std::mem::MaybeUninit::zeroed().assume_init() }
    }
}

pub(crate) type Splat = [f32; 60];

/// A point cloud of splats
pub struct Scene {
    pub(crate) compute_bind_groups: [wgpu::BindGroup; 4],
    pub(crate) render_bind_group: wgpu::BindGroup,
    pub(crate) splat_buffer: wgpu::Buffer,
    pub(crate) splat_positions: Vec<f32>,
    pub splat_count: usize,
}

impl Scene {
    /// Constructs a new [Scene] and allocates memory for it
    pub fn new(device: &wgpu::Device, renderer: &Renderer, mut splat_count: usize) -> Self {
        splat_count = splat_count.min(renderer.config.max_splat_count);
        let splat_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (splat_count * std::mem::size_of::<Splat>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let uniform_bind_group_entry = wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer: &renderer.uniform_buffer,
                offset: 0,
                size: std::num::NonZeroU64::new(std::mem::size_of::<Uniforms>() as u64),
            }),
        };
        let sorting_bind_group_entry = wgpu::BindGroupEntry {
            binding: 2,
            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer: &renderer.sorting_buffer,
                offset: 0,
                size: std::num::NonZeroU64::new(renderer.sorting_buffer_size as u64),
            }),
        };
        let splats_bind_group_entry = wgpu::BindGroupEntry {
            binding: 6,
            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer: &splat_buffer,
                offset: 0,
                size: std::num::NonZeroU64::new((splat_count * std::mem::size_of::<Splat>()) as u64),
            }),
        };
        let compute_bind_groups: [wgpu::BindGroup; 4] = (0..4)
            .map(|pass_index| {
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout: &renderer.compute_bind_group_layout,
                    entries: &[
                        uniform_bind_group_entry.clone(),
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                buffer: &renderer.sorting_pass_buffers[pass_index],
                                offset: 0,
                                size: std::num::NonZeroU64::new(std::mem::size_of::<u32>() as u64),
                            }),
                        },
                        sorting_bind_group_entry.clone(),
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                buffer: if pass_index & 1 == 0 {
                                    &renderer.entry_buffer_a
                                } else {
                                    &renderer.entry_buffer_b
                                },
                                offset: 0,
                                size: std::num::NonZeroU64::new((splat_count * std::mem::size_of::<(u32, u32)>()) as u64),
                            }),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                buffer: if pass_index & 1 == 0 {
                                    &renderer.entry_buffer_b
                                } else {
                                    &renderer.entry_buffer_a
                                },
                                offset: 0,
                                size: std::num::NonZeroU64::new((splat_count * std::mem::size_of::<(u32, u32)>()) as u64),
                            }),
                        },
                        splats_bind_group_entry.clone(),
                    ],
                })
            })
            .collect::<Vec<wgpu::BindGroup>>()
            .try_into()
            .unwrap();
        let render_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &renderer.render_bind_group_layout,
            entries: &[
                uniform_bind_group_entry,
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: if matches!(renderer.config.depth_sorting, DepthSorting::Cpu) || renderer.radix_digit_places & 1 == 0 {
                            &renderer.entry_buffer_a
                        } else {
                            &renderer.entry_buffer_b
                        },
                        offset: 0,
                        size: std::num::NonZeroU64::new((splat_count * std::mem::size_of::<(u32, u32)>()) as u64),
                    }),
                },
                splats_bind_group_entry,
            ],
        });
        Self {
            compute_bind_groups,
            render_bind_group,
            splat_buffer,
            splat_positions: if matches!(renderer.config.depth_sorting, DepthSorting::Cpu) {
                vec![0.0; splat_count * 3]
            } else {
                Vec::new()
            },
            splat_count,
        }
    }

    /// Loads the development test scene
    pub fn new_dev_test(device: &wgpu::Device, renderer: &Renderer, queue: &wgpu::Queue) -> Self {
        let mut scene = Self::new(device, renderer, 3);
        let splat_index_range = 0..3;
        let mut splat_data = vec![[0.0; 60]; splat_index_range.len()];
        for index in 0..splat_data.len() {
            splat_data[index][0] = 1.0;
            splat_data[index][1] = 0.0;
            splat_data[index][2] = 0.0;
            splat_data[index][3] = 0.0;
            splat_data[index][4] = if index == 0 { 1.0 } else { 0.0 };
            splat_data[index][5] = if index == 1 { 1.0 } else { 0.0 };
            splat_data[index][6] = if index == 2 { 1.0 } else { 0.0 };
            splat_data[index][8] = 0.2;
            splat_data[index][9] = 0.1;
            splat_data[index][10] = 0.05;
            splat_data[index][11] = 1.0;
            splat_data[index][12] = if index == 0 { 1.0 } else { 0.0 };
            splat_data[index][13] = if index == 1 { 1.0 } else { 0.0 };
            splat_data[index][14] = if index == 2 { 1.0 } else { 0.0 };
        }
        queue.write_buffer(
            &scene.splat_buffer,
            (splat_index_range.start * std::mem::size_of::<Splat>()) as u64,
            transmute_slice(&splat_data),
        );
        if !scene.splat_positions.is_empty() {
            for index in 0..splat_data.len() {
                let splat = &splat_data[index];
                scene.splat_positions[index * 3 + 0] = splat[4];
                scene.splat_positions[index * 3 + 1] = splat[5];
                scene.splat_positions[index * 3 + 2] = splat[6];
            }
        }
        scene
    }

    /// Parses the header of a splat file
    ///
    /// Returns the header length in bytes, the number of splats in the file and the file handle
    pub fn parse_file_header(file: File) -> (u16, usize, File) {
        let mut splat_count: usize = 0;
        let mut reader = BufReader::new(file);
        let mut line = String::new();
        loop {
            reader.read_line(&mut line).unwrap();
            if line == "end_header\n" {
                break;
            }
            if line.starts_with("element vertex ") {
                splat_count = line[15..line.len() - 1].parse().unwrap();
            }
            line.clear();
        }
        let file_header_size = reader.stream_position().unwrap() as u16;
        (file_header_size, splat_count, reader.into_inner())
    }

    /// Loads a chunk of the splat file or the entire file
    pub fn load_chunk(&mut self, queue: &wgpu::Queue, file: &mut File, file_header_size: u16, mut splat_index_range: std::ops::Range<usize>) {
        splat_index_range.end = splat_index_range.end.min(self.splat_count);
        let mut splat_data = vec![[0.0; 60]; splat_index_range.len()];
        let mut serialized_splats = vec![SerializedSplat::default(); splat_index_range.len()];
        file.seek(SeekFrom::Start(
            file_header_size as u64 + (splat_index_range.start * std::mem::size_of::<SerializedSplat>()) as u64,
        ))
        .unwrap();
        file.read_exact(transmute_slice_mut::<_, u8>(&mut serialized_splats)).unwrap();
        for index in 0..splat_data.len() {
            let serialized_splat = &serialized_splats[index];
            let rotor = <[f32; 4]>::from(Rotor::from(serialized_splat.rotation).signum());
            splat_data[index][0] = rotor[0];
            splat_data[index][1] = rotor[1];
            splat_data[index][2] = rotor[2];
            splat_data[index][3] = rotor[3];
            splat_data[index][4] = serialized_splat.center[0];
            splat_data[index][5] = serialized_splat.center[1];
            splat_data[index][6] = serialized_splat.center[2];
            let average = (serialized_splat.scale[0] + serialized_splat.scale[1] + serialized_splat.scale[2]) / 3.0;
            splat_data[index][8] = serialized_splat.scale[0]
                .max(average - MAX_SIZE_VARIANCE)
                .min(average + MAX_SIZE_VARIANCE)
                .exp();
            splat_data[index][9] = serialized_splat.scale[1]
                .max(average - MAX_SIZE_VARIANCE)
                .min(average + MAX_SIZE_VARIANCE)
                .exp();
            splat_data[index][10] = serialized_splat.scale[2]
                .max(average - MAX_SIZE_VARIANCE)
                .min(average + MAX_SIZE_VARIANCE)
                .exp();
            splat_data[index][11] = 1.0 / (1.0 + (-serialized_splat.alpha).exp());
            splat_data[index][12..].copy_from_slice(&serialized_splat.color[0..3 * 16]);
        }
        queue.write_buffer(
            &self.splat_buffer,
            (splat_index_range.start * std::mem::size_of::<Splat>()) as u64,
            transmute_slice(&splat_data),
        );
        if !self.splat_positions.is_empty() {
            for index in 0..splat_data.len() {
                let splat = &splat_data[index];
                self.splat_positions[index * 3 + 0] = splat[4];
                self.splat_positions[index * 3 + 1] = splat[5];
                self.splat_positions[index * 3 + 2] = splat[6];
            }
        }
    }
}
