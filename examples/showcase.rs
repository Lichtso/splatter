use geometric_algebra::{
    ppga3d::{Rotor, Translator},
    GeometricProduct, One, Signum, Transformation,
};
use splatter::{
    renderer::{Configuration, DepthSorting, Renderer},
    scene::Scene,
};
use std::{collections::HashSet, env, fs::File};

mod application_framework;

const LOAD_CHUNK_SIZE: usize = 0; // 1024 * 32;

/// Creates a [Rotor] which represents a rotation by `angle` radians around `axis`.
fn rotate_around_axis(angle: f32, axis: &[f32; 3]) -> Rotor {
    let sinus = (angle * 0.5).sin();
    Rotor::new((angle * 0.5).cos(), axis[0] * sinus, axis[1] * sinus, axis[2] * sinus)
}

struct Application {
    renderer: Renderer,
    scene: Scene,
    file: File,
    file_header_size: u16,
    chunks_left_to_load: usize,
    depth_stencil_texture_view: Option<wgpu::TextureView>,
    viewport_size: wgpu::Extent3d,
    camera_rotation: Rotor,
    camera_translation: Translator,
    pressed_keys: HashSet<winit::event::VirtualKeyCode>,
}

impl application_framework::Application for Application {
    fn new(device: &wgpu::Device, queue: &mut wgpu::Queue, surface_configuration: &wgpu::SurfaceConfiguration) -> Self {
        let file = File::open(env::args().nth(1).unwrap()).unwrap();
        let renderer = Renderer::new(
            device,
            Configuration {
                surface_configuration: surface_configuration.clone(),
                depth_sorting: DepthSorting::Gpu,
                use_covariance_for_scale: true,
                use_unaligned_rectangles: true,
                spherical_harmonics_order: 2,
                max_splat_count: 1024 * 1024 * 4,
                radix_bits_per_digit: 8,
                frustum_culling_tolerance: 1.1,
                ellipse_margin: 2.0,
                splat_scale: 1.0,
            },
        );
        let (file_header_size, splat_count, mut file) = Scene::parse_file_header(file);
        let mut scene = Scene::new(device, &renderer, splat_count);
        let chunks_left_to_load = if LOAD_CHUNK_SIZE == 0 {
            scene.load_chunk(queue, &mut file, file_header_size, 0..splat_count);
            0
        } else {
            (scene.splat_count + LOAD_CHUNK_SIZE - 1) / LOAD_CHUNK_SIZE
        };
        Self {
            renderer,
            scene,
            file,
            file_header_size,
            chunks_left_to_load,
            depth_stencil_texture_view: None,
            viewport_size: wgpu::Extent3d::default(),
            camera_rotation: Rotor::one(),
            camera_translation: Translator::one(),
            pressed_keys: HashSet::new(),
        }
    }

    fn resize(&mut self, device: &wgpu::Device, _queue: &mut wgpu::Queue, surface_configuration: &wgpu::SurfaceConfiguration) {
        self.viewport_size = wgpu::Extent3d {
            width: surface_configuration.width,
            height: surface_configuration.height,
            depth_or_array_layers: 1,
        };
        let depth_stencil_texture_descriptor = wgpu::TextureDescriptor {
            size: self.viewport_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth24PlusStencil8,
            view_formats: &[wgpu::TextureFormat::Depth24PlusStencil8],
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            label: None,
        };
        let depth_stencil_texture = device.create_texture(&depth_stencil_texture_descriptor);
        self.depth_stencil_texture_view = Some(depth_stencil_texture.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2),
            ..wgpu::TextureViewDescriptor::default()
        }));
    }

    fn render(&mut self, device: &wgpu::Device, queue: &mut wgpu::Queue, frame: &wgpu::SurfaceTexture, frame_time: f32) {
        if self.chunks_left_to_load > 0 {
            self.chunks_left_to_load -= 1;
            let load_range = self.chunks_left_to_load * LOAD_CHUNK_SIZE..(self.chunks_left_to_load + 1) * LOAD_CHUNK_SIZE;
            self.scene.load_chunk(queue, &mut self.file, self.file_header_size, load_range);
            queue.submit([]);
        }
        for keycode in &self.pressed_keys {
            let speed = frame_time * 2.0;
            match keycode {
                winit::event::VirtualKeyCode::A => {
                    self.camera_translation += self.camera_rotation.transformation(Translator::new(0.0, speed, 0.0, 0.0));
                }
                winit::event::VirtualKeyCode::D => {
                    self.camera_translation += self.camera_rotation.transformation(Translator::new(0.0, -speed, 0.0, 0.0));
                }
                winit::event::VirtualKeyCode::W => {
                    self.camera_translation += self.camera_rotation.transformation(Translator::new(0.0, 0.0, 0.0, -speed));
                }
                winit::event::VirtualKeyCode::S => {
                    self.camera_translation += self.camera_rotation.transformation(Translator::new(0.0, 0.0, 0.0, speed));
                }
                winit::event::VirtualKeyCode::Q => {
                    self.camera_translation += self.camera_rotation.transformation(Translator::new(0.0, 0.0, -speed, 0.0));
                }
                winit::event::VirtualKeyCode::E => {
                    self.camera_translation += self.camera_rotation.transformation(Translator::new(0.0, 0.0, speed, 0.0));
                }
                winit::event::VirtualKeyCode::Z => {
                    self.camera_rotation = self
                        .camera_rotation
                        .geometric_product(rotate_around_axis(-0.5 * speed, &[0.0, 0.0, 1.0]))
                        .signum();
                }
                winit::event::VirtualKeyCode::X => {
                    self.camera_rotation = self
                        .camera_rotation
                        .geometric_product(rotate_around_axis(0.5 * speed, &[0.0, 0.0, 1.0]))
                        .signum();
                }
                _ => {}
            }
        }
        let camera_motor = self.camera_translation.geometric_product(self.camera_rotation);
        let frame_view = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());
        self.renderer
            .render_frame(device, queue, &frame_view, self.viewport_size, camera_motor, &self.scene);
    }

    fn mouse_motion(&mut self, delta: (f64, f64)) {
        let position = [
            -1.0 * std::f32::consts::PI * (delta.0 as f32 / self.viewport_size.width as f32),
            -1.0 * std::f32::consts::PI * (delta.1 as f32 / self.viewport_size.height as f32),
        ];
        self.camera_rotation = self
            .camera_rotation
            .geometric_product(rotate_around_axis(position[1], &[1.0, 0.0, 0.0]))
            .geometric_product(rotate_around_axis(position[0], &[0.0, 1.0, 0.0]))
            .signum();
    }

    fn keyboard_input(&mut self, input: winit::event::KeyboardInput) {
        let keycode = if let Some(keycode) = input.virtual_keycode {
            keycode
        } else {
            return;
        };
        match input.state {
            winit::event::ElementState::Pressed => {
                self.pressed_keys.insert(keycode);
            }
            winit::event::ElementState::Released => {
                self.pressed_keys.remove(&keycode);
            }
        }
    }
}

fn main() {
    application_framework::ApplicationManager::run::<Application>("Splatter Renderer");
}
