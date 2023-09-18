//! Miscellaneous utility and helper functions
#![allow(dead_code)]

use geometric_algebra::{ppga3d, Transformation, Zero};
use std::convert::TryInto;

/// Transmutes a vector.
pub fn transmute_vec<S, T>(mut vec: Vec<S>) -> Vec<T> {
    let ptr = vec.as_mut_ptr() as *mut T;
    let len = vec.len() * std::mem::size_of::<S>() / std::mem::size_of::<T>();
    let capacity = vec.capacity() * std::mem::size_of::<S>() / std::mem::size_of::<T>();
    std::mem::forget(vec);
    unsafe { Vec::from_raw_parts(ptr, len, capacity) }
}

/// Transmutes a slice.
pub fn transmute_slice<S, T>(slice: &[S]) -> &[T] {
    let ptr = slice.as_ptr() as *const T;
    let len = std::mem::size_of_val(slice) / std::mem::size_of::<T>();
    unsafe { std::slice::from_raw_parts(ptr, len) }
}

/// Transmutes a mutable slice.
pub fn transmute_slice_mut<S, T>(slice: &mut [S]) -> &mut [T] {
    let ptr = slice.as_mut_ptr() as *mut T;
    let len = std::mem::size_of_val(slice) / std::mem::size_of::<T>();
    unsafe { std::slice::from_raw_parts_mut(ptr, len) }
}

/// Converts a [ppga3d::Motor] to a 4x4 matrix for WebGPU.
pub fn motor3d_to_mat4(motor: &ppga3d::Motor) -> [ppga3d::Point; 4] {
    let result = [1, 2, 3, 0]
        .iter()
        .map(|index| {
            let mut point = ppga3d::Point::zero();
            point[*index] = 1.0;
            let row = motor.transformation(point);
            ppga3d::Point::new(row[1], row[2], row[3], row[0])
        })
        .collect::<Vec<_>>();
    result.try_into().unwrap()
}

/// Creates a 4x4 perspective projection matrix for GLSL.
pub fn perspective_projection(width: f32, height: f32, near: f32, far: f32) -> [ppga3d::Point; 4] {
    // let height = 1.0 / (field_of_view_y * 0.5).tan();
    // let width = 1.0 / (field_of_view_x * 0.5).tan(); // = height / aspect_ratio;
    let denominator = 1.0 / (near - far);
    [
        ppga3d::Point::new(1.0 / width, 0.0, 0.0, 0.0),
        ppga3d::Point::new(0.0, 1.0 / height, 0.0, 0.0),
        ppga3d::Point::new(0.0, 0.0, -far * denominator, 1.0),
        ppga3d::Point::new(0.0, 0.0, near * far * denominator, 0.0),
    ]
}

/// Calculates the product of two 4x4 matrices
pub fn mat4_multiplication(a: &[ppga3d::Point; 4], b: &[ppga3d::Point; 4]) -> [ppga3d::Point; 4] {
    [
        a[0] * b[0][0] + a[1] * b[0][1] + a[2] * b[0][2] + a[3] * b[0][3],
        a[0] * b[1][0] + a[1] * b[1][1] + a[2] * b[1][2] + a[3] * b[1][3],
        a[0] * b[2][0] + a[1] * b[2][1] + a[2] * b[2][2] + a[3] * b[2][3],
        a[0] * b[3][0] + a[1] * b[3][1] + a[2] * b[3][2] + a[3] * b[3][3],
    ]
}

/// Calculates the product a 4x4 matrix and a point
pub fn mat4_transform(a: &[ppga3d::Point; 4], b: &ppga3d::Point) -> ppga3d::Point {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]
}
