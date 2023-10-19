struct Uniforms {
    camera_matrix: mat4x4<f32>,
    view_matrix: mat4x4<f32>,
    view_projection_matrix: mat4x4<f32>,
    view_size: vec2<f32>,
    image_size: vec2<u32>,
    frustum_culling_tolerance: f32,
    ellipse_size_bias: f32,
    ellipse_margin: f32,
    splat_scale: f32,
}
struct DrawIndirect {
    vertex_count: u32,
    instance_count: atomic<u32>,
    base_vertex: u32,
    base_instance: u32,
}
struct SortingGlobal {
    status_counters: array<array<atomic<u32>, RADIX_BASE>, MAX_TILE_COUNT_C>,
    digit_histogram: array<array<atomic<u32>, RADIX_BASE>, RADIX_DIGIT_PLACES>,
    draw_indirect: DrawIndirect,
    assignment_counter: atomic<u32>,
}
struct Entry {
    key: u32,
    value: u32,
}
struct Splat {
    rotation: vec4<f32>,
    center: vec3<f32>,
    paddingA: f32,
    scale: vec3<f32>,
    alpha: f32,
    colorSH: array<f32, 48>,
}
@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<uniform> sorting_pass_index: u32;
@group(0) @binding(2) var<storage, read_write> sorting: SortingGlobal;
@group(0) @binding(3) var<storage, read_write> input_entries: array<Entry>;
@group(0) @binding(4) var<storage, read_write> output_entries: array<Entry>;
@group(0) @binding(5) var<storage, read> sorted_entries: array<Entry>;
@group(0) @binding(6) var<storage> splats: array<Splat>;

fn screenToClipSpace(screen_space_pos: vec2<f32>) -> vec2<f32> {
    var result = ((screen_space_pos.xy / vec2<f32>(uniforms.image_size)) - vec2<f32>(0.5));
    return vec2<f32>(2.0 * result.x, -2.0 * result.y);
}

fn clipToScreenSpace(clip_space_pos: vec2<f32>) -> vec2<f32> {
    var result = vec2<f32>(0.5 * clip_space_pos.x, -0.5 * clip_space_pos.y) + vec2<f32>(0.5);
    return result * vec2<f32>(uniforms.image_size);
}

fn worldToClipSpace(world_pos: vec3<f32>) -> vec4<f32> {
    var homogenous_pos = uniforms.view_projection_matrix * vec4<f32>(world_pos, 1.0);
    return vec4<f32>(homogenous_pos.xyz, 1.0) / (homogenous_pos.w + 0.0000001);
}

fn isInFrustum(clip_space_pos: vec3<f32>) -> bool {
    return abs(clip_space_pos.x) < uniforms.frustum_culling_tolerance && abs(clip_space_pos.y) < uniforms.frustum_culling_tolerance && abs(clip_space_pos.z - 0.5) < 0.5;
}

fn quatToMat(p: vec4<f32>) -> mat3x3<f32> {
  var q = p * sqrt(2.0);
  var yy = q.y * q.y;
  var yz = q.y * q.z;
  var yw = q.y * q.w;
  var yx = q.y * q.x;
  var zz = q.z * q.z;
  var zw = q.z * q.w;
  var zx = q.z * q.x;
  var ww = q.w * q.w;
  var wx = q.w * q.x;
  return mat3x3<f32>(
    1.0 - zz - ww, yz + wx, yw - zx,
    yz - wx, 1.0 - yy - ww, zw + yx,
    yw + zx, zw - yx, 1.0 - yy - zz,
  );
}

/*
    When it comes to finding the projected contour of an ellipsoid the original and all other ports of it
    use the following approach, which looks and feels about right, but is wrong.
    The issue is that this method of taking the 3D covariance and projecting it to 2D only works for
    parallel / orthographic projections, not perspective projections.

    The reason is that perspective projections have three additional effects:
      - Parallax movements (that is the view plane moves parallel to the ellipsoids) change the shape of the projected ellipse.
        E.g. a sphere only appears circular when in center of the view, once it moves to the edges it becomes stretched into an ellipse.
      - Rotating an ellipse can change the position it appears at, or in other words creates additional translation.
        This effect is zero if the ellipse has one of its three axes pointing straight at the view (parallel to the normal of the view plane).
        But, if it is rotated 45°, then the tip of the ellipse that is closer to the view plane becomes larger
        through the perspective while the other end becomes smaller.
        Put together, this slightly shifts the center of the appearance away from the projected center of the ellipsoid.
      - Conic sections can not only result in ellipses but also parabola and hyperbola.
        This however is an edge case that only happens when the ellipsoid intersects with the view plane and
        can probably be ignored as one would clip away such ellipsoids anyway.
*/
fn projectedCovarianceOfEllipsoid(scale: vec3<f32>, rotation: vec4<f32>, translation: vec3<f32>) -> mat3x3<f32> {
    let camera_matrix = mat3x3<f32>(uniforms.camera_matrix.x.xyz, uniforms.camera_matrix.y.xyz, uniforms.camera_matrix.z.xyz);
    var transform = quatToMat(rotation);
    transform.x *= scale.x;
    transform.y *= scale.y;
    transform.z *= scale.z;

    // 3D Covariance
    var view_pos = uniforms.view_matrix * vec4<f32>(translation, 1.0);
    view_pos.x = clamp(view_pos.x / view_pos.z, -1.0, 1.0) * view_pos.z;
    view_pos.y = clamp(view_pos.y / view_pos.z, -1.0, 1.0) * view_pos.z;
    let T = transpose(transform) * camera_matrix * mat3x3(
        1.0 / view_pos.z, 0.0, -view_pos.x / (view_pos.z * view_pos.z),
        0.0, 1.0 / view_pos.z, -view_pos.y / (view_pos.z * view_pos.z),
        0.0, 0.0, 0.0,
    );
    let covariance_matrix = transpose(T) * T;

    return covariance_matrix;
}

/*
    The correct approach is to instead construct a bounding cone with its vertex at the camera position, which bounds the ellipsoid in 3D.
    Then find the intersection between that bounding cone and the view plane. The resulting conic section is the correct contour in 2D,
    formulated as an algebraic / implicit curve: 0 = M.x.x * x^2 + M.y.y * y^2 + M.x.y * 2.0 * x * y + M.x.z * 2.0 * x + M.y.z * 2.0 * y + M.z.z
*/
fn projectedContourOfEllipsoid(scale: vec3<f32>, rotation: vec4<f32>, translation: vec3<f32>) -> mat3x3<f32> {
    let camera_matrix = mat3x3<f32>(uniforms.camera_matrix.x.xyz, uniforms.camera_matrix.y.xyz, uniforms.camera_matrix.z.xyz);
    var transform = quatToMat(rotation);
    transform.x /= scale.x;
    transform.y /= scale.y;
    transform.z /= scale.z;
    let ray_origin = uniforms.camera_matrix.w.xyz - translation;
    let local_ray_origin = ray_origin * transform;
    let local_ray_origin_squared = local_ray_origin * local_ray_origin;

    // Calculate the bounding cone of the ellipsoid with its vertex at the camera position
    let diagonal = 1.0 - local_ray_origin_squared.yxx - local_ray_origin_squared.zzy;
    let triangle = local_ray_origin.yxx * local_ray_origin.zzy;
    let A = mat3x3<f32>(
        diagonal.x, triangle.z, triangle.y,
        triangle.z, diagonal.y, triangle.x,
        triangle.y, triangle.x, diagonal.z,
    );

    /*
    let c = local_ray_origin_squared.x + local_ray_origin_squared.y + local_ray_origin_squared.z;
    let d = sqrt(c + 1.0);
    let e = 1.0 / c;
    let diagonal = (local_ray_origin_squared + (local_ray_origin_squared.yxx + local_ray_origin_squared.zzy) * d) * e;
    let triangle = local_ray_origin.yxx * local_ray_origin.zzy * (1.0 - d) * e;
    let A = transform * mat3x3<f32>(
        diagonal.x, triangle.z, triangle.y,
        triangle.z, diagonal.y, triangle.x,
        triangle.y, triangle.x, diagonal.z,
    );
    let sqrt_M = transpose(A) * camera_matrix;
    */

    // Given: let pos_in_view_plane = vec3<f32>(screenToClipSpace(stage_in.gl_Position.xy) * uniforms.view_size, 1.0);
    // And: let local_ray_direction = camera_matrix * pos_in_view_plane * transform;
    // The matrix A would be sufficient to render the ellipse: dot(local_ray_direction, A * local_ray_direction) = 0
    // However, we want to be independent of the ray direction, as we do not need to do this work per fragment.

    // Calculate the projected contour from the intersection between the view plane and the bounding cone of the ellipsoid
    transform = transpose(camera_matrix) * transform;
    let M = transform * A * transpose(transform);

    // Again, M is sufficient to render the ellipse: dot(pos_in_view_plane, M * pos_in_view_plane) = 0
    return M;
}

/*
    Decompose the implicit curve of the ellipse into its three components: scale, rotation, translation.
    This is not necessary for rendering but it allows optimizing rasterization by using rotated rectangles instead of axis aligned squares.

    Formulas: https://math.stackexchange.com/questions/616645/determining-the-major-minor-axes-of-an-ellipse-from-general-form
    Formulas: https://mathworld.wolfram.com/Ellipse.html
*/

fn extractTranslationOfEllipse(M: mat3x3<f32>) -> vec2<f32> {
    /*
        The center of the ellipse is at the extremum (minimum / maximum) of the implicit curve.
        So, take the partial derivative in x and y, which is: (2.0 * M.x.x * x + M.x.y * y + M.x.z, M.x.y * x + 2.0 * M.y.y * y + M.y.z)
        And the roots of that partial derivative are the position of the extremum, thus the translation of the ellipse.
    */
    let discriminant = M.x.x * M.y.y - M.x.y * M.x.y;
    let inverse_discriminant = 1.0 / discriminant;
    return vec2<f32>(
        M.x.y * M.y.z - M.y.y * M.x.z,
        M.x.y * M.x.z - M.x.x * M.y.z,
    ) * inverse_discriminant;
}

fn extractRotationOfEllipse(M: mat3x3<f32>) -> vec2<f32> {
    /*
        phi = atan(2.0 * M.x.y / (M.x.x - M.y.y)) / 2
        k = cos(phi)
        j = sin(phi)
        Insert angle phi into cos() and then apply the half-angle identity to get:
    */
    let a = (M.x.x - M.y.y) * (M.x.x - M.y.y);
    let b = a + 4.0 * M.x.y * M.x.y;
    let c = 0.5 * sqrt(a / b);
    var j = sqrt(0.5 - c);
    var k = -sqrt(0.5 + c) * sign(M.x.y) * sign(M.x.x - M.y.y);
    if(M.x.y < 0.0 || M.x.x - M.y.y < 0.0) {
        k = -k;
        j = -j;
    }
    if(M.x.x - M.y.y < 0.0) {
        let t = j;
        j = -k;
        k = t;
    }
    return vec2<f32>(j, k);
}

fn extractScaleOfEllipse(M: mat3x3<f32>, translation: vec2<f32>, rotation: vec2<f32>) -> vec2<f32> {
    /*b = sqrt(b);
    let q = (M.z.z + (2.0 * M.x.y * M.x.z * M.y.z - (M.x.x * M.y.z * M.y.z + M.y.y * M.x.z * M.x.z)) * inverse_discriminant) * inverse_discriminant * 0.25;
    let focal_point_squared = 4.0 * abs(q) * b;
    let semi_major_axis_squared = (b - sign(q) * (M.x.x + M.y.y)) * q * 2.0;
    let semi_major_axis = sqrt(semi_major_axis_squared);
    let semi_minor_axis = sqrt(semi_major_axis_squared - focal_point_squared);*/

    /*b = sqrt(b);
    let numerator = -2.0 * (M.x.x * M.y.z * M.y.z + M.y.y * M.x.z * M.x.z + M.z.z * M.x.y * M.x.y - 2.0 * M.x.y * M.x.z * M.y.z - M.x.x * M.y.y * M.z.z);
    let semi_major_axis = sqrt(numerator / (discriminant * (-b - (M.x.x + M.y.y))));
    let semi_minor_axis = sqrt(numerator / (discriminant * (b - (M.x.x + M.y.y))));*/

    let d = 2.0 * M.x.y * rotation.x * rotation.y;
    let e = M.z.z - (M.x.x * translation.x * translation.x + M.y.y * translation.y * translation.y + 2.0 * M.x.y * translation.x * translation.y);
    // let e = dot(sqrt_M.z, sqrt_M.z) - dot(sqrt_M.x * translation.x + sqrt_M.y * translation.y, sqrt_M.x * translation.x + sqrt_M.y * translation.y);
    let semi_major_axis = sqrt(abs(e / (M.x.x * rotation.y * rotation.y + M.y.y * rotation.x * rotation.x - d)));
    let semi_minor_axis = sqrt(abs(e / (M.x.x * rotation.x * rotation.x + M.y.y * rotation.y * rotation.y + d)));

    return vec2<f32>(semi_major_axis, semi_minor_axis);
}

// Same as extractScaleOfEllipse() but expects the input to be normalized
fn extractScaleOfCovariance(M: mat3x3<f32>) -> vec2<f32> {
    let a = (M.x.x - M.y.y) * (M.x.x - M.y.y);
    let b = sqrt(a + 4.0 * M.x.y * M.x.y);
    let semi_major_axis = sqrt((M.x.x + M.y.y + b) * 0.5);
    let semi_minor_axis = sqrt((M.x.x + M.y.y - b) * 0.5);
    return vec2<f32>(semi_major_axis, semi_minor_axis);
}

// Spherical harmonics coefficients
const shc = array<f32, 16>(
    0.28209479177387814,
    -0.4886025119029199,
    0.4886025119029199,
    -0.4886025119029199,
    1.0925484305920792,
	-1.0925484305920792,
	0.31539156525252005,
	-1.0925484305920792,
	0.5462742152960396,
    -0.5900435899266435,
	2.890611442640554,
	-0.4570457994644658,
	0.3731763325901154,
	-0.4570457994644658,
	1.445305721320277,
	-0.5900435899266435,
);

fn sphericalHarmonicsLookup(ray_direction: vec3<f32>, splat_index: u32) -> vec3<f32> {
    var ray_direction_squared = ray_direction * ray_direction;
    var color = vec3<f32>(0.5);
    color += shc[ 0] * vec3<f32>(splats[splat_index].colorSH[ 0], splats[splat_index].colorSH[ 1], splats[splat_index].colorSH[ 2]);
    if(SPHERICAL_HARMONICS_ORDER > 0u) {
        color += shc[ 1] * vec3<f32>(splats[splat_index].colorSH[ 3], splats[splat_index].colorSH[ 4], splats[splat_index].colorSH[ 5]) * ray_direction.y;
        color += shc[ 2] * vec3<f32>(splats[splat_index].colorSH[ 6], splats[splat_index].colorSH[ 7], splats[splat_index].colorSH[ 8]) * ray_direction.z;
        color += shc[ 3] * vec3<f32>(splats[splat_index].colorSH[ 9], splats[splat_index].colorSH[10], splats[splat_index].colorSH[11]) * ray_direction.x;
    }
    if(SPHERICAL_HARMONICS_ORDER > 1u) {
        color += shc[ 4] * vec3<f32>(splats[splat_index].colorSH[12], splats[splat_index].colorSH[13], splats[splat_index].colorSH[14]) * ray_direction.x * ray_direction.y;
        color += shc[ 5] * vec3<f32>(splats[splat_index].colorSH[15], splats[splat_index].colorSH[16], splats[splat_index].colorSH[17]) * ray_direction.y * ray_direction.z;
        color += shc[ 6] * vec3<f32>(splats[splat_index].colorSH[18], splats[splat_index].colorSH[19], splats[splat_index].colorSH[20]) * (2.0 * ray_direction_squared.z - ray_direction_squared.x - ray_direction_squared.y);
        color += shc[ 7] * vec3<f32>(splats[splat_index].colorSH[21], splats[splat_index].colorSH[22], splats[splat_index].colorSH[23]) * ray_direction.x * ray_direction.z;
        color += shc[ 8] * vec3<f32>(splats[splat_index].colorSH[24], splats[splat_index].colorSH[25], splats[splat_index].colorSH[26]) * (ray_direction_squared.x - ray_direction_squared.y);
    }
    if(SPHERICAL_HARMONICS_ORDER > 2u) {
        color += shc[ 9] * vec3<f32>(splats[splat_index].colorSH[27], splats[splat_index].colorSH[28], splats[splat_index].colorSH[29]) * ray_direction.y * (3.0 * ray_direction_squared.x - ray_direction_squared.y);
        color += shc[10] * vec3<f32>(splats[splat_index].colorSH[30], splats[splat_index].colorSH[31], splats[splat_index].colorSH[32]) * ray_direction.x * ray_direction.y * ray_direction.z;
        color += shc[11] * vec3<f32>(splats[splat_index].colorSH[33], splats[splat_index].colorSH[34], splats[splat_index].colorSH[35]) * ray_direction.y * (4.0 * ray_direction_squared.z - ray_direction_squared.x - ray_direction_squared.y);
        color += shc[12] * vec3<f32>(splats[splat_index].colorSH[36], splats[splat_index].colorSH[37], splats[splat_index].colorSH[38]) * ray_direction.z * (2.0 * ray_direction_squared.z - 3.0 * ray_direction_squared.x - 3.0 * ray_direction_squared.y);
        color += shc[13] * vec3<f32>(splats[splat_index].colorSH[39], splats[splat_index].colorSH[40], splats[splat_index].colorSH[41]) * ray_direction.x * (4.0 * ray_direction_squared.z - ray_direction_squared.x - ray_direction_squared.y);
        color += shc[14] * vec3<f32>(splats[splat_index].colorSH[42], splats[splat_index].colorSH[43], splats[splat_index].colorSH[44]) * ray_direction.z * (ray_direction_squared.x - ray_direction_squared.y);
        color += shc[15] * vec3<f32>(splats[splat_index].colorSH[45], splats[splat_index].colorSH[46], splats[splat_index].colorSH[47]) * ray_direction.x * (ray_direction_squared.x - 3.0 * ray_direction_squared.y);
    }
    return color;
}

// Onesweep Radix Sort

struct SortingSharedA {
    digit_histogram: array<array<atomic<u32>, RADIX_BASE>, RADIX_DIGIT_PLACES>,
}
var<workgroup> sorting_shared_a: SortingSharedA;

@compute @workgroup_size(RADIX_BASE, RADIX_DIGIT_PLACES)
fn radixSortA(
    @builtin(local_invocation_id) gl_LocalInvocationID: vec3<u32>,
    @builtin(global_invocation_id) gl_GlobalInvocationID: vec3<u32>,
) {
    sorting_shared_a.digit_histogram[gl_LocalInvocationID.y][gl_LocalInvocationID.x] = 0u;
    workgroupBarrier();

    let thread_index = gl_GlobalInvocationID.x * RADIX_DIGIT_PLACES + gl_GlobalInvocationID.y;
    let start_entry_index = thread_index * ENTRIES_PER_INVOCATION_A;
    let end_entry_index = start_entry_index + ENTRIES_PER_INVOCATION_A;
    for(var entry_index = start_entry_index; entry_index < end_entry_index; entry_index += 1u) {
        if(entry_index >= arrayLength(&splats)) {
            continue;
        }
        var key: u32 = 0xFFFFFFFFu; // Stream compaction for frustum culling
        let clip_space_pos = worldToClipSpace(splats[entry_index].center);
        if(isInFrustum(clip_space_pos.xyz)) {
            // key = bitcast<u32>(clip_space_pos.z);
            key = u32(clip_space_pos.z * 0xFFFF.0) << 16u;
            key |= u32((clip_space_pos.x * 0.5 + 0.5) * 0xFF.0) << 8u;
            key |= u32((clip_space_pos.y * 0.5 + 0.5) * 0xFF.0);
        }
        output_entries[entry_index].key = key;
        output_entries[entry_index].value = entry_index;
        for(var shift = 0u; shift < RADIX_DIGIT_PLACES; shift += 1u) {
            let digit = (key >> (shift * RADIX_BITS_PER_DIGIT)) & (RADIX_BASE - 1u);
            atomicAdd(&sorting_shared_a.digit_histogram[shift][digit], 1u);
        }
    }
    workgroupBarrier();

    atomicAdd(&sorting.digit_histogram[gl_LocalInvocationID.y][gl_LocalInvocationID.x], sorting_shared_a.digit_histogram[gl_LocalInvocationID.y][gl_LocalInvocationID.x]);
}

@compute @workgroup_size(1)
fn radixSortB(
    @builtin(global_invocation_id) gl_GlobalInvocationID: vec3<u32>,
) {
    var sum = 0u;
    for(var digit = 0u; digit < RADIX_BASE; digit += 1u) {
        let tmp = sorting.digit_histogram[gl_GlobalInvocationID.y][digit];
        sorting.digit_histogram[gl_GlobalInvocationID.y][digit] = sum;
        sum += tmp;
    }
}

struct SortingSharedC {
    entries: array<atomic<u32>, WORKGROUP_ENTRIES_C>,
    gather_sources: array<atomic<u32>, WORKGROUP_ENTRIES_C>,
    scan: array<atomic<u32>, WORKGROUP_INVOCATIONS_C>,
    total: u32,
}
var<workgroup> sorting_shared_c: SortingSharedC;

const NUM_BANKS: u32 = 16u;
const LOG_NUM_BANKS: u32 = 4u;
fn conflicFreeOffset(n: u32) -> u32 {
    return 0u; // n >> NUM_BANKS + n >> (2u * LOG_NUM_BANKS);
}

fn exclusiveScan(gl_LocalInvocationIndex: u32, value: u32) -> u32 {
    sorting_shared_c.scan[gl_LocalInvocationIndex + conflicFreeOffset(gl_LocalInvocationIndex)] = value;
    var offset = 1u;
    for(var d = WORKGROUP_INVOCATIONS_C >> 1u; d > 0u; d >>= 1u) {
        workgroupBarrier();
        if(gl_LocalInvocationIndex < d) {
            var ai = offset * (2u * gl_LocalInvocationIndex + 1u) - 1u;
            var bi = offset * (2u * gl_LocalInvocationIndex + 2u) - 1u;
            ai += conflicFreeOffset(ai);
            bi += conflicFreeOffset(bi);
            sorting_shared_c.scan[bi] += sorting_shared_c.scan[ai];
        }
        offset <<= 1u;
    }
    if(gl_LocalInvocationIndex == 0u) {
      var i = WORKGROUP_INVOCATIONS_C - 1u;
      i += conflicFreeOffset(i);
      sorting_shared_c.total = sorting_shared_c.scan[i];
      sorting_shared_c.scan[i] = 0u;
    }
    for(var d = 1u; d < WORKGROUP_INVOCATIONS_C; d <<= 1u) {
        workgroupBarrier();
        offset >>= 1u;
        if(gl_LocalInvocationIndex < d) {
            var ai = offset * (2u * gl_LocalInvocationIndex + 1u) - 1u;
            var bi = offset * (2u * gl_LocalInvocationIndex + 2u) - 1u;
            ai += conflicFreeOffset(ai);
            bi += conflicFreeOffset(bi);
            let t = sorting_shared_c.scan[ai];
            sorting_shared_c.scan[ai] = sorting_shared_c.scan[bi];
            sorting_shared_c.scan[bi] += t;
        }
    }
    workgroupBarrier();
    return sorting_shared_c.scan[gl_LocalInvocationIndex + conflicFreeOffset(gl_LocalInvocationIndex)];
}

@compute @workgroup_size(WORKGROUP_INVOCATIONS_C)
fn radixSortC(
    @builtin(local_invocation_id) gl_LocalInvocationID: vec3<u32>,
    @builtin(global_invocation_id) gl_GlobalInvocationID: vec3<u32>,
) {
    // Draw an assignment number
    if(gl_LocalInvocationID.x == 0u) {
        sorting_shared_c.entries[0] = atomicAdd(&sorting.assignment_counter, 1u);
    }
    // Reset histogram
    sorting_shared_c.scan[gl_LocalInvocationID.x + conflicFreeOffset(gl_LocalInvocationID.x)] = 0u;
    workgroupBarrier();

    let assignment = sorting_shared_c.entries[0];
    let global_entry_offset = assignment * WORKGROUP_ENTRIES_C;
    // TODO: Specialize end shader
    if(gl_LocalInvocationID.x == 0u && assignment * WORKGROUP_ENTRIES_C + WORKGROUP_ENTRIES_C >= arrayLength(&splats)) {
        // Last workgroup resets the assignment number for the next pass
        sorting.assignment_counter = 0u;
    }

    // Load keys from global memory into registers and rank them
    var keys: array<u32, ENTRIES_PER_INVOCATION_C>;
    var ranks: array<u32, ENTRIES_PER_INVOCATION_C>;
    for(var entry_index = 0u; entry_index < ENTRIES_PER_INVOCATION_C; entry_index += 1u) {
        keys[entry_index] = input_entries[global_entry_offset + WORKGROUP_INVOCATIONS_C * entry_index + gl_LocalInvocationID.x][0];
        let digit = (keys[entry_index] >> (sorting_pass_index * RADIX_BITS_PER_DIGIT)) & (RADIX_BASE - 1u);
        // TODO: Implement warp-level multi-split (WLMS) once WebGPU supports subgroup operations
        ranks[entry_index] = atomicAdd(&sorting_shared_c.scan[digit + conflicFreeOffset(digit)], 1u);
    }
    workgroupBarrier();

    // Cumulate histogram
    let local_digit_count = sorting_shared_c.scan[gl_LocalInvocationID.x + conflicFreeOffset(gl_LocalInvocationID.x)];
    let local_digit_offset = exclusiveScan(gl_LocalInvocationID.x, local_digit_count);
    sorting_shared_c.scan[gl_LocalInvocationID.x + conflicFreeOffset(gl_LocalInvocationID.x)] = local_digit_offset;

    // Chained decoupling lookback
    atomicStore(&sorting.status_counters[assignment][gl_LocalInvocationID.x], 0x40000000u | local_digit_count);
    var global_digit_count = 0u;
    var previous_tile = assignment;
    while true {
        if(previous_tile == 0u) {
            global_digit_count += sorting.digit_histogram[sorting_pass_index][gl_LocalInvocationID.x];
            break;
        }
        previous_tile -= 1u;
        var status_counter = 0u;
        while((status_counter & 0xC0000000u) == 0u) {
            status_counter = atomicLoad(&sorting.status_counters[previous_tile][gl_LocalInvocationID.x]);
        }
        global_digit_count += status_counter & 0x3FFFFFFFu;
        if((status_counter & 0x80000000u) != 0u) {
            break;
        }
    }
    atomicStore(&sorting.status_counters[assignment][gl_LocalInvocationID.x], 0x80000000u | (global_digit_count + local_digit_count));
    if(sorting_pass_index == RADIX_DIGIT_PLACES - 1u && gl_LocalInvocationID.x == WORKGROUP_INVOCATIONS_C - 2u && global_entry_offset + WORKGROUP_ENTRIES_C >= arrayLength(&splats)) {
        sorting.draw_indirect.vertex_count = 4u;
        sorting.draw_indirect.instance_count = global_digit_count + local_digit_count;
    }

    // Scatter keys inside shared memory
    for(var entry_index = 0u; entry_index < ENTRIES_PER_INVOCATION_C; entry_index += 1u) {
        let key = keys[entry_index];
        let digit = (key >> (sorting_pass_index * RADIX_BITS_PER_DIGIT)) & (RADIX_BASE - 1u);
        ranks[entry_index] += sorting_shared_c.scan[digit + conflicFreeOffset(digit)];
        sorting_shared_c.entries[ranks[entry_index]] = key;
    }
    workgroupBarrier();

    // Add global offset
    sorting_shared_c.scan[gl_LocalInvocationID.x + conflicFreeOffset(gl_LocalInvocationID.x)] = global_digit_count - local_digit_offset;
    workgroupBarrier();

    // Store keys from shared memory into global memory
    for(var entry_index = 0u; entry_index < ENTRIES_PER_INVOCATION_C; entry_index += 1u) {
        let key = sorting_shared_c.entries[WORKGROUP_INVOCATIONS_C * entry_index + gl_LocalInvocationID.x];
        let digit = (key >> (sorting_pass_index * RADIX_BITS_PER_DIGIT)) & (RADIX_BASE - 1u);
        keys[entry_index] = digit;
        output_entries[sorting_shared_c.scan[digit + conflicFreeOffset(digit)] + WORKGROUP_INVOCATIONS_C * entry_index + gl_LocalInvocationID.x][0] = key;
    }
    workgroupBarrier();

    // Load values from global memory and scatter them inside shared memory
    for(var entry_index = 0u; entry_index < ENTRIES_PER_INVOCATION_C; entry_index += 1u) {
        let value = input_entries[global_entry_offset + WORKGROUP_INVOCATIONS_C * entry_index + gl_LocalInvocationID.x][1];
        sorting_shared_c.entries[ranks[entry_index]] = value;
    }
    workgroupBarrier();

    // Store values from shared memory into global memory
    for(var entry_index = 0u; entry_index < ENTRIES_PER_INVOCATION_C; entry_index += 1u) {
        let value = sorting_shared_c.entries[WORKGROUP_INVOCATIONS_C * entry_index + gl_LocalInvocationID.x];
        let digit = keys[entry_index];
        output_entries[sorting_shared_c.scan[digit + conflicFreeOffset(digit)] + WORKGROUP_INVOCATIONS_C * entry_index + gl_LocalInvocationID.x][1] = value;
    }
}

struct VertexOutput {
    @builtin(position) gl_Position: vec4<f32>,
    @location(0) @interpolate(flat) color: vec4<f32>,
    @location(1) @interpolate(linear) gl_TexCoord: vec2<f32>,
    // @location(2) @interpolate(flat) splat_index: u32,
}

@vertex
fn vertex(
    @builtin(instance_index) gl_InstanceID: u32,
    @builtin(vertex_index) gl_VertexID: u32,
) -> VertexOutput {
    var stage_out: VertexOutput;
    var splat_index: u32;
    var discard_quad: bool;
    if(USE_INDIRECT_DRAW) {
        splat_index = sorted_entries[gl_InstanceID][1];
        discard_quad = false;
    } else if(USE_DEPTH_SORTING) {
        splat_index = sorted_entries[gl_InstanceID][1];
        discard_quad = sorted_entries[gl_InstanceID][0] == 0xFFFFFFFFu;
    } else {
        splat_index = gl_InstanceID;
        discard_quad = !isInFrustum(worldToClipSpace(splats[splat_index].center).xyz);
    }
    if(discard_quad) {
        stage_out.gl_Position = vec4<f32>(0.0);
        return stage_out;
    }
    // stage_out.splat_index = splat_index;
    let world_position = splats[splat_index].center;
    let ray_direction = normalize(world_position - uniforms.camera_matrix.w.xyz);
    stage_out.color = vec4<f32>(sphericalHarmonicsLookup(ray_direction, splat_index), splats[splat_index].alpha);
    let M = projectedContourOfEllipsoid(splats[splat_index].scale * uniforms.splat_scale, splats[splat_index].rotation, world_position);
    let translation = extractTranslationOfEllipse(M);
    let rotation = extractRotationOfEllipse(M);
    var semi_axes: vec2<f32>;
    if(USE_COVARIANCE_FOR_SCALE) {
        let covariance = projectedCovarianceOfEllipsoid(splats[splat_index].scale * uniforms.splat_scale, splats[splat_index].rotation, world_position);
        semi_axes = extractScaleOfCovariance(covariance);
    } else {
        semi_axes = extractScaleOfEllipse(M, translation, rotation);
    }
    var transformation = mat3x2<f32>(
        vec2<f32>(rotation.y, -rotation.x) * (uniforms.ellipse_size_bias + semi_axes.x),
        vec2<f32>(rotation.x, rotation.y) * (uniforms.ellipse_size_bias + semi_axes.y),
        translation,
    );
    var quad_vertices = array<vec2<f32>, 4>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(-1.0,  1.0),
        vec2<f32>( 1.0, -1.0),
        vec2<f32>( 1.0,  1.0),
    );
    if(USE_UNALIGNED_RECTANGLES) {
        let T = mat3x3(
            vec3<f32>(transformation.x, 0.0),
            vec3<f32>(transformation.y, 0.0),
            vec3<f32>(transformation.z, 1.0),
        );
        stage_out.gl_TexCoord = quad_vertices[gl_VertexID] * uniforms.ellipse_margin;
        stage_out.gl_Position = vec4<f32>((T * vec3<f32>(stage_out.gl_TexCoord, 1.0)).xy / uniforms.view_size, 0.0, 1.0);
    } else {
        let inverse = mat2x2<f32>(
            transformation.y.y, -transformation.x.y,
            -transformation.y.x, transformation.x.x,
        ) * (1.0 / (transformation.x.x * transformation.y.y - transformation.x.y * transformation.y.x));
        let radius = sqrt(max(dot(transformation.x, transformation.x), dot(transformation.y, transformation.y)));
        stage_out.gl_TexCoord = quad_vertices[gl_VertexID] * radius * uniforms.ellipse_margin;
        stage_out.gl_Position = vec4<f32>((transformation.z + stage_out.gl_TexCoord) / uniforms.view_size, 0.0, 1.0);
        stage_out.gl_TexCoord = inverse * stage_out.gl_TexCoord;
    }
    return stage_out;
}

struct FragmentOutput {
    @location(0) gl_Color: vec4<f32>,
    // @builtin(frag_depth) gl_FragDepth: f32,
}

@fragment
fn fragment(
    stage_in: VertexOutput,
) -> FragmentOutput {
    var stage_out: FragmentOutput;
    let power = dot(stage_in.gl_TexCoord, stage_in.gl_TexCoord);
    let alpha = stage_in.color.a * exp(-0.5 * power);
    if(alpha < 1.0/255.0) {
        discard;
    }
    stage_out.gl_Color = vec4<f32>(stage_in.color.rgb * alpha, alpha);
    return stage_out;
}
