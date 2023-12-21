//! Rust port of the ESDT ("Euclidean Subpixel Distance Transform") algorithm,
//! originally published as the [`@use-gpu/glyph`](https://www.npmjs.com/package/@use-gpu/glyph)
//! `npm` package, and described in <https://acko.net/blog/subpixel-distance-transform/>.

use crate::SdfUnorm8;

#[cfg(replace_f32_with_f64)]
type f32 = f64;

#[derive(Copy, Clone, Debug)]
pub struct Params {
    pub pad: usize,
    pub radius: f32,
    pub cutoff: f32,
    pub solidify: bool,
    pub preprocess: bool,
    // FIXME(eddyb) implement.
    // pub postprocess: bool,
}

impl Default for Params {
    fn default() -> Self {
        Self {
            pad: 4,
            radius: 3.0,
            cutoff: 0.25,
            solidify: true,
            preprocess: false,
            // FIXME(eddyb) implement.
            // postprocess: false,
        }
    }
}

// Convert grayscale glyph to SDF
pub fn glyph_to_sdf(data: &mut [u8], w: usize, h: usize, params: Params) -> SdfUnorm8 {
    if params.solidify {
        solidify_alpha(data, w, h);
    }
    glyph_to_esdt(data, w, h, params)
}

// Solidify semi-transparent areas
fn solidify_alpha(data: &mut [u8], w: usize, h: usize) {
    let mut mask: Vec<u8> = vec![0; w * h];

    let get_data = |x: isize, y: isize| {
        if x >= 0 && (x as usize) < w && y >= 0 && (y as usize) < h {
            data[(y as usize) * w + (x as usize)]
        } else {
            0
        }
    };

    let mut masked = 0;

    // Mask pixels whose alpha matches their 4 adjacent neighbors (within 16 steps)
    // and who don't have black or white neighbors.
    for y in 0..(h as isize) {
        for x in 0..(w as isize) {
            let o = (x as usize) + (y as usize) * w;

            let a = get_data(x, y);
            if a == 0 || a >= 254 {
                continue;
            }

            let l = get_data(x - 1, y);
            let r = get_data(x + 1, y);
            let t = get_data(x, y - 1);
            let b = get_data(x, y + 1);

            let (min, max) = [a, l, r, t, b]
                .into_iter()
                .map(|x| (x, x))
                .reduce(|(a_min, a_max), (b_min, b_max)| (a_min.min(b_min), a_max.max(b_max)))
                .unwrap();

            if (max - min) < 16 && min > 0 && max < 254 {
                // Spread to 4 neighbors with max
                mask[o - 1] = mask[o - 1].max(a);
                mask[o - w] = mask[o - w].max(a);
                mask[o] = a;
                mask[o + 1] = mask[o + 1].max(a);
                mask[o + w] = mask[o + w].max(a);
                masked += 1;
            }
        }
    }

    if masked == 0 {
        return;
    }

    let get_mask = |x, y| mask[y * w + x];

    // Sample 3x3 area for alpha normalization factor
    for y in 0..h {
        for x in 0..w {
            let a = &mut data[y * w + x];
            if *a == 0 || *a >= 254 {
                continue;
            }

            let c = get_mask(x, y);

            let l = get_mask(x - 1, y);
            let r = get_mask(x + 1, y);
            let t = get_mask(x, y - 1);
            let b = get_mask(x, y + 1);

            let tl = get_mask(x - 1, y - 1);
            let tr = get_mask(x + 1, y - 1);
            let bl = get_mask(x - 1, y + 1);
            let br = get_mask(x + 1, y + 1);

            let m = [c, l, r, t, b, tl, tr, bl, br]
                .into_iter()
                .find(|&x| x != 0)
                .unwrap_or(0);
            if m != 0 {
                *a = (*a as f32 / m as f32 * 255.0) as u8;
            }
        }
    }
}

// Convert grayscale or color glyph to SDF using subpixel distance transform
fn glyph_to_esdt(data: &mut [u8], w: usize, h: usize, params: Params) -> SdfUnorm8 {
    // FIXME(eddyb) use `Params` itself directly in more places.
    let Params {
        pad,
        radius,
        cutoff,
        solidify: _,
        preprocess,
    } = params;

    let wp = w + pad * 2;
    let hp = h + pad * 2;
    let np = wp * hp;
    let sp = wp.max(hp);

    let mut state = State::new(sp);

    state.init_from_alpha(data, w, h, pad);
    state.compute_subpixel_offsets(data, w, h, pad, preprocess);

    {
        let State {
            outer,
            inner,
            xo,
            yo,
            xi,
            yi,
            f,
            z,
            b,
            t,
            v,
            ..
        } = &mut state;
        esdt(outer, xo, yo, wp, hp, f, z, b, t, v);
        esdt(inner, xi, yi, wp, hp, f, z, b, t, v);
    }

    // FIXME(eddyb) implement.
    // if postprocess { state.relax_subpixel_offsets(data, w, h, pad); }

    let mut alpha: Vec<u8> = (0..np)
        .map(|i| {
            let State { xo, yo, xi, yi, .. } = &state;
            let outer = ((sqr(xo[i]) + sqr(yo[i])).sqrt() - 0.5).max(0.0);
            let inner = ((sqr(xi[i]) + sqr(yi[i])).sqrt() - 0.5).max(0.0);
            let d = if outer >= inner { outer } else { -inner };
            (255.0 - 255.0 * (d / radius + cutoff))
                .round()
                .clamp(0.0, 255.0) as u8
        })
        .collect();

    if !preprocess {
        paint_into_distance_field(&mut alpha, data, w, h, pad, radius, cutoff);
    }

    SdfUnorm8 {
        width: wp,
        height: hp,
        data: alpha,
    }
}

// Helpers
fn is_black(x: f32) -> bool {
    x == 0.0
}
fn is_white(x: f32) -> bool {
    x == 1.0
}
fn is_solid(x: f32) -> bool {
    x == 0.0 || x == 1.0
}

// FIXME(eddyb) replace with `.powi(2)`
fn sqr(x: f32) -> f32 {
    x * x
}

// Paint original alpha channel into final SDF when gray
fn paint_into_distance_field(
    image: &mut [u8],
    data: &[u8],
    w: usize,
    h: usize,
    pad: usize,
    radius: f32,
    cutoff: f32,
) {
    let wp = w + pad * 2;

    let get_data = |x, y| data[y * w + x] as f32 / 255.0;

    for y in 0..h {
        for x in 0..w {
            let a = get_data(x, y);
            if !is_solid(a) {
                let j = x + pad + (y + pad) * wp;
                let d = 0.5 - a;
                image[j] = (255.0 - 255.0 * (d / radius + cutoff))
                    .round()
                    .clamp(0.0, 255.0) as u8;
            }
        }
    }
}

struct State {
    // FIXME(eddyb) `outer`/`inner` seem to always contain 0 or +∞ ???
    // (should they just be bitmaps?)

    // FIXME(eddyb) group `outer` with `{x,y}o`.
    outer: Vec<f32>,
    // FIXME(eddyb) group `inner` with `{x,y}i``.
    inner: Vec<f32>,

    xo: Vec<f32>,
    yo: Vec<f32>,
    xi: Vec<f32>,
    yi: Vec<f32>,

    // Group these separately (they're merely 1D temporary buffers).
    f: Vec<f32>, // Squared distance
    z: Vec<f32>, // Voronoi threshold
    b: Vec<f32>, // Subpixel offset parallel
    t: Vec<f32>, // Subpixel offset perpendicular
    v: Vec<u16>, // Array index
}

// FIXME(eddyb) this is a pretty misleading name.
const INF: f32 = 1e10;

impl State {
    fn new(size: usize) -> Self {
        let n = size * size;

        // FIXME(eddyb) use `v: Vec<u32>` or expose this limitation gracefully.
        assert_eq!(size as u16 as usize, size);

        let outer = vec![0.0; n];
        let inner = vec![0.0; n];

        let xo = vec![0.0; n];
        let yo = vec![0.0; n];
        let xi = vec![0.0; n];
        let yi = vec![0.0; n];

        let f = vec![0.0; size];
        let z = vec![0.0; size + 1];
        let b = vec![0.0; size];
        let t = vec![0.0; size];
        let v = vec![0; size];

        Self {
            outer,
            inner,
            xo,
            yo,
            xi,
            yi,
            f,
            z,
            b,
            t,
            v,
        }
    }

    fn init_from_alpha(&mut self, data: &mut [u8], w: usize, h: usize, pad: usize) {
        let wp = w + pad * 2;
        let hp = h + pad * 2;
        let np = wp * hp;

        let Self { outer, inner, .. } = self;

        outer[..np].fill(INF);
        inner[..np].fill(0.0);

        for y in 0..h {
            for x in 0..w {
                let a = &mut data[y * w + x];
                if *a == 0 {
                    continue;
                }

                let i = (y + pad) * wp + x + pad;
                if *a >= 254 {
                    // Fix for bad rasterizer rounding
                    *a = 255;

                    outer[i] = 0.0;
                    inner[i] = INF;
                } else {
                    outer[i] = 0.0;
                    inner[i] = 0.0;
                }
            }
        }
    }

    // Generate subpixel offsets for all border pixels
    fn compute_subpixel_offsets(
        &mut self,
        data: &[u8],
        w: usize,
        h: usize,
        pad: usize,
        relax: bool,
    ) {
        let wp = w + pad * 2;
        let hp = h + pad * 2;
        let np = wp * hp;

        let Self {
            outer,
            inner,
            xo,
            yo,
            xi,
            yi,
            ..
        } = self;

        xo[..np].fill(0.0);
        yo[..np].fill(0.0);
        xi[..np].fill(0.0);
        yi[..np].fill(0.0);

        let get_data = |x: isize, y: isize| {
            if x >= 0 && (x as usize) < w && y >= 0 && (y as usize) < h {
                data[(y as usize) * w + (x as usize)] as f32 / 255.0
            } else {
                0.0
            }
        };

        // Make vector from pixel center to nearest boundary
        for y in 0..(h as isize) {
            for x in 0..(w as isize) {
                let c = get_data(x, y);
                // NOTE(eddyb) `j - 1` (X-) / `j - wp` (Y-) positive (`pad >= 1`).
                let j = ((y as usize) + pad) * wp + (x as usize) + pad;

                if !is_solid(c) {
                    let dc = c - 0.5;

                    // NOTE(eddyb) l(eft) r(ight) t(op) b(ottom)
                    let l = get_data(x - 1, y);
                    let r = get_data(x + 1, y);
                    let t = get_data(x, y - 1);
                    let b = get_data(x, y + 1);

                    let tl = get_data(x - 1, y - 1);
                    let tr = get_data(x + 1, y - 1);
                    let bl = get_data(x - 1, y + 1);
                    let br = get_data(x + 1, y + 1);

                    let ll = (tl + l * 2.0 + bl) / 4.0;
                    let rr = (tr + r * 2.0 + br) / 4.0;
                    let tt = (tl + t * 2.0 + tr) / 4.0;
                    let bb = (bl + b * 2.0 + br) / 4.0;

                    let (min, max) = [l, r, t, b, tl, tr, bl, br]
                        .into_iter()
                        .map(|x| (x, x))
                        .reduce(|(a_min, a_max), (b_min, b_max)| {
                            (a_min.min(b_min), a_max.max(b_max))
                        })
                        .unwrap();

                    if min > 0.0 {
                        // Interior creases
                        inner[j] = INF;
                        continue;
                    }
                    if max < 1.0 {
                        // Exterior creases
                        outer[j] = INF;
                        continue;
                    }

                    let mut dx = rr - ll;
                    let mut dy = bb - tt;
                    let dl = 1.0 / (sqr(dx) + sqr(dy)).sqrt();
                    dx *= dl;
                    dy *= dl;

                    xo[j] = -dc * dx;
                    yo[j] = -dc * dy;
                } else if is_white(c) {
                    // NOTE(eddyb) l(eft) r(ight) t(op) b(ottom)
                    let l = get_data(x - 1, y);
                    let r = get_data(x + 1, y);
                    let t = get_data(x, y - 1);
                    let b = get_data(x, y + 1);

                    if is_black(l) {
                        xo[j - 1] = 0.4999;
                        outer[j - 1] = 0.0;
                        inner[j - 1] = 0.0;
                    }
                    if is_black(r) {
                        xo[j + 1] = -0.4999;
                        outer[j + 1] = 0.0;
                        inner[j + 1] = 0.0;
                    }

                    if is_black(t) {
                        yo[j - wp] = 0.4999;
                        outer[j - wp] = 0.0;
                        inner[j - wp] = 0.0;
                    }
                    if is_black(b) {
                        yo[j + wp] = -0.4999;
                        outer[j + wp] = 0.0;
                        inner[j + wp] = 0.0;
                    }
                }
            }
        }

        // Blend neighboring offsets but preserve normal direction
        // Uses xo as input, xi as output
        // Improves quality slightly, but slows things down.
        if relax {
            let check_cross = |nx, ny, dc, dl, dr, dxl, dyl, dxr, dyr| {
                ((dxl * nx + dyl * ny) * (dc * dl) > 0.0)
                    && ((dxr * nx + dyr * ny) * (dc * dr) > 0.0)
                    && ((dxl * dxr + dyl * dyr) * (dl * dr) > 0.0)
            };

            for y in 0..(h as isize) {
                for x in 0..(w as isize) {
                    // NOTE(eddyb) `j - 1` (X-) / `j - wp` (Y-) positive (`pad >= 1`).
                    let j = ((y as usize) + pad) * wp + (x as usize) + pad;

                    let nx = xo[j];
                    let ny = yo[j];
                    if nx == 0.0 && ny == 0.0 {
                        continue;
                    }

                    // NOTE(eddyb) c(enter) l(eft) r(ight) t(op) b(ottom)
                    let c = get_data(x, y);
                    let l = get_data(x - 1, y);
                    let r = get_data(x + 1, y);
                    let t = get_data(x, y - 1);
                    let b = get_data(x, y + 1);

                    let dxl = xo[j - 1];
                    let dxr = xo[j + 1];
                    let dxt = xo[j - wp];
                    let dxb = xo[j + wp];

                    let dyl = yo[j - 1];
                    let dyr = yo[j + 1];
                    let dyt = yo[j - wp];
                    let dyb = yo[j + wp];

                    let mut dx = nx;
                    let mut dy = ny;
                    let mut dw = 1.0;

                    let dc = c - 0.5;
                    let dl = l - 0.5;
                    let dr = r - 0.5;
                    let dt = t - 0.5;
                    let db = b - 0.5;

                    if !is_solid(l) && !is_solid(r) {
                        if check_cross(nx, ny, dc, dl, dr, dxl, dyl, dxr, dyr) {
                            dx += (dxl + dxr) / 2.0;
                            dy += (dyl + dyr) / 2.0;
                            dw += 1.0;
                        }
                    }

                    if !is_solid(t) && !is_solid(b) {
                        if check_cross(nx, ny, dc, dt, db, dxt, dyt, dxb, dyb) {
                            dx += (dxt + dxb) / 2.0;
                            dy += (dyt + dyb) / 2.0;
                            dw += 1.0;
                        }
                    }

                    if !is_solid(l) && !is_solid(t) {
                        if check_cross(nx, ny, dc, dl, dt, dxl, dyl, dxt, dyt) {
                            dx += (dxl + dxt - 1.0) / 2.0;
                            dy += (dyl + dyt - 1.0) / 2.0;
                            dw += 1.0;
                        }
                    }

                    if !is_solid(r) && !is_solid(t) {
                        if check_cross(nx, ny, dc, dr, dt, dxr, dyr, dxt, dyt) {
                            dx += (dxr + dxt + 1.0) / 2.0;
                            dy += (dyr + dyt - 1.0) / 2.0;
                            dw += 1.0;
                        }
                    }

                    if !is_solid(l) && !is_solid(b) {
                        if check_cross(nx, ny, dc, dl, db, dxl, dyl, dxb, dyb) {
                            dx += (dxl + dxb - 1.0) / 2.0;
                            dy += (dyl + dyb + 1.0) / 2.0;
                            dw += 1.0;
                        }
                    }

                    if !is_solid(r) && !is_solid(b) {
                        if check_cross(nx, ny, dc, dr, db, dxr, dyr, dxb, dyb) {
                            dx += (dxr + dxb + 1.0) / 2.0;
                            dy += (dyr + dyb + 1.0) / 2.0;
                            dw += 1.0;
                        }
                    }

                    let nn = (nx * nx + ny * ny).sqrt();
                    let ll = (dx * nx + dy * ny) / nn;

                    dx = nx * ll / dw / nn;
                    dy = ny * ll / dw / nn;

                    xi[j] = dx;
                    yi[j] = dy;
                }
            }
        }

        // Produce zero points for positive and negative DF, at +0.5 / -0.5.
        // Splits xs into xo/xi
        for y in 0..(h as isize) {
            for x in 0..(w as isize) {
                // NOTE(eddyb) `j - 1` (X-) / `j - wp` (Y-) positive (`pad >= 1`).
                let j = ((y as usize) + pad) * wp + (x as usize) + pad;

                // NOTE(eddyb) `if relax` above changed `xs`/`ys` in the original.
                let (nx, ny) = if relax {
                    (xi[j], yi[j])
                } else {
                    (xo[j], yo[j])
                };
                if nx == 0.0 && ny == 0.0 {
                    continue;
                }

                let nn = (sqr(nx) + sqr(ny)).sqrt();

                let sx = if ((nx / nn).abs() - 0.5) > 0.0 {
                    nx.signum() as isize
                } else {
                    0
                };
                let sy = if ((ny / nn).abs() - 0.5) > 0.0 {
                    ny.signum() as isize
                } else {
                    0
                };

                let c = get_data(x, y);
                let d = get_data(x + sx, y + sy);
                // FIXME(eddyb) is this inefficient? (was `Math.sign(d - c)`)
                let s = (d - c).total_cmp(&0.0) as i8 as f32;

                let dlo = (nn + 0.4999 * s) / nn;
                let dli = (nn - 0.4999 * s) / nn;

                xo[j] = nx * dlo;
                yo[j] = ny * dlo;
                xi[j] = nx * dli;
                yi[j] = ny * dli;
            }
        }
    }
}

// 2D subpixel distance transform by unconed
// extended from Felzenszwalb & Huttenlocher https://cs.brown.edu/~pff/papers/dt-final.pdf
fn esdt(
    mask: &mut [f32],
    xs: &mut [f32],
    ys: &mut [f32],
    w: usize,
    h: usize,
    f: &mut [f32],
    z: &mut [f32],
    b: &mut [f32],
    t: &mut [f32],
    v: &mut [u16],
) {
    // FIXME(eddyb) use `v: &mut [u32]` or expose this limitation gracefully.
    let w_as_u16 = u16::try_from(w).unwrap();
    let h_as_u16 = u16::try_from(h).unwrap();

    for x in 0..w {
        esdt1d(mask, ys, xs, x, w, h_as_u16, f, z, b, t, v)
    }
    for y in 0..h {
        esdt1d(mask, xs, ys, y * w, 1, w_as_u16, f, z, b, t, v)
    }
}

// 1D subpixel distance transform
fn esdt1d(
    mask: &mut [f32],
    xs: &mut [f32],
    ys: &mut [f32],
    offset: usize,
    stride: usize,
    length: u16,
    f: &mut [f32], // Squared distance
    z: &mut [f32], // Voronoi threshold
    b: &mut [f32], // Subpixel offset parallel
    t: &mut [f32], // Subpixel offset perpendicular
    v: &mut [u16], // Array index
) {
    v[0] = 0;
    b[0] = xs[offset];
    t[0] = ys[offset];
    z[0] = -INF;
    z[1] = INF;
    f[0] = if mask[offset] != 0.0 {
        INF
    } else {
        ys[offset] * ys[offset]
    };

    // Scan along array and build list of critical minima
    {
        let mut k_len = 1;
        for q in 1..length {
            let o = offset + usize::from(q) * stride;

            // Perpendicular
            let dx = xs[o];
            let dy = ys[o];
            let fq = if mask[o] != 0.0 { INF } else { dy * dy };
            f[usize::from(q)] = fq;
            t[usize::from(q)] = dy;

            // Parallel
            let qs = q as f32 + dx;
            let q2 = qs * qs;
            b[usize::from(q)] = qs;

            // Remove any minima eclipsed by this one
            let mut s;
            loop {
                let r = usize::from(v[k_len - 1]);
                let rs = b[r];

                let r2 = rs * rs;
                s = (fq - f[r] + q2 - r2) / (qs - rs) / 2.0;

                if !(s <= z[k_len - 1]) {
                    break;
                }

                k_len -= 1;
                if k_len == 0 {
                    break;
                }
            }

            // Add to minima list
            v[k_len] = q;
            z[k_len] = s;
            z[k_len + 1] = INF;
            k_len += 1;
        }
    }

    // Resample array based on critical minima
    {
        let mut k = 0;
        for q in 0..length {
            // Skip eclipsed minima
            while z[k + 1] < q as f32 {
                k += 1;
            }

            let r = v[k];
            let rs = b[usize::from(r)];
            let dy = t[usize::from(r)];

            // Distance from integer index to subpixel location of minimum
            let rq = rs - q as f32;

            let o = offset + usize::from(q) * stride;
            xs[o] = rq;
            ys[o] = dy;

            // Mark cell as having propagated
            if r != q {
                mask[o] = 0.0;
            }
        }
    }
}
