//! Bitmap-based "closest included (black) pixel" bruteforce search, a well-known
//! algorithm, notably popularized by Valve in their 2007 SIGGRAPH paper on SDFs,
//! ["Improved Alpha-Tested Magnification for Vector Textures and Special Effects"](https://steamcdn-a.akamaihd.net/apps/valve/2007/SIGGRAPH2007_AlphaTestedMagnification.pdf).

use crate::{Bitmap, SdfUnorm8};

pub fn sdf(bitmap: &Bitmap, sdf_size: usize, spread: usize) -> SdfUnorm8 {
    assert!(bitmap.w.is_power_of_two() && bitmap.h.is_power_of_two() && sdf_size.is_power_of_two());
    let scale = bitmap.w.max(bitmap.h) / sdf_size;
    assert_ne!(scale, 0);

    let spread = spread * scale;

    // FIXME(eddyb) maybe this should be a
    let width = bitmap.w / scale;
    let height = bitmap.h / scale;
    SdfUnorm8 {
        width,
        height,
        data: (0..width * height)
            .map(|i| (i % width, i / width))
            .map(|(x, y)| {
                let (x, y) = (x * scale + scale / 2, y * scale + scale / 2);
                let inside = bitmap.get(x, y);
                // FIXME(eddyb) this could use a spiral search, and maybe better bitmap
                // access, e.g. "which bits in a block are different than the center x,y".
                let dist = (((y.saturating_sub(spread)..=(y + spread))
                    .flat_map(|y2| {
                        (x.saturating_sub(spread)..=(x + spread)).map(move |x2| (x2, y2))
                    })
                    .filter(|&(x2, y2)| {
                        x2 < bitmap.w && y2 < bitmap.h && bitmap.get(x2, y2) != inside
                    })
                    .map(|(x2, y2)| x2.abs_diff(x).pow(2) + y2.abs_diff(y).pow(2))
                    .min()
                    .unwrap_or(usize::MAX) as f32)
                    .sqrt()
                    / (spread as f32))
                    .clamp(0.0, 1.0);
                let signed_dist = if inside { -dist } else { dist };

                // [-1, +1] -> [0, 1]
                (((signed_dist + 1.0) / 2.0) * 255.0).round() as u8
            })
            .collect(),
    }
}
