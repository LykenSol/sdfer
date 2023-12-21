//! SDF ("Signed Distance Field") generation algorithms.
//!
//! (see per-module documentation for more details)

pub mod bruteforce_bitmap;
pub mod esdt;

mod img;
pub use img::*;
