//! SDF ("Signed Distance Field") generation algorithms.

pub mod bruteforce_bitmap;
pub mod esdt;

// FIXME(eddyb) maybe consider making each word a `8x8` pixel block instead?
pub struct Bitmap {
    pub w: usize,
    pub h: usize,
    words: Vec<u64>,
}
pub struct BitmapEntry<'a> {
    word: &'a mut u64,
    mask: u64,
}

impl Bitmap {
    pub fn new(w: usize, h: usize) -> Self {
        Self {
            w,
            h,
            words: vec![0; (w * h + 63) / 64],
        }
    }

    pub fn word_idx_and_mask(&self, x: usize, y: usize) -> (usize, u64) {
        let i = y * self.w + x;
        (i / 64, 1 << (i % 64))
    }

    pub fn get(&self, x: usize, y: usize) -> bool {
        let (i, mask) = self.word_idx_and_mask(x, y);
        (self.words[i] & mask) != 0
    }

    pub fn at(&mut self, x: usize, y: usize) -> BitmapEntry<'_> {
        let (i, mask) = self.word_idx_and_mask(x, y);
        BitmapEntry {
            word: &mut self.words[i],
            mask,
        }
    }
}
impl BitmapEntry<'_> {
    pub fn get(&self) -> bool {
        (*self.word & self.mask) != 0
    }
    pub fn set(&mut self, value: bool) {
        if value {
            *self.word |= self.mask;
        } else {
            *self.word &= !self.mask;
        }
    }
}

pub struct SdfUnorm8 {
    pub width: usize,
    pub height: usize,
    pub data: Vec<u8>,
}
