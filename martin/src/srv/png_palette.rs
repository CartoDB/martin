//! Indexed (palette) PNG encoding for rendered images.
//!
//! Basemap tiles are mostly flat fills, so a small per-image palette keeps them close to
//! the RGBA original at a fraction of the bytes. Each tile gets the smallest palette that
//! is good enough, up to `max_colors`. Indices are packed at the smallest bit depth the
//! palette allows and written unfiltered, which is what compresses best here.

use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};

/// Palette sizes tried in order. 16 is the largest 4-bit palette, so smaller ones save little.
const BUDGETS: [u16; 7] = [16, 24, 32, 48, 64, 96, 128];
/// Mean squared RGBA error per pixel.
const MAX_MSE: u64 = 2;
/// Squared RGBA error allowed for any colour covering at least 1/`VISIBLE_SHARE` of the
/// image: a road, a park, a label fill. Antialiasing pixels are rarer and may be off more.
const MAX_VISIBLE_ERR: u32 = 75;
const VISIBLE_SHARE: usize = 4000;

/// Quantise `img` to at most `max_colors` colours and encode it as an indexed PNG.
///
/// Tries each of `BUDGETS` below `max_colors`, then `max_colors` itself, and keeps the
/// first palette within `MAX_MSE` and `MAX_VISIBLE_ERR`.
pub fn encode(img: &image::RgbaImage, max_colors: u16) -> Result<Vec<u8>, String> {
    let (w, h) = (img.width() as usize, img.height() as usize);
    if w == 0 || h == 0 {
        return Err("cannot encode an empty image".to_owned());
    }
    let qimg = quantizr::Image::new(img.as_raw(), w, h).map_err(|e| e.to_string())?;
    let mut hist = quantizr::Histogram::new();
    hist.add_image(&qimg);
    let colors = Colors::of(img.as_raw());
    let distinct = colors.counts.len();
    let uniq = quantizr::Image::new(&colors.uniq, distinct, 1).map_err(|e| e.to_string())?;
    let mut lut = vec![0u8; distinct];

    let budgets = BUDGETS.iter().copied().filter(|&b| b < max_colors);
    let mut res = None;
    for budget in budgets.chain([max_colors]) {
        let mut opts = quantizr::Options::default();
        opts.set_max_colors(i32::from(budget))
            .map_err(|e| e.to_string())?;
        let mut r = quantizr::QuantizeResult::quantize_histogram(&hist, &opts);
        r.set_dithering_level(0.0).map_err(|e| e.to_string())?;
        // No-dither remap is per colour, so remapping each distinct colour once is exact.
        r.remap_image(&uniq, &mut lut).map_err(|e| e.to_string())?;
        let done = budget == max_colors
            || distinct <= usize::from(budget)
            || colors.good_enough(r.get_palette(), &lut);
        res = Some(r);
        if done {
            break;
        }
    }
    let res = res.ok_or("no palette")?;
    let palette = res.get_palette();
    let palette = &palette.entries[..palette.count as usize];

    // Translucent entries first, so the tRNS chunk can stop at the last of them.
    let mut order: Vec<usize> = (0..palette.len()).collect();
    order.sort_by_key(|&i| palette[i].a == 255);
    let mut remap = [0u8; 256];
    for (new, &old) in order.iter().enumerate() {
        remap[old] = u8::try_from(new).map_err(|e| e.to_string())?;
    }

    let depth = match palette.len() {
        0..=2 => png::BitDepth::One,
        3..=4 => png::BitDepth::Two,
        5..=16 => png::BitDepth::Four,
        _ => png::BitDepth::Eight,
    };
    let bits = depth as usize;
    let per_byte = 8 / bits;
    let row_len = w.div_ceil(per_byte);
    let mut data = vec![0u8; row_len * h];
    for (y, row) in colors.ids.chunks_exact(w).enumerate() {
        for (x, &id) in row.iter().enumerate() {
            let shift = 8 - bits * (x % per_byte + 1);
            data[y * row_len + x / per_byte] |= remap[usize::from(lut[id as usize])] << shift;
        }
    }

    let mut out = Vec::new();
    let mut enc = png::Encoder::new(&mut out, img.width(), img.height());
    enc.set_color(png::ColorType::Indexed);
    enc.set_depth(depth);
    enc.set_palette(
        order
            .iter()
            .flat_map(|&i| [palette[i].r, palette[i].g, palette[i].b])
            .collect::<Vec<u8>>(),
    );
    let trns: Vec<u8> = order
        .iter()
        .map(|&i| palette[i].a)
        .take_while(|&a| a != 255)
        .collect();
    if !trns.is_empty() {
        enc.set_trns(trns);
    }
    enc.set_deflate_compression(png::DeflateCompression::Level(6));
    enc.set_filter(png::Filter::NoFilter);
    enc.write_header()
        .and_then(|mut wtr| wtr.write_image_data(&data))
        .map_err(|e| e.to_string())?;
    Ok(out)
}

/// The distinct colours of an image, how often each appears, and which one each pixel is.
struct Colors {
    uniq: Vec<u8>,
    counts: Vec<u32>,
    ids: Vec<u32>,
}

impl Colors {
    #[expect(clippy::cast_possible_truncation)]
    fn of(rgba: &[u8]) -> Self {
        let mut slots = HashMap::<u32, u32, BuildHasherDefault<MulHasher>>::default();
        let (mut uniq, mut counts) = (Vec::new(), Vec::new());
        let mut ids = Vec::with_capacity(rgba.len() / 4);
        let (mut last, mut slot) = (None, 0);
        for px in rgba.as_chunks::<4>().0 {
            // Like quantizr, every alpha-0 pixel is transparent black.
            let k = if px[3] == 0 {
                0
            } else {
                u32::from_le_bytes(*px)
            };
            if last != Some(k) {
                last = Some(k);
                slot = *slots.entry(k).or_insert_with(|| {
                    uniq.extend_from_slice(&k.to_le_bytes());
                    counts.push(0);
                    (counts.len() - 1) as u32
                });
            }
            counts[slot as usize] += 1;
            ids.push(slot);
        }
        Self { uniq, counts, ids }
    }

    fn good_enough(&self, palette: &quantizr::Palette, lut: &[u8]) -> bool {
        let visible = self.ids.len().div_ceil(VISIBLE_SHARE);
        let mut sq_err = 0u64;
        let uniq = self.uniq.as_chunks::<4>().0;
        for ((c, &n), &i) in uniq.iter().zip(&self.counts).zip(lut) {
            let p = palette.entries[usize::from(i)];
            let d2: u32 = c
                .iter()
                .zip([p.r, p.g, p.b, p.a])
                .map(|(&a, b)| u32::from(a.abs_diff(b)).pow(2))
                .sum();
            if n as usize >= visible && d2 > MAX_VISIBLE_ERR {
                return false;
            }
            sq_err += u64::from(n) * u64::from(d2);
        }
        sq_err <= MAX_MSE * self.ids.len() as u64
    }
}

/// Pixel colours as keys need no strong hashing.
#[derive(Default)]
struct MulHasher(u64);

impl Hasher for MulHasher {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 = (self.0 ^ u64::from(b)).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        }
    }
    fn write_u32(&mut self, k: u32) {
        self.0 = u64::from(k).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    }
}
