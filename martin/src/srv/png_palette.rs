//! Indexed (palette) PNG encoding for rendered images.
//!
//! Basemap tiles are mostly flat fills, so a small per-image palette keeps them close to
//! the RGBA original at a fraction of the bytes. Indices are packed at the smallest bit
//! depth the palette allows and written unfiltered, which is what compresses best here.

/// Quantise `img` to at most `max_colors` colours and encode it as an indexed PNG.
pub fn encode(img: &image::RgbaImage, max_colors: u16) -> Result<Vec<u8>, String> {
    let (w, h) = (img.width() as usize, img.height() as usize);
    let qimg = quantizr::Image::new(img.as_raw(), w, h).map_err(|e| e.to_string())?;
    let mut opts = quantizr::Options::default();
    opts.set_max_colors(i32::from(max_colors)).map_err(|e| e.to_string())?;
    let mut res = quantizr::QuantizeResult::quantize(&qimg, &opts);
    res.set_dithering_level(0.0).map_err(|e| e.to_string())?;
    let mut indices = vec![0u8; w * h];
    res.remap_image(&qimg, &mut indices).map_err(|e| e.to_string())?;
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
    for (y, row) in indices.chunks_exact(w).enumerate() {
        for (x, &i) in row.iter().enumerate() {
            let shift = 8 - bits * (x % per_byte + 1);
            data[y * row_len + x / per_byte] |= remap[usize::from(i)] << shift;
        }
    }

    let mut out = Vec::new();
    let mut enc = png::Encoder::new(&mut out, img.width(), img.height());
    enc.set_color(png::ColorType::Indexed);
    enc.set_depth(depth);
    enc.set_palette(order.iter().flat_map(|&i| [palette[i].r, palette[i].g, palette[i].b]).collect::<Vec<u8>>());
    let trns: Vec<u8> = order.iter().map(|&i| palette[i].a).take_while(|&a| a != 255).collect();
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
