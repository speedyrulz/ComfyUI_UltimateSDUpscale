/**
 * The four overlapping tiles the batch latent overlap node samples.
 *
 * This mirrors latent_tile_regions() in batch_latent_overlap.py so that what the tile prompt node
 * shows is what the upscale node will actually sample. That module works in latent units, so the
 * padding is rounded to whole latent blocks there; here everything is in pixels and the difference
 * is at most one block, which is not visible in a preview.
 */

export const CORNERS = [
    { key: "top_left", label: "top left" },
    { key: "top_right", label: "top right" },
    { key: "bottom_left", label: "bottom left" },
    { key: "bottom_right", label: "bottom right" },
];

/**
 * The four tile regions of an image, in pixels.
 *
 * The image is split into four evenly sized quadrants, and every edge that touches another tile is
 * extended into it by the padding, so all four tiles are the same size and together cover the image.
 *
 * @param {number} width image width
 * @param {number} height image height
 * @param {number} padding how far a tile reaches into its neighbours
 * @returns {{regions: Array<{x: number, y: number, width: number, height: number, key: string, label: string}>,
 *            overlapWidth: number, overlapHeight: number}}
 */
export function tileRegions(width, height, padding) {
    const halfW = Math.floor(width / 2);
    const halfH = Math.floor(height / 2);

    const padW = Math.min(Math.max(1, Math.round(padding)), halfW);
    const padH = Math.min(Math.max(1, Math.round(padding)), halfH);

    const tileW = Math.min(halfW + padW, width);
    const tileH = Math.min(halfH + padH, height);

    const xs = [0, width - tileW];
    const ys = [0, height - tileH];

    const regions = [];
    for (let row = 0; row < 2; row++) {
        for (let column = 0; column < 2; column++) {
            const corner = CORNERS[row * 2 + column];
            regions.push({
                x: xs[column],
                y: ys[row],
                width: tileW,
                height: tileH,
                key: corner.key,
                label: corner.label,
            });
        }
    }

    return {
        regions,
        overlapWidth: 2 * tileW - width,
        overlapHeight: 2 * tileH - height,
    };
}

/**
 * The part of a tile that is its own quarter of the image, as an offset within the tile.
 *
 * The rest of the tile is the padding it shares with its neighbours, which is what gets averaged
 * together during sampling. Drawing the two apart is what makes the overlap visible in the preview.
 */
export function ownQuarter(region, width, height) {
    const halfW = Math.floor(width / 2);
    const halfH = Math.floor(height / 2);
    const left = region.x === 0 ? 0 : halfW - region.x;
    const top = region.y === 0 ? 0 : halfH - region.y;
    return {
        left,
        top,
        width: Math.min(halfW, region.width - left),
        height: Math.min(halfH, region.height - top),
    };
}
