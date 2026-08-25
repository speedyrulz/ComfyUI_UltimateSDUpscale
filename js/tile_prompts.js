/**
 * The tile prompt node: shows the loaded image split into the four tiles the batch latent overlap
 * node samples, with a prompt box under each one.
 *
 * The four prompt widgets stay the real, serialised inputs; they are only hidden on the canvas and
 * mirrored into the text boxes of the grid, so a workflow saved by this node loads anywhere.
 */

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { tileRegions, ownQuarter, CORNERS } from "./tile_geometry.js";

const NODE_TYPE = "UltimateSDUpscaleTilePrompts";
const PREVIEW_WIDTH = 300;      // the width each tile is drawn at, in pixels
const MIN_NODE_WIDTH = 460;
const STYLE_ID = "usdu-tile-prompts-style";

const STYLE = `
.usdu-tiles {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
    box-sizing: border-box;
    width: 100%;
    height: 100%;
    overflow: auto;
    padding: 2px;
}
.usdu-tiles .usdu-tile {
    display: flex;
    flex-direction: column;
    gap: 4px;
    min-width: 0;
}
.usdu-tiles .usdu-tile-label {
    font-size: 10px;
    line-height: 12px;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    opacity: 0.75;
    color: var(--descrip-text, #999);
}
.usdu-tiles canvas {
    width: 100%;
    height: auto;
    display: block;
    border-radius: 4px;
    border: 1px solid var(--border-color, #4e4e4e);
    background: var(--comfy-input-bg, #222);
}
.usdu-tiles textarea {
    width: 100%;
    box-sizing: border-box;
    resize: none;
    min-height: 48px;
    font-family: inherit;
    font-size: 12px;
    padding: 4px 6px;
    border-radius: 4px;
    border: 1px solid var(--border-color, #4e4e4e);
    background: var(--comfy-input-bg, #222);
    color: var(--input-text, #ddd);
}
.usdu-tiles .usdu-empty {
    grid-column: 1 / -1;
    font-size: 12px;
    opacity: 0.7;
    color: var(--descrip-text, #999);
    padding: 6px 2px;
}
`;

function installStyle() {
    if (document.getElementById(STYLE_ID)) return;
    const style = document.createElement("style");
    style.id = STYLE_ID;
    style.textContent = STYLE;
    document.head.appendChild(style);
}

/** The url ComfyUI serves a picked input image from, handling subfolders and the [type] suffix. */
function imageUrl(value) {
    if (!value) return null;
    let name = value;
    let type = "input";

    const annotation = name.lastIndexOf(" [");
    if (annotation > -1 && name.endsWith("]")) {
        type = name.substring(annotation + 2, name.length - 1);
        name = name.substring(0, annotation);
    }

    let subfolder = "";
    const separator = name.lastIndexOf("/");
    if (separator > -1) {
        subfolder = name.substring(0, separator);
        name = name.substring(separator + 1);
    }

    return api.apiURL(
        `/view?filename=${encodeURIComponent(name)}&type=${encodeURIComponent(type)}` +
        `&subfolder=${encodeURIComponent(subfolder)}&rand=${Math.random()}`
    );
}

app.registerExtension({
    name: "UltimateSDUpscale.TilePrompts",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== NODE_TYPE) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);
            try {
                buildTileGrid(this);
            } catch (error) {
                console.error("[UltimateSDUpscale] could not build the tile grid", error);
            }
            return result;
        };
    },
});

function buildTileGrid(node) {
    installStyle();

    const promptWidgets = CORNERS.map((corner) =>
        node.widgets?.find((widget) => widget.name === `prompt_${corner.key}`)
    );
    if (promptWidgets.some((widget) => !widget)) return;

    // The prompt widgets keep holding the values, they are just not drawn on the canvas any more
    for (const widget of promptWidgets) {
        widget.hidden = true;
        widget.computeSize = () => [0, -4];
    }

    const container = document.createElement("div");
    container.className = "usdu-tiles";

    const empty = document.createElement("div");
    empty.className = "usdu-empty";
    empty.textContent = "Pick an image to see its four tiles.";
    container.appendChild(empty);

    const cells = CORNERS.map((corner, index) => {
        const cell = document.createElement("div");
        cell.className = "usdu-tile";

        const label = document.createElement("div");
        label.className = "usdu-tile-label";
        label.textContent = corner.label;

        const canvas = document.createElement("canvas");
        canvas.width = PREVIEW_WIDTH;
        canvas.height = Math.round(PREVIEW_WIDTH * 0.75);

        const textarea = document.createElement("textarea");
        textarea.placeholder = `${corner.label} prompt`;
        textarea.spellcheck = false;
        textarea.value = promptWidgets[index].value ?? "";
        textarea.addEventListener("input", () => {
            promptWidgets[index].value = textarea.value;
        });
        // Keep canvas shortcuts (delete, arrow keys) from firing while typing
        textarea.addEventListener("keydown", (event) => event.stopPropagation());

        cell.append(label, canvas, textarea);
        container.appendChild(cell);
        return { cell, canvas, textarea };
    });

    const widget = node.addDOMWidget("tiles", "usdu_tiles", container, { serialize: false });

    const state = { image: null, width: 0, height: 0 };

    const showCells = (visible) => {
        empty.style.display = visible ? "none" : "";
        for (const { cell } of cells) cell.style.display = visible ? "" : "none";
    };
    showCells(false);

    const draw = () => {
        if (!state.image || !state.width || !state.height) {
            showCells(false);
            resize();
            return;
        }
        showCells(true);

        const paddingWidget = node.widgets?.find((w) => w.name === "tile_padding");
        const padding = Number(paddingWidget?.value) || 32;
        const { regions } = tileRegions(state.width, state.height, padding);

        regions.forEach((region, index) => {
            const canvas = cells[index].canvas;
            const scale = PREVIEW_WIDTH / region.width;
            canvas.width = PREVIEW_WIDTH;
            canvas.height = Math.max(1, Math.round(region.height * scale));

            const context = canvas.getContext("2d");
            context.clearRect(0, 0, canvas.width, canvas.height);
            context.drawImage(
                state.image,
                region.x, region.y, region.width, region.height,
                0, 0, canvas.width, canvas.height
            );

            // Mark off the tile's own quarter: everything outside it is the overlap that is
            // averaged with the neighbouring tiles while sampling
            const quarter = ownQuarter(region, state.width, state.height);
            context.save();
            context.strokeStyle = "rgba(255, 255, 255, 0.55)";
            context.lineWidth = 1;
            context.setLineDash([4, 3]);
            context.strokeRect(
                Math.round(quarter.left * scale) + 0.5,
                Math.round(quarter.top * scale) + 0.5,
                Math.max(1, Math.round(quarter.width * scale) - 1),
                Math.max(1, Math.round(quarter.height * scale) - 1)
            );
            context.restore();
        });

        resize();
    };

    const resize = () => {
        // Two rows of (label + preview + text box), plus the gaps around them
        const previewHeight = cells[0].canvas.height
            ? (cells[0].canvas.clientHeight || cells[0].canvas.height * 0.5)
            : 0;
        const rowHeight = 12 + previewHeight + 56 + 12;
        const height = state.image ? Math.round(rowHeight * 2) : 40;
        widget.computeSize = () => [node.size[0], height];
        container.style.minHeight = `${height}px`;
        const width = Math.max(node.size[0], MIN_NODE_WIDTH);
        node.setSize([width, node.computeSize([width, node.size[1]])[1]]);
        node.setDirtyCanvas(true, true);
    };

    const loadImage = () => {
        const value = node.widgets?.find((w) => w.name === "image")?.value;
        const url = imageUrl(value);
        if (!url) {
            state.image = null;
            draw();
            return;
        }
        const image = new Image();
        image.onload = () => {
            state.image = image;
            state.width = image.naturalWidth;
            state.height = image.naturalHeight;
            draw();
        };
        image.onerror = () => {
            state.image = null;
            draw();
        };
        image.src = url;
    };

    // Redraw whenever the picked image or the padding changes
    for (const name of ["image", "tile_padding"]) {
        const target = node.widgets?.find((w) => w.name === name);
        if (!target) continue;
        const original = target.callback;
        target.callback = function (...args) {
            const value = original?.apply(this, args);
            (name === "image" ? loadImage : draw)();
            return value;
        };
    }

    // A workflow that was just loaded brings its own prompt values and picked image
    const onConfigure = node.onConfigure;
    node.onConfigure = function () {
        const value = onConfigure?.apply(this, arguments);
        cells.forEach(({ textarea }, index) => {
            textarea.value = promptWidgets[index].value ?? "";
        });
        loadImage();
        return value;
    };

    requestAnimationFrame(loadImage);
}
