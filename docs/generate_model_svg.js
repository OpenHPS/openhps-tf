// Generates SVG architecture diagrams for the accuracy model from the saved
// TF.js model topologies, replacing hand-drawn ASCII in docs/MODEL.md.
//
//   node docs/generate_model_svg.js
//
// Writes docs/model/*.svg. Pure Node (no graphviz/browser needed).
'use strict';
const fs = require('fs');
const path = require('path');

const OUT = path.join(__dirname, 'model');
fs.mkdirSync(OUT, { recursive: true });

const COLORS = {
    Input: { fill: '#e3f2fd', stroke: '#1e88e5', text: '#0d47a1' },
    Dense: { fill: '#f3e5f5', stroke: '#8e24aa', text: '#4a148c' },
    Dropout: { fill: '#fce4ec', stroke: '#d81b60', text: '#880e4f' },
    Concatenate: { fill: '#fff9c4', stroke: '#f9a825', text: '#f57f17' },
    Output: { fill: '#ffe0b2', stroke: '#ef6c00', text: '#e65100' },
    default: { fill: '#eceff1', stroke: '#607d8b', text: '#263238' }
};
const BOX_W = 190;
const BOX_H = 62;
const COL_GAP = 90;
const ROW_GAP = 40;
const MARGIN = 60;

function shapeOf(layer, inputDims) {
    switch (layer.class) {
        case 'InputLayer':
            return layer.shape ? layer.shape[layer.shape.length - 1] : 0;
        case 'Dense':
            return layer.units;
        case 'Dropout':
            return inputDims[0];
        case 'Concatenate':
            return inputDims.reduce((a, b) => a + b, 0);
        default:
            return inputDims[0] || 0;
    }
}

function colorFor(layer, isOutput) {
    if (isOutput) return COLORS.Output;
    return COLORS[layer.class] || COLORS.default;
}

function esc(s) {
    return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

function layerLabel(layer, inputShape) {
    const lines = [`<tspan x="0" dy="0" font-weight="700">${esc(layer.class)}</tspan>`];
    if (layer.name) {
        lines.push(`<tspan x="0" dy="15" font-size="9" fill="#546e7a">${esc(layer.name)}</tspan>`);
    }
    if (layer.class === 'InputLayer') {
        lines.push(`<tspan x="0" dy="15" font-size="10" fill="#455a64">shape [${inputShape}]</tspan>`);
    } else if (layer.class === 'Dense') {
        const inSh = layer.shape ? `in [${layer.shape[1]}] · ` : '';
        lines.push(`<tspan x="0" dy="15" font-size="10" fill="#455a64">${inSh}${layer.units} · ${layer.activation}</tspan>`);
    } else if (layer.class === 'Dropout') {
        lines.push(`<tspan x="0" dy="15" font-size="10" fill="#455a64">rate ${layer.rate}</tspan>`);
    } else if (layer.class === 'Concatenate') {
        lines.push(`<tspan x="0" dy="15" font-size="10" fill="#455a64">→ [${inputShape}]</tspan>`);
    }
    return `<text x="0" y="0" text-anchor="middle" font-family="Segoe UI, sans-serif" font-size="12" fill="${COLORS.default.text}">${lines.join('')}</text>`;
}

// Returns {layers: [{class,name,units,activation,rate,shape,inbound}], inputs: [name...], output: name}
function parseTopology(topology, subModels) {
    const cfg = topology.config;
    const layers = [];
    const nameMap = {};
    const subModelLast = {};
    for (const l of cfg.layers) {
        const c = l.config;
        let cls = l.class_name;
        // Expand nested Sequential/Model sub-models (e.g. the encoder inside
        // the composed accuracy model): chain the inner layers onto the
        // parent's actual input.
        if ((cls === 'Sequential' || cls === 'Model') && subModels[c.name]) {
            const sub = parseTopology(subModels[c.name], subModels).layers;
            const parentInbound = l.inbound_nodes || [];
            sub.forEach((sl, idx) => {
                sl.inbound = idx === 0
                    ? parentInbound
                    : [[[sub[idx - 1].name, 0, 0, {}]]];
                nameMap[sl.name] = sl;
                layers.push(sl);
            });
            subModelLast[c.name] = sub[sub.length - 1].name;
            continue;
        }
        const layer = {
            class: cls,
            name: c.name,
            units: c.units,
            activation: c.activation,
            rate: c.rate,
            shape: c.batch_input_shape,
            inbound: l.inbound_nodes || []
        };
        nameMap[layer.name] = layer;
        layers.push(layer);
    }
    // Remap any inbound reference to a sub-model onto its output layer.
    for (const l of layers) {
        if (l.inbound[0]) {
            l.inbound[0].forEach((e) => { e[0] = subModelLast[e[0]] || e[0]; });
        }
    }
    // Sequential topologies have no inbound_nodes: chain consecutive layers.
    if (!cfg.input_layers) {
        for (let i = 1; i < layers.length; i++) {
            if (!layers[i].inbound || !layers[i].inbound.length) {
                layers[i].inbound = [[[layers[i - 1].name, 0, 0, {}]]];
            }
        }
    }
    const inputs = (cfg.input_layers || []).map((i) => i[0]);
    if (inputs.length === 0) {
        inputs.push(layers[0].name);
    }
    let output = (cfg.output_layers || [])[0];
    output = output ? output[0] : layers[layers.length - 1].name;
    return { layers, inputs, output, nameMap };
}

function computeLayout(graph) {
    // column = longest path from any input
    const col = {};
    const memo = (name) => {
        if (col[name] !== undefined) return col[name];
        const layer = graph.nameMap[name];
        if (graph.inputs.includes(name)) { col[name] = 0; return 0; }
        const preds = (layer.inbound[0] || []).map((e) => e[0]);
        col[name] = preds.length === 0 ? 0 : Math.max(...preds.map(memo)) + 1;
        return col[name];
    };
    graph.layers.forEach((l) => memo(l.name));
    // order within a column by index in layers array (already topological-ish)
    const byCol = {};
    graph.layers.forEach((l) => {
        (byCol[col[l.name]] = byCol[col[l.name]] || []).push(l);
    });
    return { col, byCol };
}

function renderGraph(graph, subModels, title, filename) {
    const layers = graph.layers;
    const { col, byCol } = computeLayout(graph);
    const maxCol = Math.max(...Object.keys(byCol).map(Number));
    const maxRows = Math.max(...Object.values(byCol).map((r) => r.length));
    const W = MARGIN * 2 + maxCol * (BOX_W + COL_GAP);
    const H = MARGIN * 2 + maxRows * (BOX_H + ROW_GAP);

    const pos = {};
    Object.keys(byCol).forEach((c) => {
        byCol[c].forEach((l, r) => {
            pos[l.name] = {
                x: MARGIN + Number(c) * (BOX_W + COL_GAP),
                y: MARGIN + r * (BOX_H + ROW_GAP)
            };
        });
    });

    const parts = [];
    parts.push(`<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" font-family="Segoe UI, sans-serif">`);
    parts.push(`<rect width="${W}" height="${H}" fill="#ffffff"/>`);

    // input dims flowing into each layer
    const inDims = {};
    const computeInputShape = (name) => {
        const l = graph.nameMap[name];
        if (l.class === 'InputLayer') return l.shape ? l.shape[1] : 0;
        if (l.inbound[0] && l.inbound[0].length) {
            const dims = l.inbound[0].map((e) => {
                const src = e[0];
                if (!graph.nameMap[src]) return 0;
                return shapeOf(graph.nameMap[src], []);
            });
            return dims;
        }
        return [0];
    };
    layers.forEach((l) => {
        if (l.inbound[0] && l.inbound[0].length) {
            inDims[l.name] = l.inbound[0].map((e) => shapeOf(graph.nameMap[e[0]], []));
        } else {
            inDims[l.name] = l.shape ? [l.shape[1]] : [0];
        }
    });

    // arrows (edges)
    layers.forEach((l) => {
        if (!(l.inbound[0] && l.inbound[0].length)) return;
        const from = pos[l.name];
        l.inbound[0].forEach((e) => {
            const srcName = e[0];
            const to = pos[srcName];
            if (!from || !to) return;
            const x1 = to.x + BOX_W, y1 = to.y + BOX_H / 2;
            const x2 = from.x, y2 = from.y + BOX_H / 2;
            const mx = (x1 + x2) / 2;
            const d = `M ${x1} ${y1} C ${mx} ${y1}, ${mx} ${y2}, ${x2} ${y2}`;
            parts.push(`<path d="${d}" fill="none" stroke="#b0bec5" stroke-width="1.5" marker-end="url(#arrow)"/>`);
        });
    });

    parts.push(`<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" fill="#90a4ae"/></marker></defs>`);

    // title
    parts.push(`<text x="${MARGIN}" y="28" font-size="16" font-weight="700" fill="#37474f">${esc(title)}</text>`);

    // boxes
    layers.forEach((l) => {
        const p = pos[l.name];
        const isOutput = l.name === graph.output;
        const c = colorFor(l, isOutput);
        const dims = inDims[l.name] || [];
        const inpShape = Array.isArray(dims) ? dims.join(' + ') : dims;
        parts.push(`<rect x="${p.x}" y="${p.y}" width="${BOX_W}" height="${BOX_H}" rx="8" fill="${c.fill}" stroke="${c.stroke}" stroke-width="1.5"/>`);
        parts.push(`<g transform="translate(${p.x + BOX_W / 2}, ${p.y + 18})">${layerLabel(l, inpShape)}</g>`);
    });

    parts.push('</svg>');
    fs.writeFileSync(path.join(OUT, filename), parts.join('\n'));
    console.log('wrote', path.join(OUT, filename));
}

// ---- Build the current model (no training) so the diagram matches the code ----
require('ts-node/register');
require('reflect-metadata');
const { AccuracyModel } = require('../test/specs/AccuracyModel');
const trainedOptions = JSON.parse(fs.readFileSync(
    path.join(__dirname, '..', 'test', 'data', 'model_ens3', 'options.json'), 'utf8'));

const BUILD_DIR = path.join(require('os').tmpdir(), 'openhps-tf-viz');
fs.rmSync(BUILD_DIR, { recursive: true, force: true });

(async () => {
    const model = new AccuracyModel({ accessPoints: trainedOptions.accessPoints });
    await model.save(BUILD_DIR);

    const base = BUILD_DIR;
    const read = (p) => JSON.parse(fs.readFileSync(p, 'utf8'));
    const accTopo = read(path.join(base, 'accuracyModel', 'model.json')).modelTopology;
    const encTopo = read(path.join(base, 'encoder', 'model.json')).modelTopology;
    const decTopo = read(path.join(base, 'decoder', 'model.json')).modelTopology;
    const rssiTopo = read(path.join(base, 'rssiModel', 'model.json')).modelTopology;

    // 1. Accuracy model: composed accuracyModel + expanded encoder.
    const accGraph = parseTopology(accTopo, { encoder: encTopo });
    renderGraph(accGraph, {}, 'Accuracy model — predict kNN error (metres)', 'accuracy-model.svg');

    // 2. Autoencoder: encoder chain followed by decoder chain.
    const autoLayers = [...encTopo.config.layers, ...decTopo.config.layers];
    const autoGraph = parseTopology({
        class_name: 'Sequential',
        config: { name: 'autoencoder', layers: autoLayers }
    }, {});
    autoGraph.inputs = [autoLayers[0].config.name];
    autoGraph.output = autoLayers[autoLayers.length - 1].config.name;
    renderGraph(autoGraph, {}, 'Denoising autoencoder — fingerprint reconstruction', 'autoencoder.svg');

    // 3. Radio map: position -> fingerprint.
    renderGraph(parseTopology(rssiTopo, {}), {}, 'Radio map (rssiModel) — position → fingerprint', 'rssi-model.svg');
})().catch((e) => { console.error(e); process.exit(1); });
