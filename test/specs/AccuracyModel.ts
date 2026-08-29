import * as tf from '../tf';
import * as path from 'path';
import * as fs from 'fs';

export type SignalStrenghts = { [key: `WAP_${number}`]: number };

const ARCHITECTURE = 'dense-v2';

/**
 * Loss used to train the fingerprint autoencoder and the radio map model.
 *
 * A fingerprint is represented as [normalizedRSSI(1..N) | seenMask(1..N)].
 * The RSSI part is only penalised for access points that were actually
 * observed (mask === 1), so the model is never rewarded for reproducing the
 * "missing" (100) value that dominates raw fingerprints. The mask part is
 * learned with binary cross entropy.
 */
function fingerprintLoss(N: number): (yTrue: tf.Tensor, yPred: tf.Tensor) => tf.Scalar {
    return (yTrue: tf.Tensor, yPred: tf.Tensor): tf.Scalar => tf.tidy(() => {
        const rssiTrue = yTrue.slice([0, 0], [-1, N]);
        const maskTrue = yTrue.slice([0, N], [-1, N]);
        const rssiPred = yPred.slice([0, 0], [-1, N]);
        const maskPred = yPred.slice([0, N], [-1, N]);
        const count = maskTrue.sum().add(1e-6);
        const rssiError = rssiTrue.sub(rssiPred).square().mul(maskTrue).sum().div(count);
        const maskError = tf.losses.sigmoidCrossEntropy(maskTrue, maskPred).mean();
        return rssiError.add(maskError.mul(0.25));
    });
}

export class AccuracyModel {
    protected encoder: tf.Sequential;
    protected decoder: tf.Sequential;
    protected autoencoder: tf.LayersModel;
    protected accuracyModel: tf.LayersModel;
    protected rssiModel: tf.Sequential;
    protected options: AccuracyModelOptions;

    constructor(options: AccuracyModelOptions) {
        this.options = options;
        this.createModels();
    }

    /**
     * Number of access points in the fingerprint space. Every fingerprint is
     * represented as a vector of `2 * N` values: normalized RSSI followed by a
     * seen/missing mask.
     */
    private get N(): number {
        return this.options.accessPoints.length;
    }

    /**
     * Training batch size. Larger batches use the GPU much more efficiently
     * (the model is tiny, so small batches are latency-bound). BATCH_SIZE env
     * overrides, default 256.
     */
    private get batchSize(): number {
        const v = process.env.BATCH_SIZE;
        return v !== undefined && v !== '' ? parseInt(v, 10) : 256;
    }

    /**
     * Early-stopping patience in epochs. Stops a sub-model when its validation
     * loss has not improved for this many epochs. 0 disables early stopping.
     * PATIENCE env overrides, default 20.
     */
    private get patience(): number {
        const v = process.env.PATIENCE;
        return v !== undefined && v !== '' ? parseInt(v, 10) : 20;
    }

    /**
     * Fit callbacks: optional early stopping (monitor val_loss) + a logger.
     * Both are real callback instances so tfjs's CallbackList can handle them.
     */
    private fitCallbacks(logFn: (epoch: number, logs: tf.Logs) => void): any[] {
        const callbacks: any[] = [];
        if (this.patience > 0) {
            callbacks.push(tf.callbacks.earlyStopping({
                monitor: 'val_loss',
                minDelta: 1e-4,
                patience: this.patience,
                mode: 'min',
                verbose: 1
            }));
        }
        callbacks.push(new tf.CustomCallback({ onEpochEnd: (epoch, logs) => logFn(epoch, logs) }));
        return callbacks;
    }

    load(directory: string): Promise<void> {
        return new Promise((resolve, reject) => {
            const optionsPath = path.join(directory, 'options.json');
            if (!fs.existsSync(optionsPath) ||
                JSON.parse(fs.readFileSync(optionsPath, 'utf-8')).architecture !== ARCHITECTURE) {
                // No compatible (dense-v2) model present - start from scratch.
                resolve();
                return;
            }
            const encoderPath = path.join(directory, 'encoder');
            const decoderPath = path.join(directory, 'decoder');
            const accuracyPath = path.join(directory, 'accuracyModel');
            const rssiPath = path.join(directory, 'rssiModel');
            tf.loadLayersModel(`file://${path.join(encoderPath, 'model.json')}`)
                .then((model) => {
                    this.encoder = model as tf.Sequential;
                    return tf.loadLayersModel(`file://${path.join(decoderPath, 'model.json')}`);
                })
                .then((model) => {
                    this.decoder = model as tf.Sequential;
                    return tf.loadLayersModel(`file://${path.join(accuracyPath, 'model.json')}`);
                })
                .then((model) => {
                    this.accuracyModel = model;
                    return tf.loadLayersModel(`file://${path.join(rssiPath, 'model.json')}`);
                })
                .then((model) => {
                    this.rssiModel = model as tf.Sequential;
                    this.options = Object.assign(this.options,
                        JSON.parse(fs.readFileSync(optionsPath, 'utf-8')));
                    this.rebuildAutoencoder();
                    this.compile();
                    resolve();
                })
                .catch(reject);
        });
    }

    save(directory: string): Promise<void> {
        return new Promise((resolve, reject) => {
            fs.mkdirSync(directory, { recursive: true });
            const encoderPath = path.join(directory, 'encoder');
            const decoderPath = path.join(directory, 'decoder');
            const accuracyPath = path.join(directory, 'accuracyModel');
            const rssiPath = path.join(directory, 'rssiModel');
            Promise.all([
                this.encoder.save(`file://${encoderPath}`),
                this.decoder.save(`file://${decoderPath}`),
                this.accuracyModel.save(`file://${accuracyPath}`),
                this.rssiModel.save(`file://${rssiPath}`)
            ]).then(() => {
                fs.writeFileSync(path.join(directory, 'options.json'),
                    JSON.stringify(Object.assign({}, this.options, { architecture: ARCHITECTURE })));
                resolve();
            }).catch(reject);
        });
    }

    protected compile(): void {
        this.autoencoder.compile({
            optimizer: tf.train.adam(1e-4),
            loss: fingerprintLoss(this.N),
            metrics: ['mse']
        });
        this.accuracyModel.compile({
            optimizer: tf.train.adam(1e-4),
            loss: tf.losses.huberLoss,
            metrics: ['mae']
        });
        this.rssiModel.compile({
            optimizer: tf.train.adam(1e-4),
            loss: fingerprintLoss(this.N),
            metrics: ['mse']
        });
    }

    /**
     * Build a fingerprint vector [normalizedRSSI | seenMask]. Normalized RSSI
     * maps the weakest observed signal to 0 and the strongest to 1; missing
     * (100 / undefined) access points are encoded as 0 with a 0 mask.
     */
    private _toFingerprintVector(fingerprint: SignalStrenghts): number[] {
        const N = this.N;
        const rssi = new Array(N).fill(0);
        const mask = new Array(N).fill(0);
        const strongest = this.options.minRSSI ?? -99;
        const weakest = this.options.maxRSSI ?? -96;
        this.options.accessPoints.forEach((ap, i) => {
            const value = fingerprint[ap];
            if (value !== undefined && value !== 100 && !Number.isNaN(value)) {
                const span = strongest - weakest;
                rssi[i] = span === 0 ? 0 : Math.min(Math.max((value - weakest) / span, 0), 1);
                mask[i] = 1;
            }
        });
        return [...rssi, ...mask];
    }

    /**
     * Fingerprint statistics that strongly correlate with kNN error: the
     * fraction of observed access points and the mean/std of observed RSSI.
     */
    private _fingerprintStats(fingerprint: SignalStrenghts): number[] {
        const seen: number[] = [];
        this.options.accessPoints.forEach((ap) => {
            const value = fingerprint[ap];
            if (value !== undefined && value !== 100 && !Number.isNaN(value)) {
                seen.push(value);
            }
        });
        if (seen.length === 0) {
            return [0, 0, 0];
        }
        const strongest = this.options.minRSSI ?? -99;
        const weakest = this.options.maxRSSI ?? -96;
        const span = strongest - weakest;
        const normalized = seen.map(v => span === 0 ? 0 : Math.min(Math.max((v - weakest) / span, 0), 1));
        const mean = normalized.reduce((a, b) => a + b, 0) / normalized.length;
        const variance = normalized.reduce((a, b) => a + (b - mean) * (b - mean), 0) / normalized.length;
        return [seen.length / this.N, mean, Math.sqrt(variance)];
    }

    /**
     * Randomly mask observed access points (denoising augmentation).
     */
    private _maskFingerprint(vec: number[]): number[] {
        const N = this.N;
        const out = vec.slice();
        for (let i = 0; i < N; i++) {
            if (out[N + i] === 1 && Math.random() < 0.3) {
                out[i] = 0;
                out[N + i] = 0;
            }
        }
        return out;
    }

    /**
     * Split samples into train/validation groups by location so that no
     * location appears in both sets (a random split would leak the same
     * fingerprints across train and validation).
     */
    private splitByLocation(data: AccuracyModelData, ratio = 0.8): { trainIdx: number[], valIdx: number[] } {
        const groups = new Map<string, number[]>();
        data.xs.forEach((input, index) => {
            const key = `${input.x},${input.y},${input.z ?? 0}`;
            const group = groups.get(key) ?? [];
            group.push(index);
            groups.set(key, group);
        });
        const keys = Array.from(groups.keys()).sort(() => Math.random() - 0.5);
        const splitAt = Math.max(1, Math.floor(keys.length * ratio));
        const trainIdx: number[] = [];
        const valIdx: number[] = [];
        keys.forEach((key, i) => {
            (i < splitAt ? trainIdx : valIdx).push(...groups.get(key));
        });
        return { trainIdx, valIdx };
    }

    protected createModels(): void {
        const N = this.N;

        // Dense denoising autoencoder (fingerprints have no spatial ordering,
        // so a 2D convolution over AP index order is not meaningful).
        this.encoder = tf.sequential({ name: 'encoder' });
        this.encoder.add(tf.layers.dense({ units: 256, activation: 'relu', inputShape: [N * 2] }));
        this.encoder.add(tf.layers.dense({ units: 128, activation: 'relu' }));
        this.encoder.add(tf.layers.dense({ units: 64, activation: 'relu' }));

        this.decoder = tf.sequential({ name: 'decoder' });
        this.decoder.add(tf.layers.dense({ units: 128, activation: 'relu', inputShape: [64] }));
        this.decoder.add(tf.layers.dense({ units: 256, activation: 'relu' }));
        this.decoder.add(tf.layers.dense({ units: N * 2, activation: 'sigmoid' }));

        this.rebuildAutoencoder();

        // Accuracy model: [fingerprint | stats | position + k] -> error in metres.
        // The fingerprint is embedded by the (frozen) autoencoder bottleneck.
        const fpIn = tf.input({ shape: [N * 2] });
        const statsIn = tf.input({ shape: [3] });
        const posIn = tf.input({ shape: [4] });
        const bn = this.encoder.apply(fpIn) as tf.SymbolicTensor;
        let merged = tf.layers.concatenate().apply([bn, statsIn, posIn]) as tf.SymbolicTensor;
        merged = tf.layers.dense({ units: 128, activation: 'relu' }).apply(merged) as tf.SymbolicTensor;
        merged = tf.layers.dropout({ rate: 0.2 }).apply(merged) as tf.SymbolicTensor;
        merged = tf.layers.dense({ units: 64, activation: 'relu' }).apply(merged) as tf.SymbolicTensor;
        merged = tf.layers.dense({ units: 32, activation: 'relu' }).apply(merged) as tf.SymbolicTensor;
        const accOut = tf.layers.dense({ units: 1, activation: 'linear' }).apply(merged) as tf.SymbolicTensor;
        this.accuracyModel = tf.model({ inputs: [fpIn, statsIn, posIn], outputs: accOut });

        // Radio map: position (x, y, z) -> fingerprint [normalizedRSSI | seenMask].
        this.rssiModel = tf.sequential({ name: 'rssiModel' });
        this.rssiModel.add(tf.layers.dense({ units: 128, activation: 'relu', inputShape: [3] }));
        this.rssiModel.add(tf.layers.dense({ units: 256, activation: 'relu' }));
        this.rssiModel.add(tf.layers.dense({ units: N * 2, activation: 'sigmoid' }));

        this.compile();
    }

    private rebuildAutoencoder(): void {
        const autoIn = tf.input({ shape: [this.N * 2] });
        const bottleneck = this.encoder.apply(autoIn) as tf.SymbolicTensor;
        const autoOut = this.decoder.apply(bottleneck) as tf.SymbolicTensor;
        this.autoencoder = tf.model({ inputs: autoIn, outputs: autoOut });
    }

    train(data: AccuracyModelData, epochs: number): Promise<void> {
        return new Promise<void>((resolve, reject) => {
            this.trainAsync(data, epochs).then(resolve).catch(reject);
        });
    }

    private async trainAsync(data: AccuracyModelData, epochs: number): Promise<void> {
        // Determine RSSI bounds from the training fingerprints.
        let strongest = -99;
        let weakest = 0;
        data.xs.forEach((input) => {
            Object.values(input.fingerprint).forEach((rssi) => {
                if (rssi !== 100) {
                    strongest = Math.max(strongest, rssi);
                    weakest = Math.min(weakest, rssi);
                }
            });
        });
        this.options.minRSSI = strongest;
        this.options.maxRSSI = weakest;
        const N = this.N;

        // Build feature vectors.
        const fp = data.xs.map((input) => this._toFingerprintVector(input.fingerprint));
        const stats = data.xs.map((input) => this._fingerprintStats(input.fingerprint));
        const pos4 = data.xs.map((input) => [input.x, input.y, input.z ?? 0, input.k]);
        const pos3 = data.xs.map((input) => [input.x, input.y, input.z ?? 0]);

        // Split by location.
        const { trainIdx, valIdx } = this.splitByLocation(data);
        const pick = (arr: number[][], idx: number[]) => idx.map(i => arr[i]);
        const trainFp = tf.tensor2d(pick(fp, trainIdx), [trainIdx.length, N * 2]);
        const valFp = tf.tensor2d(pick(fp, valIdx), [valIdx.length, N * 2]);
        const trainStats = tf.tensor2d(pick(stats, trainIdx), [trainIdx.length, 3]);
        const valStats = tf.tensor2d(pick(stats, valIdx), [valIdx.length, 3]);
        const trainPos4 = tf.tensor2d(pick(pos4, trainIdx), [trainIdx.length, 4]);
        const valPos4 = tf.tensor2d(pick(pos4, valIdx), [valIdx.length, 4]);
        const trainPos3 = tf.tensor2d(pick(pos3, trainIdx), [trainIdx.length, 3]);
        const valPos3 = tf.tensor2d(pick(pos3, valIdx), [valIdx.length, 3]);
        const trainY = tf.tensor1d(trainIdx.map(i => data.ys[i]));
        const valY = tf.tensor1d(valIdx.map(i => data.ys[i]));

        // 1. Denoising autoencoder (with masked augmentation on the train split).
        this.encoder.trainable = true;
        this.decoder.trainable = true;
        this.compile();
        const autoTrain = [...pick(fp, trainIdx), ...pick(fp, trainIdx).map(v => this._maskFingerprint(v))];
        const autoTrainTensor = tf.tensor2d(autoTrain, [autoTrain.length, N * 2]);
        await this.autoencoder.fit(autoTrainTensor, autoTrainTensor, {
            epochs,
            batchSize: this.batchSize,
            verbose: 0,
            validationData: [valFp, valFp],
            callbacks: this.fitCallbacks((epoch, logs) => {
                console.log(`AE | Epoch ${epoch + 1}/${epochs} - Loss: ${logs.loss.toFixed(4)} - Val: ${(logs.val_loss ?? 0).toFixed(4)}`);
            })
        });

        // 2. Accuracy model. The fingerprint encoder is frozen so the head is
        //    trained on the reconstruction-optimized bottleneck.
        this.encoder.trainable = false;
        this.decoder.trainable = false;
        this.compile();
        await this.accuracyModel.fit([trainFp, trainStats, trainPos4], trainY, {
            epochs,
            batchSize: this.batchSize,
            verbose: 0,
            validationData: [[valFp, valStats, valPos4], valY],
            callbacks: this.fitCallbacks((epoch, logs) => {
                console.log(`ACC | Epoch ${epoch + 1}/${epochs} - Loss: ${logs.loss.toFixed(4)} - MAE: ${(logs.mae ?? 0).toFixed(4)}m - Val MAE: ${(logs.val_mae ?? 0).toFixed(4)}m`);
            })
        });

        // 3. Radio map: position -> fingerprint.
        await this.rssiModel.fit(trainPos3, trainFp, {
            epochs,
            batchSize: this.batchSize,
            verbose: 0,
            validationData: [valPos3, valFp],
            callbacks: this.fitCallbacks((epoch, logs) => {
                console.log(`RSSI | Epoch ${epoch + 1}/${epochs} - Loss: ${logs.loss.toFixed(4)} - Val: ${(logs.val_loss ?? 0).toFixed(4)}`);
            })
        });
    }

    async predict(input: AccuracyModelInput): Promise<number> {
        const vec = this._toFingerprintVector(input.fingerprint);
        const stats = this._fingerprintStats(input.fingerprint);
        const fpTensor = tf.tensor2d([vec], [1, this.N * 2]);
        const statsTensor = tf.tensor2d([stats], [1, 3]);
        const posTensor = tf.tensor2d([[input.x, input.y, input.z ?? 0, input.k]], [1, 4]);
        const output = this.accuracyModel.predict([fpTensor, statsTensor, posTensor]) as tf.Tensor;
        const values = await output.array() as number[][];
        fpTensor.dispose();
        statsTensor.dispose();
        posTensor.dispose();
        output.dispose();
        return values[0][0];
    }

    async predictRSSI(x: number, y: number, z: number): Promise<SignalStrenghts> {
        const posTensor = tf.tensor2d([[x, y, z]], [1, 3]);
        const out = this.rssiModel.predict(posTensor) as tf.Tensor;
        const prediction = await out.array() as number[][];
        posTensor.dispose();
        out.dispose();
        const signalStrengths: SignalStrenghts = {};
        const N = this.N;
        const strongest = this.options.minRSSI ?? -99;
        const weakest = this.options.maxRSSI ?? -96;
        prediction[0].forEach((normalized, i) => {
            const seen = prediction[0][N + i] >= 0.5;
            signalStrengths[this.options.accessPoints[i]] = seen
                ? Math.round(weakest + normalized * (strongest - weakest))
                : 100;
        });
        return signalStrengths;
    }
}

export interface AccuracyModelInput {
    fingerprint: SignalStrenghts;
    k: number;
    x: number;
    y: number;
    z?: number;
}

export interface AccuracyModelData {
    xs: AccuracyModelInput[]
    ys: number[];
}

export interface AccuracyModelOptions {
    accessPoints: string[];
    minRSSI?: number;
    maxRSSI?: number;
}
