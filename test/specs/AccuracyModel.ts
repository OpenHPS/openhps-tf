import * as tf from '../tf';
import * as path from 'path';
import * as fs from 'fs';

export type SignalStrenghts = { [key: `WAP_${number}`]: number };

function rssiLoss(yTrue: tf.Tensor, yPred: tf.Tensor) {
    return tf.tidy(() => {
        const zeros = tf.zerosLike(yTrue);
        const nonZeroIndices = yTrue.notEqual(zeros);
        const zeroIndices = yTrue.equal(zeros);

        const zerosTrue = yTrue.where(zeroIndices, yTrue);
        const nonZerosTrue = yTrue.where(nonZeroIndices, yTrue);
        const zerosPred = yPred.where(zeroIndices, yPred);
        const nonZerosPred = yPred.where(nonZeroIndices, yPred);
        return zerosTrue.sum().sub(zerosPred.sum()).abs().div(zeroIndices.sum()).add(
            nonZerosTrue.sum().sub(nonZerosPred.sum()).abs().square().div(nonZeroIndices.sum()));
    });
}

export class AccuracyModel {
    protected encoder: tf.Sequential;
    protected decoder: tf.Sequential;
    protected autoencoder: tf.Sequential;
    protected model1: tf.Sequential;
    protected model2: tf.Sequential;
    protected accuracyModel: tf.Sequential;     // Model to convert RSSI and X,Y coordinate to accuracy
    protected rssiModel: tf.Sequential;         // Model to convert X,Y coordinate to RSSI
    protected options: AccuracyModelOptions;
    
    constructor(options: AccuracyModelOptions) {
        this.options = options;
        this.createModels();
    }

    private get inputDimension(): number {
        if (this.options.accessPoints.length <= 256) {
            return 16;
        } else if (this.options.accessPoints.length <= 1024) {
            return 32;
        }
    }

    load(directory: string): Promise<void> {
        return new Promise((resolve, reject) => {
            const encoderPath = path.join(directory, 'encoder');
            const decoderPath = path.join(directory, 'decoder');
            const model2Path = path.join(directory, 'model2');
            const model1Path = path.join(directory, 'accuracyModel');
            const rssiModelPath = path.join(directory, 'rssiModel');
            if (fs.existsSync(`${encoderPath}/model.json`)) {
                tf.loadLayersModel(`file://${encoderPath}/model.json`)
                    .then((model) => {
                        this.encoder = model as tf.Sequential;
                        return tf.loadLayersModel(`file://${decoderPath}/model.json`);
                    })
                    .then((model) => {
                        this.decoder = model as tf.Sequential;
                        return tf.loadLayersModel(`file://${model1Path}/model.json`);
                    })
                    .then((model) => {
                        this.model1 = model as tf.Sequential;
                        return tf.loadLayersModel(`file://${rssiModelPath}/model.json`);
                    }).then((model) => {
                        this.rssiModel = model as tf.Sequential;
                        return tf.loadLayersModel(`file://${model2Path}/model.json`);
                    }).then((model) => {
                        this.model2 = model as tf.Sequential;
                        this.options = JSON.parse(fs.readFileSync(path.join(directory, 'options.json'), 'utf-8'));

                        this.autoencoder = tf.sequential({ 
                            layers: [
                                this.encoder, 
                                tf.layers.dropout({ rate: 0.2 }), 
                                this.decoder
                            ] 
                        });

                        const input1 = tf.input({ shape: [this.inputDimension * this.inputDimension] });
                        const input2 = tf.input({ shape: [4] });
                        this.encoder.layers[0].apply(input1 as tf.SymbolicTensor);

                        const modelInput = tf.layers.conv2d({ activation: 'relu', filters: 128, kernelSize: [3, 3], padding: 'same', inputShape: [ 4, 4, 128 ] });
                        modelInput.apply(this.encoder.output);
                        modelInput.setWeights(this.model2.layers[0].getWeights());
                        this.model2 = tf.sequential({
                            layers: [
                                modelInput,
                                ...this.model2.layers.slice(1)
                            ]
                        });
                        
                        const merge = tf.layers.concatenate({ axis: 1 });
                        merge.apply([this.model2.output as tf.SymbolicTensor, input2 as tf.SymbolicTensor]);
                        this.accuracyModel = tf.sequential({ layers: [ merge, this.model1 ] });
                        
                        this.compile();
                        resolve();
                    })
                    .catch(reject);
            } else {
                resolve();
            }
        });
    }

    save(directory: string): Promise<void> {
        return new Promise((resolve, reject) => {
            const encoderPath = path.join(directory, 'encoder');
            const decoderPath = path.join(directory, 'decoder');
            const model2Path = path.join(directory, 'model2');
            const model1Path = path.join(directory, 'accuracyModel');
            const rssiModelPath = path.join(directory, 'rssiModel');
            Promise.all([
                this.encoder.save(`file://${encoderPath}`),
                this.decoder.save(`file://${decoderPath}`),
                this.model1.save(`file://${model1Path}`),
                this.rssiModel.save(`file://${rssiModelPath}`),
                this.model2.save(`file://${model2Path}`)
            ]).then(() => {
                fs.writeFileSync(path.join(directory, 'options.json'), JSON.stringify(this.options));
                resolve();
            }).catch(reject);
        });
    }

    protected compile(): void {
        this.autoencoder.compile({
            optimizer: tf.train.adam(1e-4),
            loss: rssiLoss,
            metrics: [ tf.metrics.meanSquaredError]
        });
        this.accuracyModel.compile({
            optimizer: tf.train.adam(1e-4),
            loss: tf.losses.meanSquaredError,
            metrics: [((yTrue, yPred) => {
                return yTrue.sub(yPred).abs().mean();
            })]
        });
        this.rssiModel.compile({
            optimizer: tf.train.adam(1e-4),
            loss: rssiLoss,
            metrics: [ 'accuracy' ]
        });
    }

    private _normalizeRSSI(rssi: number[], min: number, max: number): number[] {
        const normalized = rssi.map(f => {
            return f === 100 ? 0 : (f + (-max)) / (99 + min);
        }).concat(Array((Math.pow(this.inputDimension, 2)) - rssi.length).fill(0));
        return normalized;
    }

    protected createModels(): void {
        // Input dimensions
        const input2 = tf.input({ shape: [ 4 ] });
        const inputSize = this.inputDimension;

        // CNN Encoder
        this.encoder = tf.sequential({ name: 'encoder' });
        this.encoder.add(tf.layers.reshape({ 
            targetShape: [ inputSize, inputSize, 1 ], 
            inputShape: [ inputSize * inputSize ]
        }));
        this.encoder.add(tf.layers.conv2d({ activation: 'relu', filters: 16, kernelSize: [3, 3], padding: 'same' }));
        this.encoder.add(tf.layers.maxPooling2d({ poolSize: [2, 2] }));
        this.encoder.add(tf.layers.conv2d({ activation: 'relu', filters: 32, kernelSize: [3, 3], padding: 'same' }));
        this.encoder.add(tf.layers.maxPooling2d({ poolSize: [2, 2] }));
        this.encoder.add(tf.layers.conv2d({ activation: 'relu', filters: 64, kernelSize: [3, 3], padding: 'same' }));
        this.encoder.add(tf.layers.conv2d({ activation: 'relu', filters: 128, kernelSize: [3, 3], padding: 'same' }));
        if (inputSize === 32) {
            this.encoder.add(tf.layers.maxPooling2d({ poolSize: [2, 2] }));
            this.encoder.add(tf.layers.conv2d({ activation: 'relu', filters: 128, kernelSize: [3, 3], padding: 'same' }));    
        }

        // CNN Decoder
        this.decoder = tf.sequential({ name: 'decoder' });
        this.decoder.add(tf.layers.conv2dTranspose({ activation: 'relu', filters: 128, kernelSize: [3, 3], padding: 'same', inputShape: [ 4, 4, 128 ] }));
        if (inputSize === 32) {
            this.decoder.add(tf.layers.upSampling2d({ size: [2, 2] }));
            this.decoder.add(tf.layers.conv2dTranspose({ activation: 'relu', filters: 128, kernelSize: [3, 3], padding: 'same' }));
        }
        this.decoder.add(tf.layers.conv2dTranspose({ activation: 'relu', filters: 64, kernelSize: [3, 3], padding: 'same' }));
        this.decoder.add(tf.layers.upSampling2d({ size: [2, 2] }));
        this.decoder.add(tf.layers.conv2dTranspose({ activation: 'relu', filters: 32, kernelSize: [3, 3], padding: 'same' }));
        this.decoder.add(tf.layers.upSampling2d({ size: [2, 2] }));
        this.decoder.add(tf.layers.conv2dTranspose({ activation: 'relu', filters: 16, kernelSize: [3, 3], padding: 'same' }));
        this.decoder.add(tf.layers.flatten());
        this.decoder.add(tf.layers.dense({ activation: 'relu', units: inputSize * inputSize }));

        // CNN Autoencoder
        this.autoencoder = tf.sequential({ 
            layers: [
                this.encoder, 
                tf.layers.dropout({ rate: 0.4 }), 
                this.decoder
            ] 
        });

        this.model1 = tf.sequential({
            layers: [
                tf.layers.dense({ units: 512, activation: 'relu', inputShape: [ 256 + 4 ] }),
                tf.layers.dense({ units: 256, activation: 'relu' }),
                tf.layers.dropout({ rate: 0.2 }),
                tf.layers.dense({ units: 128, activation: 'relu' }),
                tf.layers.dense({ units: 64, activation: 'relu' }),
                tf.layers.dense({ units: 32, activation: 'relu' }),
                tf.layers.dense({ units: 1, activation: 'linear' })
            ]
        });

        const modelInput = tf.layers.conv2d({ activation: 'relu', filters: 128, kernelSize: [3, 3], padding: 'same', inputShape: [ 4, 4, 128 ] });
        modelInput.apply(this.encoder.output);
        this.model2 = tf.sequential({
            layers: [
                modelInput,
                tf.layers.avgPooling2d({ poolSize: [2, 2] }),
                tf.layers.conv2d({ activation: 'relu', filters: 256, kernelSize: [3, 3], padding: 'same' }),
                tf.layers.avgPooling2d({ poolSize: [2, 2] }),
                tf.layers.conv2d({ activation: 'relu', filters: 256, kernelSize: [3, 3], padding: 'same' }),
                tf.layers.flatten(),
            ]
        });
        
        const merge = tf.layers.concatenate({ axis: 1 });
        merge.apply([this.model2.output as tf.SymbolicTensor, input2 as tf.SymbolicTensor]);
        this.accuracyModel = tf.sequential({ layers: [ merge, this.model1 ] });

        this.rssiModel = tf.sequential({ name: 'rssiModel' });
        this.rssiModel.add(tf.layers.dense({ units: 16, activation: 'relu', inputShape: [ 3 ] }));
        this.rssiModel.add(tf.layers.reshape({ targetShape: [ 1, 1, 16 ] }));
        this.rssiModel.add(tf.layers.conv2dTranspose({ activation: 'relu', filters: 16, kernelSize: [3, 3], padding: 'same' }));
        this.rssiModel.add(tf.layers.upSampling2d({ size: [2, 2] }));
        this.rssiModel.add(tf.layers.conv2dTranspose({ activation: 'relu', filters: 32, kernelSize: [3, 3], padding: 'same' }));
        this.rssiModel.add(tf.layers.upSampling2d({ size: [2, 2] }));
        this.rssiModel.add(tf.layers.conv2dTranspose({ activation: 'relu', filters: 64, kernelSize: [3, 3], padding: 'same' }));
        this.rssiModel.add(tf.layers.conv2dTranspose({ activation: 'relu', filters: 128, kernelSize: [3, 3], padding: 'same' }));
        this.rssiModel.add(this.decoder);
        this.compile();
    }

    train(data: AccuracyModelData, epochs: number): Promise<void> {
        return new Promise((resolve, reject) => {
            const x = [];
            const y = [];
            // Prepare the data
            let min = -99;
            let max = 0;
            data.xs.forEach((input) => {
                Object.values(input.fingerprint).forEach((rssi) => {
                    if (rssi !== 100) {
                        min = Math.max(min, rssi);
                        max = Math.min(max, rssi);
                    }
                });
            });
            this.options.minRSSI = min;
            this.options.maxRSSI = max;
            
            const locations: Array<{ x: number, y: number, z: number }> = [];

            // Add the raw data
            data.xs.forEach((input) => {
                // Add the location if it doesn't exist
                if (!locations.find(l => l.x === input.x && l.y === input.y && l.z === input.z)) {
                    locations.push({ x: input.x, y: input.y, z: input.z });
                }

                const fingerprint = {};
                this.options.accessPoints.forEach((ap) => {
                    fingerprint[ap] = input.fingerprint[ap] || 100;
                });
                x.push(this._normalizeRSSI(Object.values(fingerprint), min, max));
                y.push(this._normalizeRSSI(Object.values(fingerprint), min, max));
            });

            // Add the aggregated data on the same X, Y, Z coordinate
            const aggregatedFingerprints: Array<{ fingerprint: SignalStrenghts }> = [];
            locations.forEach(location => {
                const locationFingerprints: Array<{ fingerprint: SignalStrenghts }> = data.xs.filter(f => f.x === location.x && f.y === location.y && f.z === location.z);
                aggregatedFingerprints.push({
                    fingerprint: locationFingerprints.reduce((a, b) => {
                        const rssi: SignalStrenghts = {};
                        this.options.accessPoints.forEach(col => {
                            if (a[col]) {
                                rssi[col] = (a[col] + b.fingerprint[col]) / 2;
                            } else {
                                rssi[col] = b.fingerprint[col];
                            }
                        });
                        return rssi;
                    }, {})
                });
            });
            aggregatedFingerprints.forEach((input) => {
                const fingerprint = {};
                this.options.accessPoints.forEach((ap) => {
                    fingerprint[ap] = input.fingerprint[ap] || 100;
                });
                x.push(this._normalizeRSSI(Object.values(fingerprint), min, max));
                y.push(this._normalizeRSSI(Object.values(fingerprint), min, max));
            });

            // Add random noise to aggregated fingerprints (n=0.2)
            aggregatedFingerprints.forEach((input) => {
                const fingerprint = {};
                this.options.accessPoints.forEach((ap) => {
                    fingerprint[ap] = input.fingerprint[ap] || 100;
                });
                x.push(this._normalizeRSSI(Object.values(fingerprint), min, max).map(f => f + Math.random() * 0.2));
                y.push(this._normalizeRSSI(Object.values(fingerprint), min, max));
            });

            // Enable the training of the encoder
            this.encoder.trainable = true;
            this.decoder.trainable = true;
            this.encoder.layers.forEach((layer) => {
                layer.trainable = true;
            });
            this.decoder.layers.forEach((layer) => {
                layer.trainable = true;
            });
            this.compile();
            this.autoencoder.fit(tf.tensor(x), tf.tensor(y), {
                epochs,
                verbose: 0,
                batchSize: 32,
                validationSplit: 0.2,
                callbacks: {
                    onEpochEnd: (epoch, logs) => {
                        const loss = logs.loss.toFixed(4);
                        console.log(`CNNAE | Epoch ${epoch} - Loss: ${loss}`);
                    }
                }
            }).then(() => {
                // Disable the training of the encoder
                this.encoder.trainable = false;
                this.decoder.trainable = false;
                this.encoder.layers.forEach((layer) => {
                    layer.trainable = false;
                });
                this.decoder.layers.forEach((layer) => {
                    layer.trainable = false;
                });
                this.compile();
                // Start training the model
                const x = [];
                const x2 = [];
                const y = [];
                data.xs.forEach((input, index) => {
                    const fingerprint = {};
                    this.options.accessPoints.forEach((ap) => {
                        fingerprint[ap] = input.fingerprint[ap] || 100;
                    });
                    x.push(this._normalizeRSSI(Object.values(fingerprint), min, max));
                    x2.push([input.x, input.y, input.z ?? 0, input.k]);
                    y.push(data.ys[index]);
                });
                return this.accuracyModel.fit([tf.tensor(x), tf.tensor(x2)], tf.tensor(y), {
                    epochs,
                    verbose: 0,
                    batchSize: 32,
                    validationSplit: 0.2,
                    callbacks: {
                        onEpochEnd: (epoch, logs) => {
                            const loss = logs.val_.toFixed(4);
                            console.log(`CNN | Epoch ${epoch} - Accuracy: ${loss}m - Loss: ${logs.loss.toFixed(4)}`);
                        }
                    }
                });
            }).then(() => {
                const x = [];
                const y = [];
                data.xs.forEach((input) => {
                    const fingerprint = {};
                    this.options.accessPoints.forEach((ap) => {
                        fingerprint[ap] = input.fingerprint[ap] || 100;
                    });
                    x.push([input.x, input.y, input.z ?? 0]);
                    y.push(this._normalizeRSSI(Object.values(fingerprint), min, max));
                });
                return this.rssiModel.fit(tf.tensor(x), tf.tensor(y), {
                    epochs,
                    verbose: 0,
                    batchSize: 32,
                    validationSplit: 0.2,
                    callbacks: {
                        onEpochEnd: (epoch, logs) => {
                            const loss = logs.loss.toFixed(4);
                            console.log(`RSSI | Epoch ${epoch} - Loss: ${loss}`);
                        }
                    }
                });
            }).then(() => {
                resolve();
            }).catch(reject);
        });
    }

    predict(input: AccuracyModelInput): number {
        const fingerprint = {};
        this.options.accessPoints.forEach((ap) => {
            fingerprint[ap] = input.fingerprint[ap] || 100;
        });
        const prediction = this.accuracyModel.predict(
            [
                tf.tensor([this._normalizeRSSI(Object.values(fingerprint), this.options.minRSSI, this.options.maxRSSI)]), 
                tf.tensor([[input.x, input.y, input.z ?? 0, input.k]])
            ]) as tf.Tensor;
        return prediction.dataSync()[0];
    }

    predictRSSI(x: number, y: number, z: number): SignalStrenghts {
        const signalStrengths: SignalStrenghts = {};
        const prediction = this.rssiModel.predict(
            tf.tensor([[x, y, z]])
        ) as tf.Tensor;
        const predictionData = Array.from(prediction.dataSync());
        const normalizedData = predictionData.map(f => {
            return Math.round(f === 0 ? 100 : (f * 99) - 99);
        });
        this.options.accessPoints.forEach((ap ,i) => {
            const rssi = normalizedData[i];
            signalStrengths[ap] = rssi;
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
