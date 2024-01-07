import * as tf from '../tf';
import * as path from 'path';
import * as fs from 'fs';

export type SignalStrenghts = { [key: `WAP_${number}`]: number };

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

    load(directory: string): Promise<void> {
        return new Promise((resolve, reject) => {
            const encoderPath = path.join(directory, 'encoder');
            const decoderPath = path.join(directory, 'decoder');
            const model1Path = path.join(directory, 'model1');
            const model2Path = path.join(directory, 'model2');
            const autoencoderPath = path.join(directory, 'autoencoder');
            const rssiModelPath = path.join(directory, 'rssiModel');
            if (fs.existsSync(`${encoderPath}/model.json`)) {
                tf.loadLayersModel(`file://${encoderPath}/model.json`)
                    .then((encoder) => {
                        this.encoder = encoder as tf.Sequential;
                        return tf.loadLayersModel(`file://${decoderPath}/model.json`);
                    })
                    .then((decoder) => {
                        this.decoder = decoder as tf.Sequential;
                        return tf.loadLayersModel(`file://${model2Path}/model.json`);
                    })
                    .then((model) => {
                        this.model2 = model as tf.Sequential;
                        return tf.loadLayersModel(`file://${model1Path}/model.json`);
                    })
                    .then((model1) => {
                        this.model1 = model1 as tf.Sequential;
                        this.model1.layers[0].apply(this.encoder.output);
                        return tf.loadLayersModel(`file://${autoencoderPath}/model.json`);
                    })
                    .then((autoencoder) => {
                        this.autoencoder = autoencoder as tf.Sequential;
                        return tf.loadLayersModel(`file://${rssiModelPath}/model.json`);
                    }).then((rssiModel) => {
                        this.rssiModel = rssiModel as tf.Sequential;

                        const input1 = tf.input({ shape: [16 * 16] });
                        const input2 = tf.input({ shape: [4] });
                        this.encoder.layers[0].apply(input1 as tf.SymbolicTensor);

                        const merge = tf.layers.concatenate({ axis: 1, inputShape: [ 4 * 4 * 32 ] });
                        merge.apply([this.model1.output as tf.SymbolicTensor, input2 as tf.SymbolicTensor]);
                        this.accuracyModel = tf.sequential({ layers: [ merge, this.model2 ] });
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
            const model1Path = path.join(directory, 'model1');
            const autoencoderPath = path.join(directory, 'autoencoder');
            const rssiModelPath = path.join(directory, 'rssiModel');
            Promise.all([
                this.encoder.save(`file://${encoderPath}`),
                this.decoder.save(`file://${decoderPath}`),
                this.model2.save(`file://${model2Path}`),
                this.model1.save(`file://${model1Path}`),
                this.autoencoder.save(`file://${autoencoderPath}`),
                this.rssiModel.save(`file://${rssiModelPath}`)
            ]).then(() => {
                resolve();
            }).catch(reject);
        });
    }

    protected compile(): void {
        this.autoencoder.compile({
            optimizer: tf.train.adam(1e-4),
            loss: tf.losses.meanSquaredError,
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
            loss: tf.losses.meanSquaredError,
            metrics: [ 'accuracy' ]
        });
    }

    private _normalizeRSSI(rssi: number[]): number[] {
        const normalized = rssi.map(f => {
            return f === 100 ? 0 : (f + 99) / 99;
        }).concat(Array((16 * 16) - rssi.length).fill(0));
        return normalized;
    }

    protected createModels(): void {
        const autoencoderConfig: tf.ModelCompileArgs = {
            optimizer: tf.train.adam(1e-4),
            loss: tf.losses.meanSquaredError,
            metrics: [ tf.metrics.meanSquaredError]
        };

        // Input dimensions
        const input1 = tf.input({ shape: [ 16 * 16 ] });
        const input2 = tf.input({ shape: [ 4 ] });
        
        // CNN Encoder
        this.encoder = tf.sequential({ name: 'encoder' });
        this.encoder.add(tf.layers.reshape({ 
            targetShape: [ 16, 16, 1 ], 
            inputShape: [ 16 * 16 ]
        }));
        this.encoder.add(tf.layers.conv2d({ activation: 'relu', filters: 32, kernelSize: [3, 3], padding: 'same' }));
        this.encoder.add(tf.layers.conv2d({ activation: 'relu', filters: 64, kernelSize: [3, 3], padding: 'same' }));
        this.encoder.add(tf.layers.maxPooling2d({ poolSize: [2, 2] }));
        this.encoder.add(tf.layers.conv2d({ activation: 'relu', filters: 64, kernelSize: [3, 3], padding: 'same' }));
        this.encoder.add(tf.layers.maxPooling2d({ poolSize: [2, 2] }));
        this.encoder.add(tf.layers.conv2d({ activation: 'relu', filters: 64, kernelSize: [3, 3], padding: 'same' }));
        this.encoder.layers[0].apply(input1 as tf.SymbolicTensor);
        
        // CNN Decoder
        this.decoder = tf.sequential({ name: 'decoder' });
        this.decoder.add(tf.layers.conv2dTranspose({ activation: 'relu', filters: 64, kernelSize: [3, 3], padding: 'same', inputShape: [4, 4, 64] }));
        this.decoder.add(tf.layers.upSampling2d({ size: [2, 2] }));
        this.decoder.add(tf.layers.conv2dTranspose({ activation: 'relu', filters: 64, kernelSize: [3, 3], padding: 'same' }));
        this.decoder.add(tf.layers.upSampling2d({ size: [2, 2] }));
        this.decoder.add(tf.layers.conv2dTranspose({ activation: 'relu', filters: 64, kernelSize: [3, 3], padding: 'same' }));
        this.decoder.add(tf.layers.conv2dTranspose({ activation: 'relu', filters: 32, kernelSize: [3, 3], padding: 'same' }));
        this.decoder.add(tf.layers.flatten());
        this.decoder.add(tf.layers.dense({ activation: 'relu', units: 16 * 16 }));

        // CNN Autoencoder
        this.autoencoder = tf.sequential({ layers: [this.encoder, tf.layers.dropout({ rate: 0.7 }), this.decoder] });
        this.autoencoder.compile(autoencoderConfig);
     
        const model1Input = tf.layers.conv2d({ activation: 'relu', filters: 32, kernelSize: [3, 3], padding: 'same', inputShape: [ 4, 4, 64 ] });
        model1Input.apply(this.encoder.output);
        this.model1 = tf.sequential({ layers: [
            model1Input,
            tf.layers.flatten(),
        ]});

        this.model2 = tf.sequential({
            layers: [
                tf.layers.dense({ units: 512, activation: 'relu', inputShape: [ 4 * 4 * 32 + 4 ] }),
                tf.layers.dense({ units: 128, activation: 'relu' }),
                tf.layers.dropout({ rate: 0.2 }),
                tf.layers.dense({ units: 128, activation: 'relu' }),
                tf.layers.dense({ units: 64, activation: 'relu' }),
                tf.layers.dense({ units: 1, activation: 'relu' })
            ]
        });
        
        const merge = tf.layers.concatenate({ axis: 1, inputShape: [ 4 * 4 * 32 ] });
        merge.apply([this.model1.output as tf.SymbolicTensor, input2 as tf.SymbolicTensor]);
        this.accuracyModel = tf.sequential({ layers: [ merge, this.model2 ] });

        this.rssiModel = tf.sequential({ name: 'rssiModel' });
        this.rssiModel.add(tf.layers.dense({ units: 32, activation: 'relu', inputShape: [ 3 ] }));
        this.rssiModel.add(tf.layers.dense({ units: 64, activation: 'relu' }));
        this.rssiModel.add(tf.layers.dense({ units: 64, activation: 'relu' }));
        this.rssiModel.add(tf.layers.reshape({ targetShape: [ 8, 8, 1 ] }));
        this.rssiModel.add(tf.layers.conv2d({ activation: 'relu', filters: 32, kernelSize: [3, 3], padding: 'same' }));
        this.rssiModel.add(tf.layers.maxPooling2d({ poolSize: [2, 2] }));
        this.rssiModel.add(tf.layers.conv2d({ activation: 'relu', filters: 64, kernelSize: [3, 3], padding: 'same' }));
        this.rssiModel.add(this.decoder);

        this.compile();
    }

    train(data: AccuracyModelData, epochs: number): Promise<void> {
        return new Promise((resolve, reject) => {
            const x = [];
            const y = [];
            // Prepare the data
            data.xs.forEach((input, index) => {
                const fingerprint = {};
                this.options.accessPoints.forEach((ap) => {
                    fingerprint[ap] = input.fingerprint[ap] || 100;
                });
                x.push(this._normalizeRSSI(Object.values(fingerprint)));
                y.push(this._normalizeRSSI(Object.values(fingerprint)));
            });
            // Add random noise (n=0.2)
            data.xs.forEach((input, index) => {
                const fingerprint = {};
                this.options.accessPoints.forEach((ap) => {
                    fingerprint[ap] = input.fingerprint[ap] || 100;
                });
                x.push(this._normalizeRSSI(Object.values(fingerprint)).map(f => f + Math.random() * 0.2));
                y.push(this._normalizeRSSI(Object.values(fingerprint)).map(f => f + Math.random() * 0.2));
            });

            // Enable the training of the encoder
            this.encoder.trainable = true;
            this.decoder.trainable = true;
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
                    x.push(this._normalizeRSSI(Object.values(fingerprint)));
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
                            const loss = logs.loss.toFixed(4);
                            console.log(`CNN | Epoch ${epoch} - Loss: ${loss}`);
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
                    y.push(this._normalizeRSSI(Object.values(fingerprint)));
                });
                return this.rssiModel.fit(tf.tensor(x), tf.tensor(y), {
                    epochs,
                    verbose: 0,
                    batchSize: 32,
                    validationSplit: 0.2,
                    callbacks: {
                        onEpochEnd: (epoch, logs) => {
                            const accuracy = logs.acc.toFixed(4);
                            console.log(`RSSI | Epoch ${epoch} - Accuracy: ${accuracy}`);
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
                tf.tensor([this._normalizeRSSI(Object.values(fingerprint))]), 
                tf.tensor([[input.x, input.y, input.z ?? 0, input.k]])
            ]) as tf.Tensor;
        return prediction.dataSync()[0];
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
}
