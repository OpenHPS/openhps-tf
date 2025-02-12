import 'mocha';
import * as tf from '../../tf';
import * as fs from 'fs';
import * as csv from 'csv-parser';
import { Absolute2DPosition } from '@openhps/core';
import { expect } from 'chai';

describe('data.openhps', () => {
    let modelWLANFingerprinting: tf.Sequential;
    let modelBLEFingerprinting: tf.Sequential;

    describe('wifi fingerprinting', () => {
        const trainData: any[] = [];
        const testData: any[] = [];
        const accessPoints: string[] = [];
        let train: { xs: tf.Tensor, labels: tf.Tensor };
        let test: { xs: tf.Tensor, labels: tf.Tensor };

        function modelWiFi(): tf.Sequential {
            const model: tf.Sequential = tf.sequential();
            const input = tf.input({ shape: [ accessPoints.length ] });
            const inputLayer = tf.layers.dense({ units: 512, activation: 'relu' });
            inputLayer.apply(input);
            model.add(inputLayer);
            model.add(tf.layers.dense({ units: 128, activation: 'relu' }));
            model.add(tf.layers.dense({ units: 64, activation: 'relu' }));
            model.add(tf.layers.dense({ units: 3, activation: 'linear' }));

            model.compile({
                optimizer: tf.train.adam(0.001),
                loss: tf.losses.huberLoss,
                metrics: ['accuracy']
            });
            return model;
        }

        before((done) => {
            fs.createReadStream('test/data/OpenHPS-2021-05/train/aggregated/wlan_fingerprints.csv')
                .pipe(csv({
                    separator: ","
                })).on('data', row => {
                    trainData.push(row);
                }).on('end', () =>{
                    fs.createReadStream('test/data/OpenHPS-2021-05/test/aggregated/wlan_fingerprints.csv')
                        .pipe(csv({
                            separator: ","
                        })).on('data', row => {
                            testData.push(row);
                        }).on('end', async () =>{
                            try {
                                // Common access points
                                const trainAccessPoints = Object.keys(trainData[0]).filter(col => col.startsWith("WAP_"));
                                const testAccessPoints = Object.keys(testData[0]).filter(col => col.startsWith("WAP_"));
                                trainAccessPoints.forEach(col => {
                                    if (testAccessPoints.includes(col)) {
                                        accessPoints.push(col);
                                    }
                                });
                                testAccessPoints.forEach(col => {
                                    if (!accessPoints.includes(col)) {
                                        accessPoints.push(col);
                                    }
                                });
    
                                train = {
                                    xs: tf.tensor2d(trainData
                                        .map(d => {
                                            const inputs: number[] = [];
                                            accessPoints.forEach(ap => {
                                                const rssi = parseFloat(d[ap]);
                                                inputs.push(Number.isNaN(rssi) ? 100 : rssi);
                                            });
                                            return inputs;
                                        }), [trainData.length, accessPoints.length]),
                                    labels: tf.tensor2d(trainData
                                        .map(d => {
                                            const outputs: number[] = [
                                                parseFloat(d.X),
                                                parseFloat(d.Y),
                                                parseFloat(d.ORIENTATION)
                                            ];
                                            return outputs;
                                        }), [trainData.length, 3])
                                };
                                test = {
                                    xs: tf.tensor2d(testData
                                        .map(d => {
                                            const inputs: number[] = [];
                                            accessPoints.forEach(ap => {
                                                const rssi = parseFloat(d[ap]);
                                                inputs.push(Number.isNaN(rssi) ? 100 : rssi);
                                            });
                                            return inputs;
                                        }), [testData.length, accessPoints.length]),
                                    labels: tf.tensor2d(testData
                                        .map(d => {
                                            const outputs: number[] = [
                                                parseFloat(d.X),
                                                parseFloat(d.Y),
                                                parseFloat(d.ORIENTATION)
                                            ];
                                            return outputs;
                                        }), [testData.length, 3])
                                };
                                modelWLANFingerprinting = modelWiFi();
                                done();
                            } catch (ex) {
                                done(ex);
                            }
                        });
                });
        });

        it('should train the model and evaluate using train data', (done) => {
            let max = 0;
            modelWLANFingerprinting.fit(
                train.xs,
                train.labels, {
                batchSize: 20,
                epochs: 1000,
                verbose: 0,
                shuffle: true,
                validationSplit: .2,
                callbacks: {
                    onEpochEnd: (epoch, logs) => {
                        const accuracy = (logs.acc * 100).toFixed(2);
                        max = Math.max((logs.acc * 100), max);
                        console.log(`Epoch ${epoch + 1}: Current Accuracy: ${accuracy}% | Max Accuracy: ${max.toFixed(2)}%`);
                    },
                }
            }).then(() => {
                return Promise.resolve(modelWLANFingerprinting.evaluate(
                    test.xs, test.labels));
            }).then((info) => {
                const testAccPercent = info[1].dataSync()[0] * 100;
                expect(testAccPercent).to.be.greaterThan(60);
                done();
            }).catch(done);
        }).timeout(-1);

        it('should predict locations from test data', () => {
            let tests = testData.length;
            let error = 0;
            let maxError = 0;
            let minError = Number.MAX_SAFE_INTEGER;
            for (let i = 0 ; i < tests ; i ++) {
                const output = modelWLANFingerprinting.predict(
                    tf.tensor([test.xs.arraySync()[i]])
                ) as tf.Tensor;
                const outputPosition = new Absolute2DPosition(...Array.from(output.dataSync().values()));
                const evaluationPosition = new Absolute2DPosition(...test.labels.arraySync()[i]);
                const distance = outputPosition.distanceTo(evaluationPosition);
                maxError = Math.max(maxError, distance);
                minError = Math.min(minError, distance);
                error += distance;
            }
            const meanError = error / tests;
            console.log("Avg error: " + meanError + "m");
            console.log("Max error: " + maxError + "m");
            console.log("Min error: " + minError + "m");
        });
        
        it('should predict locations from train data', () => {
            let tests = trainData.length;
            let error = 0;
            let maxError = 0;
            let minError = Number.MAX_SAFE_INTEGER;
            for (let i = 0 ; i < tests ; i ++) {
                const output = modelWLANFingerprinting.predict(
                    tf.tensor([train.xs.arraySync()[i]])
                ) as tf.Tensor;
                const outputPosition = new Absolute2DPosition(...Array.from(output.dataSync().values()));
                const evaluationPosition = new Absolute2DPosition(...train.labels.arraySync()[i]);
                const distance = outputPosition.distanceTo(evaluationPosition);
                maxError = Math.max(maxError, distance);
                minError = Math.min(minError, distance);
                error += distance;
            }
            const meanError = error / tests;
            console.log("Avg error: " + meanError + "m");
            console.log("Max error: " + maxError + "m");
            console.log("Min error: " + minError + "m");
        });
    });


   describe('ble fingerprinting', () => {
        const trainData: any[] = [];
        const testData: any[] = [];
        const accessPoints: string[] = [];
        let train: { xs: tf.Tensor, labels: tf.Tensor };
        let test: { xs: tf.Tensor, labels: tf.Tensor };

        function modelBLE(): tf.Sequential {
            const model: tf.Sequential = tf.sequential();
            const input = tf.input({ shape: [ accessPoints.length ] });
            const inputLayer = tf.layers.dense({ units: 32, activation: 'relu' });
            inputLayer.apply(input);
            model.add(inputLayer);
            model.add(tf.layers.dense({ units: 8, activation: 'relu' }));
            model.add(tf.layers.dense({ units: 3, activation: 'linear' }));

            model.compile({
                optimizer: tf.train.adam(0.01),
                loss: tf.losses.huberLoss,
                metrics: ['accuracy']
            });
            return model;
        }

        before((done) => {
            fs.createReadStream('test/data/OpenHPS-2021-05/train/aggregated/ble_fingerprints.csv')
                .pipe(csv({
                    separator: ","
                })).on('data', row => {
                    trainData.push(row);
                }).on('end', () =>{
                    fs.createReadStream('test/data/OpenHPS-2021-05/test/aggregated/ble_fingerprints.csv')
                        .pipe(csv({
                            separator: ","
                        })).on('data', row => {
                            testData.push(row);
                        }).on('end', async () =>{
                            try {
                                // Common access points
                                const trainAccessPoints = Object.keys(trainData[0]).filter(col => col.startsWith("BEACON_"));
                                const testAccessPoints = Object.keys(testData[0]).filter(col => col.startsWith("BEACON_"));
                                trainAccessPoints.forEach(col => {
                                    if (testAccessPoints.includes(col)) {
                                        accessPoints.push(col);
                                    }
                                });
                                testAccessPoints.forEach(col => {
                                    if (accessPoints.includes(col)) {
                                        accessPoints.push(col);
                                    }
                                });
    
                                train = {
                                    xs: tf.tensor2d(trainData
                                        .map(d => {
                                            const inputs: number[] = [];
                                            accessPoints.forEach(ap => {
                                                const rssi = parseFloat(d[ap]);
                                                inputs.push(Number.isNaN(rssi) ? 100 : rssi);
                                            });
                                            return inputs;
                                        }), [trainData.length, accessPoints.length]),
                                    labels: tf.tensor2d(trainData
                                        .map(d => {
                                            const outputs: number[] = [
                                                parseFloat(d.X),
                                                parseFloat(d.Y),
                                                parseFloat(d.ORIENTATION)
                                            ];
                                            return outputs;
                                        }), [trainData.length, 3])
                                };
                                test = {
                                    xs: tf.tensor2d(testData
                                        .map(d => {
                                            const inputs: number[] = [];
                                            accessPoints.forEach(ap => {
                                                const rssi = parseFloat(d[ap]);
                                                inputs.push(Number.isNaN(rssi) ? 100 : rssi);
                                            });
                                            return inputs;
                                        }), [testData.length, accessPoints.length]),
                                    labels: tf.tensor2d(testData
                                        .map(d => {
                                            const outputs: number[] = [
                                                parseFloat(d.X),
                                                parseFloat(d.Y),
                                                parseFloat(d.ORIENTATION)
                                            ];
                                            return outputs;
                                        }), [testData.length, 3])
                                };
                                modelBLEFingerprinting = modelBLE();
                                done();
                            } catch (ex) {
                                done(ex);
                            }
                        });
                });
        });

        it('should train the model and evaluate using train data', (done) => {
            let max = 0;
            modelBLEFingerprinting.fit(
                train.xs,
                train.labels, {
                batchSize: 20,
                epochs: 500,
                verbose: 0,
                shuffle: true,
                validationSplit: .2,
                callbacks: {
                    onEpochEnd: (epoch, logs) => {
                        const accuracy = (logs.acc * 100).toFixed(2);
                        max = Math.max((logs.acc * 100), max);
                        console.log(`Epoch ${epoch + 1}: Current Accuracy: ${accuracy}% | Max Accuracy: ${max.toFixed(2)}%`);
                    },
                }
            }).then(() => {
                return Promise.resolve(modelBLEFingerprinting.evaluate(
                    test.xs, test.labels));
            }).then((info) => {
                const testAccPercent = info[1].dataSync()[0] * 100;
                expect(testAccPercent).to.be.greaterThan(60);
                done();
            }).catch(done);
        }).timeout(-1);

        it('should predict locations', () => {
            let tests = testData.length;
            let error = 0;
            let maxError = 0;
            let minError = Number.MAX_SAFE_INTEGER;
            for (let i = 0 ; i < tests ; i ++) {
                const output = modelBLEFingerprinting.predict(
                    tf.tensor([test.xs.arraySync()[i]])
                ) as tf.Tensor;
                const outputPosition = new Absolute2DPosition(...Array.from(output.dataSync().values()));
                const evaluationPosition = new Absolute2DPosition(...test.labels.arraySync()[i]);
                const distance = outputPosition.distanceTo(evaluationPosition);
                maxError = Math.max(maxError, distance);
                minError = Math.min(minError, distance);
                error += distance;
            }
            const meanError = error / tests;
            console.log("Avg error: " + meanError + "m");
            console.log("Max error: " + maxError + "m");
            console.log("Min error: " + minError + "m");
        });
        
    });
});
