import 'mocha';
import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-node';
import * as csv from 'csv-parser';
import * as fs from 'fs';

describe('data.openhps.fingerprinting', () => {
    const trainData: any[] = [];
    const testData: any[] = [];
    const accessPoints: string[] = [];

    before((done) => {
        fs.createReadStream('test/data/OpenHPS-2021-05/train/raw/wlan_fingerprints.csv')
            .pipe(csv({
                separator: ","
            })).on('data', row => {
                trainData.push(row);
            }).on('end', () =>{
                fs.createReadStream('test/data/OpenHPS-2021-05/test/raw/wlan_fingerprints.csv')
                    .pipe(csv({
                        separator: ","
                    })).on('data', row => {
                        testData.push(row);
                    }).on('end', () =>{
                        // Common access points
                        const trainAccessPoints = Object.keys(trainData[0]).filter(col => col.startsWith("WAP_"));
                        const testAccessPoints = Object.keys(testData[0]).filter(col => col.startsWith("WAP_"));
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
                        done();
                    });
            });
    });

    describe('training', () => {
        const model: tf.Sequential = tf.sequential();

        before((done) => {
            // Input layer
            const totalAccessPoints = accessPoints.length;
            model.add(tf.layers.dense({ inputShape: [totalAccessPoints], units: 128, activation: 'relu' }));
            model.add(tf.layers.dense({ units: 64, activation: 'relu' }));
            model.add(tf.layers.dense({ units: 64, activation: 'relu' }));
            // Output layer (X, Y)
            model.add(tf.layers.dense({ units: 2, activation: 'sigmoid' }));
            
            model.compile({
                optimizer: 'sgd',
                loss: 'meanSquaredError',
                metrics: ['accuracy'],
            });
            done();
        });

        it('should train the model', (done) => {
            let size = trainData.length;
            const transformedInput = trainData
                .slice(0, size)
                .map(d => {
                    const inputs: number[] = [];
                    accessPoints.forEach(ap => {
                        const rssi = parseFloat(d[ap]);
                        inputs.push(Number.isNaN(rssi) ? 100 : rssi);
                    });
                    return inputs;
                });
            const transformedOutput = trainData
                .slice(0, size)
                .map(d => {
                    const outputs: number[] = [
                        parseFloat(d.X),
                        parseFloat(d.Y),
                    ];
                    return outputs;
                });
            size = testData.length
            const transformedTestInput = testData
                .slice(0, size)
                .map(d => {
                    const inputs: number[] = [];
                    accessPoints.forEach(ap => {
                        const rssi = parseFloat(d[ap]);
                        inputs.push(Number.isNaN(rssi) ? 100 : rssi);
                    });
                    return inputs;
                });
            const transformedTestOutput = testData
                .slice(0, size)
                .map(d => {
                    const outputs: number[] = [
                        parseFloat(d.X),
                        parseFloat(d.Y),
                    ];
                    return outputs;
                });
            model.fit(
                tf.tensor(transformedInput), 
                tf.tensor(transformedOutput), {
                batchSize: 32,
                epochs: 10,
                validationData: [
                    tf.tensor(transformedTestInput), 
                    tf.tensor(transformedTestOutput)
                ]
            }).then(() => {
                return Promise.resolve(model.evaluate(
                    tf.tensor(transformedTestInput), 
                    tf.tensor(transformedTestOutput)));
            }).then((info) => {
                const testAccPercent = info[1].dataSync()[0] * 100;
                console.log(testAccPercent)
                done();
            }).catch(done);
        });

        it('should predict locations', () => {
            const size = testData.length;
            const transformedTestInput = testData
                .slice(0, size)
                .map(d => {
                    const inputs: number[] = [];
                    accessPoints.forEach(ap => {
                        const rssi = parseFloat(d[ap]);
                        inputs.push(Number.isNaN(rssi) ? 100 : rssi);
                    });
                    return inputs;
                });
            const transformedTestOutput = testData
                .slice(0, size)
                .map(d => {
                    const outputs: number[] = [
                        parseFloat(d.X),
                        parseFloat(d.Y),
                        parseFloat(d.ORIENTATION)
                    ];
                    return outputs;
                });
            for (let i = 0 ; i < 10 ; i ++) {
                const output = model.predict(
                    tf.tensor([transformedTestInput[i]])
                ) as tf.Tensor;
                console.log(output.dataSync(), transformedTestOutput[i])
            }
        });
    });
});
