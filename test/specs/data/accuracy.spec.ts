import 'mocha';
import * as fs from 'fs';
import * as csv from 'csv-parser';
import { Vector3 } from '@openhps/core';
import { expect } from 'chai';
import { ensureBackend } from '../../tf';
import { AccuracyModel, AccuracyModelData, AccuracyModelInput, SignalStrenghts } from '../AccuracyModel';
import { Fingerprinting } from '../Fingerprinting';
import * as tf from '@tensorflow/tfjs';
const epochs = process.env.EPOCHS ? parseInt(process.env.EPOCHS, 10) : 200;
const modelDir = process.env.MODEL_DIR ?? 'test/data/model';
const ensembleSize = process.env.ENSEMBLE ? parseInt(process.env.ENSEMBLE, 10) : 1;

describe('data.openhps.accuracy', () => {
    const trainData: any[] = [];
    const rawTrainData: any[] = [];
    const testData: any[] = [];
    const rawTestData: any[] = [];
    const accessPoints: string[] = [];
    const fingerprinting: Fingerprinting = new Fingerprinting();
    let models: AccuracyModel[] = [];

    // Average prediction over the ensemble members (ENSEMBLE=1 => single model).
    async function ensemblePredict(input: AccuracyModelInput): Promise<number> {
        let sum = 0;
        for (const m of models) {
            sum += await m.predict(input);
        }
        return sum / models.length;
    }

    function loadData(path: string, array: any[]): Promise<void> {
        return new Promise((resolve) => {
            fs.createReadStream(path)
                .pipe(
                    csv({
                        separator: ',',
                    }),
                )
                .on('data', (row) => {
                    array.push(row);
                })
                .on('end', () => {
                    resolve();
                });
        });
    }

    function loadTrainData(): Promise<void> {
        return loadData('test/data/OpenHPS-2021-05/train/aggregated/wlan_fingerprints.csv', trainData);
    }

    function loadTestData(): Promise<void> {
        return loadData('test/data/OpenHPS-2021-05/test/aggregated/wlan_fingerprints.csv', testData);
    }

    function loadRawTrainData(): Promise<void> {
        return loadData('test/data/OpenHPS-2021-05/train/raw/wlan_fingerprints.csv', rawTrainData);
    }

    function loadRawTestData(): Promise<void> {
        return loadData('test/data/OpenHPS-2021-05/test/raw/wlan_fingerprints.csv', rawTestData);
    }

    before((done) => {
        ensureBackend()
            .then(() => loadTrainData())
            .then(() => {
                return loadTestData();
            })
            .then(() => {
                return loadRawTestData();
            })
            .then(() => {
                return loadRawTrainData();
            })
            .then(() => {
                // Common access points
                const trainAccessPoints = Object.keys(trainData[0]).filter((col) => col.startsWith('WAP_'));
                const testAccessPoints = Object.keys(testData[0]).filter((col) => col.startsWith('WAP_'));
                trainAccessPoints.forEach((col) => {
                    if (testAccessPoints.includes(col)) {
                        accessPoints.push(col);
                    }
                });
                testAccessPoints.forEach((col) => {
                    if (!accessPoints.includes(col)) {
                        accessPoints.push(col);
                    }
                });
                // Create fingerprints
                const fingerprints = [];
                trainData.forEach((data) => {
                    const fingerprint: { rssi: SignalStrenghts; x: number; y: number } = {
                        rssi: {},
                        x: parseFloat(data.X),
                        y: parseFloat(data.Y),
                    };
                    accessPoints.forEach((col) => {
                        fingerprint.rssi[col] = parseFloat(data[col]);
                    });
                    fingerprints.push(fingerprint);
                });
                fingerprinting.load(fingerprints);

                const accessPointPerScan = rawTrainData.map((d) => {
                    let count = 0;
                    accessPoints.forEach((col) => {
                        if (parseFloat(d[col]) !== 100) {
                            count++;
                        }
                    });
                    return count;
                });
                const avgAccessPointsPerScan = accessPointPerScan.reduce((a, b) => a + b, 0) / rawTrainData.length;
                const maxAccessPointsPerScan = accessPointPerScan.reduce((a, b) => Math.max(a, b), 0);

                console.log(`Access points: ${accessPoints.length}`);
                console.log(`Average access points per scan: ${avgAccessPointsPerScan}`);
                console.log(`Max access points per scan: ${maxAccessPointsPerScan}`);

                models = Array.from(
                    { length: ensembleSize },
                    () =>
                        new AccuracyModel({
                            accessPoints,
                        }),
                );
                return Promise.all(models.map((m, i) => m.load(i === 0 ? modelDir : `${modelDir}_${i}`)));
            })
            .then(() => {
                done();
            })
            .catch(done);
    });

    after((done) => {
        // Save the model(s); ensemble member i > 0 goes to modelDir_i.
        Promise.all(models.map((m, i) => m.save(i === 0 ? modelDir : `${modelDir}_${i}`)))
            .then(() => {
                done();
            })
            .catch(done);
    });

    describe('fingerprinting', () => {
        it('should perform kNN fingerprinting', (done) => {
            const k = 3;
            const weighted = true;
            const results = [];
            testData.forEach((data) => {
                const signalStrengths: SignalStrenghts = {};
                accessPoints.forEach((col) => {
                    signalStrengths[col] = parseFloat(data[col]);
                });
                const point = fingerprinting.calc(k, signalStrengths, weighted);
                results.push(point);
                expect(point.x).to.not.be.NaN;
                expect(point.y).to.not.be.NaN;
            });
            done();
        }).timeout(10000);
    });

    describe('training', () => {
        it('should train the fingerprinting model', (done) => {
            const trainDataObject: AccuracyModelData = {
                xs: [],
                ys: [],
            };
            for (let k = 1; k <= 10; k++) {
                rawTrainData.forEach((data) => {
                    const signalStrengths: SignalStrenghts = {};
                    accessPoints.forEach((col) => {
                        signalStrengths[col] = parseFloat(data[col]);
                    });
                    const point = fingerprinting.calc(k, signalStrengths, true);
                    trainDataObject.xs.push({
                        fingerprint: signalStrengths,
                        x: point.x,
                        y: point.y,
                        z: point.z,
                        k,
                    });
                    const accuracy = point.distanceTo(new Vector3(parseFloat(data.X), parseFloat(data.Y), 0));
                    trainDataObject.ys.push(accuracy);
                });
            }
            let trainChain: Promise<void> = Promise.resolve();
            models.forEach((model) => {
                trainChain = trainChain.then(() => model.train(trainDataObject, epochs));
            });
            trainChain
                .then(() => {
                    done();
                })
                .catch(done);
        }).timeout(-1);
    });

    describe('prediction', () => {
        it('should predict the fingerprinting accuracy', async () => {
            for (let k = 1; k <= 10; k++) {
                const diffXYList = [];
                for (const data of testData) {
                    const signalStrengths: SignalStrenghts = {};
                    accessPoints.forEach((col) => {
                        signalStrengths[col] = parseFloat(data[col]);
                    });
                    const point = fingerprinting.calc(k, signalStrengths, true);
                    expect(point.x).to.not.be.NaN;
                    expect(point.y).to.not.be.NaN;
                    const distanceX = Math.abs(point.x - parseFloat(data.X));
                    const distanceY = Math.abs(point.y - parseFloat(data.Y));
                    const distance = Math.sqrt(Math.pow(distanceX, 2) + Math.pow(distanceY, 2));
                    const prediction = await ensemblePredict({
                        fingerprint: signalStrengths,
                        x: point.x,
                        y: point.y,
                        z: point.z,
                        k,
                    });
                    diffXYList.push(Math.abs(prediction - distance));
                }
                const avgDiffXY = diffXYList.reduce((a, b) => a + b, 0) / diffXYList.length;
                const maxDiffXY = diffXYList.reduce((a, b) => Math.max(a, b), 0);
                const minDiffXY = diffXYList.reduce((a, b) => Math.min(a, b), 0);
                console.log(`K: ${k}`);
                console.log(`Average diff XY: ${avgDiffXY}`);
                console.log(`Max diff XY: ${maxDiffXY}`);
                console.log(`Min diff XY: ${minDiffXY}`);
                console.log(``);
            }
        });

        it('should predict accuracy from X,Y and save to JSON', async () => {
            const results = [];
            for (let k = 1; k <= 10; k++) {
                const diffXYList = [];
                for (const data of [...testData, ...trainData]) {
                    const signalStrengths: SignalStrenghts = {};
                    accessPoints.forEach((col) => {
                        signalStrengths[col] = parseFloat(data[col]);
                    });
                    const point = fingerprinting.calc(k, signalStrengths, true);
                    expect(point.x).to.not.be.NaN;
                    expect(point.y).to.not.be.NaN;
                    const distanceX = Math.abs(point.x - parseFloat(data.X));
                    const distanceY = Math.abs(point.y - parseFloat(data.Y));
                    const distance = Math.sqrt(Math.pow(distanceX, 2) + Math.pow(distanceY, 2));
                    const prediction = await ensemblePredict({
                        fingerprint: signalStrengths,
                        x: point.x,
                        y: point.y,
                        z: point.z,
                        k,
                    });
                    results.push({
                        x: point.x,
                        y: point.y,
                        k,
                        accuracy: Math.abs(prediction),
                    });
                    diffXYList.push(Math.abs(prediction - distance));
                }
                const avgDiffXY = diffXYList.reduce((a, b) => a + b, 0) / diffXYList.length;
                console.log(`K: ${k}`);
                console.log(`Average diff XY: ${avgDiffXY}`);
            }
            fs.writeFileSync('accuracy_results.json', JSON.stringify(results, null, 2));
        });

        it('should predict RSSI values from an X, Y coordinate', async () => {
            const diffs = [];
            for (const data of testData) {
                const signalStrengths: SignalStrenghts = {};
                accessPoints.forEach((col) => {
                    signalStrengths[col] = parseFloat(data[col]);
                });
                const prediction = await models[0].predictRSSI(parseFloat(data.X), parseFloat(data.Y), 0);
                accessPoints.forEach((col) => {
                    expect(prediction[col]).to.not.be.NaN;
                });
                const accuracy1 = await ensemblePredict({
                    fingerprint: prediction,
                    x: parseFloat(data.X),
                    y: parseFloat(data.Y),
                    z: 0,
                    k: 3,
                });
                const accuracy2 = await ensemblePredict({
                    fingerprint: signalStrengths,
                    x: parseFloat(data.X),
                    y: parseFloat(data.Y),
                    z: 0,
                    k: 3,
                });
                const diff = Math.abs(accuracy1 - accuracy2);
                diffs.push(diff);
            }
            const avgDiff = diffs.reduce((a, b) => a + b, 0) / diffs.length;
            const maxDiff = diffs.reduce((a, b) => Math.max(a, b), 0);
            const minDiff = diffs.reduce((a, b) => Math.min(a, b), 0);
            console.log(`Average diff: ${avgDiff}`);
            console.log(`Max diff: ${maxDiff}`);
            console.log(`Min diff: ${minDiff}`);
        });
    });
});
