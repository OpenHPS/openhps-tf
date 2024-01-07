import 'mocha';
import * as fs from 'fs';
import * as csv from 'csv-parser';
import { Absolute2DPosition, AbsolutePosition, Vector3 } from '@openhps/core';
import { DistanceFunction, WeightFunction, KDTree, Fingerprint } from '@openhps/fingerprinting';
import { expect } from 'chai';
import { AccuracyModel, AccuracyModelData, SignalStrenghts } from '../AccuracyModel';

const epochs = 1;

describe('data.openhps.accuracy', () => {
    const trainData: any[] = [];
    const rawTrainData: any[] = [];
    const testData: any[] = [];
    const rawTestData: any[] = [];
    const accessPoints: string[] = [];
    const fingerprints = Array<{ rssi: SignalStrenghts, x: number, y: number }>();
    let model: AccuracyModel;
    let tree: KDTree;

    function loadData(path: string, array: any[]): Promise<void> {
        return new Promise((resolve) => {
            fs.createReadStream(path)
                .pipe(csv({
                    separator: ","
                })).on('data', row => {
                    array.push(row);
                }).on('end', () => {
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

    function fingerprinting(k: number, data: SignalStrenghts, weighted: boolean = false, naive: boolean = false): Vector3 {
        const dataObjectPoint: number[] = [];
        Object.keys(data)
            // Sort alphabetically
            .sort((a, b) =>
                a.localeCompare(b),
            )
            .forEach((rel) => {
                dataObjectPoint.push(data[rel]);
            });
        // Perform reverse fingerprinting
        let results = new Array<[AbsolutePosition, number]>();
        if (naive) {
            fingerprints.forEach((f) => {
                let distance = DistanceFunction.EUCLIDEAN(dataObjectPoint, Object.values(f.rssi));
                if (distance === 0) {
                    distance = 1e-5;
                }
                results.push([new Absolute2DPosition(f.x, f.y), distance]);
            });
            results = results
                // Sort by euclidean distance
                .sort((a, b) => a[1] - b[1])
                // Only the first K neighbours
                .splice(0, k);
        } else {
            results = tree.nearest(dataObjectPoint, k);
        }

        const point: Vector3 = new Vector3(0, 0, 0);
        if (weighted) {
            let scale = 0;
            results.forEach((sortedFingerprint) => {
                const weight = WeightFunction.SQUARE(sortedFingerprint[1]);
                scale += weight;
                point.add(sortedFingerprint[0].toVector3().multiplyScalar(weight));
            });
            point.divideScalar(scale);
        } else {
            results.forEach((sortedFingerprint) => {
                point.add(sortedFingerprint[0].toVector3());
            });
            point.divideScalar(k);
        }
        return point;
    }

    before((done) => {
        loadTrainData().then(() => {
            return loadTestData();
        }).then(() => {
            return loadRawTestData();
        }).then(() => {
            return loadRawTrainData();
        }).then(() => {
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
            // Create fingerprints
            trainData.forEach((data) => {
                const fingerprint: { rssi: SignalStrenghts, x: number, y: number } = { rssi: {}, x: parseFloat(data.X), y: parseFloat(data.Y) };
                accessPoints.forEach((col) => {
                    fingerprint.rssi[col] = parseFloat(data[col]);
                });
                fingerprints.push(fingerprint);
            });
            // Create KD-tree
            tree = new KDTree(fingerprints.map(f => {
                const fingerprint = new Fingerprint();
                Object.keys(f.rssi).forEach(key => {
                    fingerprint.addFeature(key, f.rssi[key]);
                });
                fingerprint.setPosition(new Absolute2DPosition(f.x, f.y));
                fingerprint.computeVector((val) => val.reduce((a, b) => a + b, 0) / val.length);
                return fingerprint;
            }), DistanceFunction.EUCLIDEAN);

            const accessPointPerScan = rawTrainData.map(d => {
                let count = 0;
                accessPoints.forEach(col => {
                    if (parseFloat(d[col]) !== 100) {
                        count++;
                    }
                });
                return count;
            });
            const avgAccessPointsPerScan = accessPointPerScan
                .reduce((a, b) => a + b, 0) / rawTrainData.length;
            const maxAccessPointsPerScan = accessPointPerScan
                .reduce((a, b) => Math.max(a, b), 0);

            console.log(`Access points: ${accessPoints.length}`);
            console.log(`Average access points per scan: ${avgAccessPointsPerScan}`);
            console.log(`Max access points per scan: ${maxAccessPointsPerScan}`);

            model = new AccuracyModel({
                accessPoints,
            });
            return model.load("test/data/model");
        }).then(() => {
            done();
        }).catch(done);
    });

    after((done) => {
        // Save the model
        model.save("test/data/model").then(() => {
            done();
        }).catch(done);
    });

    describe('fingerprinting', () => {
        it('should perform kNN fingerprinting', (done) => {
            const k = 3;
            const weighted = true;
            const naive = true;
            const results = [];
            testData.forEach((data) => {
                const signalStrengths: SignalStrenghts = {};
                accessPoints.forEach((col) => {
                    signalStrengths[col] = parseFloat(data[col]);
                });
                const point = fingerprinting(k, signalStrengths, weighted, naive);
                results.push(point);
                expect(point.x).to.not.be.NaN;
                expect(point.y).to.not.be.NaN;
            });
            done();
        }).timeout(10000);
    })


    describe('training', () => {
        it('should train the fingerprinting model', (done) => {
            const trainDataObject: AccuracyModelData = {
                xs: [],
                ys: [],
            };
            for (let k = 1; k < 5; k++) {
                rawTrainData.forEach((data) => {
                    const signalStrengths: SignalStrenghts = {};
                    accessPoints.forEach((col) => {
                        signalStrengths[col] = parseFloat(data[col]);
                    });
                    const point = fingerprinting(k, signalStrengths, true, false);
                    trainDataObject.xs.push({
                        fingerprint: signalStrengths,
                        x: point.x,
                        y: point.y,
                        z: point.z,
                        k
                    });
                    const accuracy = point.distanceTo(new Vector3(parseFloat(data.X), parseFloat(data.Y), 0));
                    trainDataObject.ys.push(accuracy);
                });
            }
            model.train(trainDataObject, epochs).then(() => {
                done();
            }).catch(done);
        }).timeout(-1);
    });

    describe('prediction', () => {	
        it('should predict the fingerprinting accuracy', (done) => {
            for (let k = 1; k < 5; k++) {
                const diffXYList = [];
                testData.forEach((data) => {
                    const signalStrengths: SignalStrenghts = {};
                    accessPoints.forEach((col) => {
                        signalStrengths[col] = parseFloat(data[col]);
                    });
                    const point = fingerprinting(k, signalStrengths, true, false);
                    expect(point.x).to.not.be.NaN;
                    expect(point.y).to.not.be.NaN;
                    const distanceX = Math.abs(point.x - parseFloat(data.X));
                    const distanceY = Math.abs(point.y - parseFloat(data.Y));
                    const distance = Math.sqrt(Math.pow(distanceX, 2) + Math.pow(distanceY, 2));
                    const prediction = model.predict({
                        fingerprint: signalStrengths,
                        x: point.x,
                        y: point.y,
                        z: point.z,
                        k
                    });
                    diffXYList.push(Math.abs(prediction - distance));
                    //console.log(`X: ${point.x} Y: ${point.y} | ${parseFloat(data.X)} ${parseFloat(data.Y)} | ${diffX} ${diffY}`);
                });
                const avgDiffXY = diffXYList.reduce((a, b) => a + b, 0) / diffXYList.length;
                const maxDiffXY = diffXYList.reduce((a, b) => Math.max(a, b), 0);
                const minDiffXY = diffXYList.reduce((a, b) => Math.min(a, b), 0);
                console.log(`K: ${k}`);
                console.log(`Average diff XY: ${avgDiffXY}`);
                console.log(`Max diff XY: ${maxDiffXY}`);
                console.log(`Min diff XY: ${minDiffXY}`);
                console.log(``);
            }
           
            done();
        });
    });

});
