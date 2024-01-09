import 'mocha';
import * as fs from 'fs';
import * as csv from 'csv-parser';
import { Vector3 } from '@openhps/core';
import { expect } from 'chai';
import { AccuracyModel, AccuracyModelData, SignalStrenghts } from '../AccuracyModel';
import { Fingerprinting } from '../Fingerprinting';

const epochs = 5;

describe('data.ujiindoorloc.accuracy', () => {
    const trainData: any[] = [];
    const testData: any[] = [];
    const accessPoints: string[] = [];
    let model: AccuracyModel;
    const fingerprinting = new Fingerprinting();

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
        return loadData('test/data/UJIIndoorLoc/trainingData.csv', trainData);
    }

    function loadTestData(): Promise<void> {
        return loadData('test/data/UJIIndoorLoc/validationData.csv', testData);
    }

    before((done) => {
        loadTrainData().then(() => {
            return loadTestData();
        }).then(() => {
            console.log(`Train data: ${trainData.length}`);
            console.log(`Test data: ${testData.length}`);

            // Common access points
            const trainAccessPoints = Object.keys(trainData[0]).filter(col => col.startsWith("WAP"));
            const testAccessPoints = Object.keys(testData[0]).filter(col => col.startsWith("WAP"));
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
            const fingerprints = [];
            const locations: Array<{ x: number, y: number, z: number }> = [];
            trainData.forEach((data) => {
                const fingerprint: { rssi: SignalStrenghts, x: number, y: number, z: number } = { rssi: {}, x: parseFloat(data.LATITUDE), y: parseFloat(data.LONGITUDE), z: parseFloat(data.FLOOR) };
                accessPoints.forEach((col) => {
                    fingerprint.rssi[col] = parseFloat(data[col]);
                });
                if (!locations.find(l => l.x === fingerprint.x && l.y === fingerprint.y && l.z === fingerprint.z)) {
                    locations.push({ x: fingerprint.x, y: fingerprint.y, z: fingerprint.z });
                }
                fingerprints.push(fingerprint);
            });
            // Aggregate fingerprints
            const aggregatedFingerprints: Array<{ rssi: SignalStrenghts, x: number, y: number, z: number }> = [];
            locations.forEach(location => {
                const locationFingerprints: Array<{ rssi: SignalStrenghts, x: number, y: number, z: number }> = fingerprints.filter(f => f.x === location.x && f.y === location.y && f.z === location.z);
                aggregatedFingerprints.push({
                    rssi: locationFingerprints.reduce((a, b) => {
                        const rssi: SignalStrenghts = {};
                        accessPoints.forEach(col => {
                            if (a[col]) {
                                rssi[col] = (a[col] + b.rssi[col]) / 2;
                            } else {
                                rssi[col] = b.rssi[col];
                            }
                        });
                        return rssi;
                    }, {}),
                    x: location.x,
                    y: location.y,
                    z: location.z
                });
            });
            fingerprinting.load(aggregatedFingerprints);

            const accessPointPerScan = trainData.map(d => {
                let count = 0;
                accessPoints.forEach(col => {
                    if (parseFloat(d[col]) !== 100) {
                        count++;
                    }
                });
                return count;
            });
            const avgAccessPointsPerScan = accessPointPerScan
                .reduce((a, b) => a + b, 0) / trainData.length;
            const maxAccessPointsPerScan = accessPointPerScan
                .reduce((a, b) => Math.max(a, b), 0);

            console.log(`Access points: ${accessPoints.length}`);
            console.log(`Average access points per scan: ${avgAccessPointsPerScan}`);
            console.log(`Max access points per scan: ${maxAccessPointsPerScan}`);

            model = new AccuracyModel({
                accessPoints,
            });
            return model.load("test/data/model2");
        }).then(() => {
            done();
        }).catch(done);
    });

    after((done) => {
        // Save the model
        model.save("test/data/model2").then(() => {
            done();
        }).catch(done);
    });

    describe('fingerprinting', () => {
        it('should perform kNN fingerprinting', (done) => {
            const k = 3;
            const weighted = true;
            const results = [];
            const accuracy = [];
            testData.forEach((data, index) => {
                const signalStrengths: SignalStrenghts = {};
                accessPoints.forEach((col) => {
                    signalStrengths[col] = parseFloat(data[col]);
                });
                const point = fingerprinting.calc(k, signalStrengths, weighted);
                results.push(point);
                accuracy.push(point.distanceTo(new Vector3(parseFloat(data.LATITUDE), parseFloat(data.LONGITUDE), parseFloat(data.FLOOR))));
                expect(point.x).to.not.be.NaN;
                expect(point.y).to.not.be.NaN;
                expect(point.z).to.not.be.NaN;
                process.stdout.write(`${index + 1} / ${testData.length}\t\t\r`);
            });
            const avgAccuracy = accuracy.reduce((a, b) => a + b, 0) / accuracy.length;
            const maxAccuracy = accuracy.reduce((a, b) => Math.max(a, b), 0);
            const minAccuracy = accuracy.reduce((a, b) => Math.min(a, b), 0);
            console.log(`Average accuracy: ${avgAccuracy}`);
            console.log(`Max accuracy: ${maxAccuracy}`);
            console.log(`Min accuracy: ${minAccuracy}`);
            done();
        }).timeout(-1);
    })

    describe('training', () => {
        it('should train the fingerprinting model', (done) => {
            const trainDataObject: AccuracyModelData = {
                xs: [],
                ys: [],
            };
            for (let k = 3; k <= 3; k++) {
                trainData.forEach((data, index) => {
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
                        k
                    });
                    const accuracy = point.distanceTo(new Vector3(parseFloat(data.LATITUDE), parseFloat(data.LONGITUDE), parseFloat(data.FLOOR)));
                    trainDataObject.ys.push(accuracy);
                    process.stdout.write(`k: ${k} | ${(index) + 1} / ${trainData.length}\t\t\r`);
                });
            }
            model.train(trainDataObject, epochs).then(() => {
                done();
            }).catch(done);
        }).timeout(-1);
    });

    describe('prediction', () => {	
        it('should predict the fingerprinting accuracy', (done) => {
            for (let k = 3; k <= 3; k++) {
                const diffXYList = [];
                testData.forEach((data) => {
                    const signalStrengths: SignalStrenghts = {};
                    accessPoints.forEach((col) => {
                        signalStrengths[col] = parseFloat(data[col]);
                    });
                    const point = fingerprinting.calc(k, signalStrengths, true);
                    expect(point.x).to.not.be.NaN;
                    expect(point.y).to.not.be.NaN;
                    expect(point.z).to.not.be.NaN;
                    const distanceX = Math.abs(point.x - parseFloat(data.LATITUDE));
                    const distanceY = Math.abs(point.y - parseFloat(data.LONGITUDE));
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
