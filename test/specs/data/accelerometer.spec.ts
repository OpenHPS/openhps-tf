import 'mocha';
import * as tf from '../../tf';
import * as fs from 'fs';
import * as csv from 'csv-parser';

describe('data.vmalyi-run-or-walk', () => {
    const model: tf.Sequential = tf.sequential();
    const data: { xs: tf.Tensor, labels: tf.Tensor } = { xs: undefined, labels: undefined };

    before((done) => {
        model.add(tf.layers.dense({units: 14, inputShape: [7] }));
        model.add(tf.layers.dense({ units: 20, activation: 'relu' }));
        model.add(tf.layers.dense({ units: 10, activation: 'relu' }));
        model.add(tf.layers.dense({ units: 5, activation: 'relu' }));
        model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));
        model.compile({
            loss: 'binaryCrossentropy',
            optimizer: 'adam',
            metrics: ['accuracy'],
        });

        const rows = [];
        fs.createReadStream('test/data/vmalyi-run-or-walk/dataset.csv')
            .pipe(csv({
                separator: ","
            })).on('data', row => {
                rows.push(row);
            }).on('end', () =>{
                data.xs = tf.tensor(rows
                    .map(d => {
                        return [
                            parseFloat(d.acceleration_x),
                            parseFloat(d.acceleration_y),
                            parseFloat(d.acceleration_z),
                            parseFloat(d.gyro_x),
                            parseFloat(d.gyro_y),
                            parseFloat(d.gyro_z),
                            parseFloat(d.wrist)
                        ];
                    }));
                data.labels = tf.tensor(rows
                    .map(d => {
                        return [
                            parseInt(d.activity)
                        ];
                    }));
                done();
            });
    });

    it('should train the model', (done) => {
        model.fit(data.xs, data.labels, {
            epochs: 10,
            verbose: 0,
            callbacks: {
                onEpochEnd: (epoch, logs) => {
                    console.log(`Epoch ${epoch}: loss = ${logs.loss}, accuracy = ${logs.acc}`);
                }
            }
        }).then(() => {
            done();
        });
    }).timeout(-1);

});
