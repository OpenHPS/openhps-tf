import { Absolute3DPosition, ServiceOptions } from '@openhps/core';
import { TensorFlowService } from './TensorFlowService';
import * as tf from '@tensorflow/tfjs';
import * as path from 'path';

type SignalStrenghts = { [key: string]: number };

export class FingerprintAccuracy extends TensorFlowService {
    protected options: FingerprintAccuracyOptions;
    protected accessPoints: string[];

    constructor(options: FingerprintAccuracyOptions) {
        super(options);
        this.accessPoints = options.accessPoints ?? [];
    }

    private get inputDimension(): number {
        if (this.accessPoints.length <= 256) {
            return 16;
        } else if (this.accessPoints.length <= 1024) {
            return 32;
        }
        return 64;
    }

    protected get rssiModel(): tf.LayersModel {
        return this.models.get('rssi').model;
    }

    protected set rssiModel(model: tf.LayersModel) {
        this.models.set('rssi', {
            name: 'rssi',
            model,
            fileOrUrl: path.join(this.options.directory, 'rssi.json'),
        });
    }

    protected predictRSSI(position: Absolute3DPosition): SignalStrenghts {
        const signalStrengths: SignalStrenghts = {};
        const prediction = this.rssiModel.predict(tf.tensor([[position.x, position.y, position.z ?? 0]])) as tf.Tensor;
        const predictionData = Array.from(prediction.dataSync());
        const N = this.accessPoints.length;
        const strongest = this.options.minRSSI ?? -99;
        const weakest = this.options.maxRSSI ?? -96;
        this.accessPoints.forEach((ap, i) => {
            const normalized = predictionData[i];
            const seen = predictionData[N + i] >= 0.5;
            signalStrengths[ap] = seen ? Math.round(weakest + normalized * (strongest - weakest)) : 100;
        });
        return signalStrengths;
    }
}

export interface FingerprintAccuracyOptions extends ServiceOptions {
    directory?: string;
    accessPoints?: string[];
    minRSSI?: number;
    maxRSSI?: number;
}
