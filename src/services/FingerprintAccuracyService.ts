import { Absolute3DPosition, ServiceOptions } from "@openhps/core";
import { TensorFlowService } from "./TensorFlowService";
import * as tf from '@tensorflow/tfjs';
import * as path from "path";

type SignalStrenghts = { [key: string]: number };

export class FingerprintAccuracy extends TensorFlowService {
    protected options: FingerprintAccuracyOptions;
    protected accessPoints: string[];

    constructor(options: FingerprintAccuracyOptions) {
        super(options);
    }

    private get inputDimension(): number {
        if (this.accessPoints.length <= 256) {
            return 16;
        } else if (this.accessPoints.length <= 1024) {
            return 32;
        }
    }

    protected get rssiModel(): tf.LayersModel {
        return this.models.get('rssi').model;
    }

    protected set rssiModel(model: tf.LayersModel) {
        this.models.set('rssi', { 
            name: 'rssi', 
            model, 
            fileOrUrl: path.join(this.options.directory, 'rssi.json') 
        });
    }
    
    protected predictRSSI(position: Absolute3DPosition): SignalStrenghts {
        const signalStrengths: SignalStrenghts = {};
        const prediction = this.rssiModel.predict(
            tf.tensor([[x, y, z]])
        ) as tf.Tensor;
        const predictionData = Array.from(prediction.dataSync());
        const normalizedData = predictionData.map(f => {
            return Math.round(f === 0 ? 100 : (f * 99) - 99);
        });
        this.accessPoints.forEach((ap ,i) => {
            const rssi = normalizedData[i];
            signalStrengths[ap] = rssi;
        });
        return signalStrengths;
    }

}

export interface FingerprintAccuracyOptions extends ServiceOptions {
    directory?: string;
}
