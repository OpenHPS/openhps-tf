import { DistanceFunction, Fingerprint, KDTree, WeightFunction } from "@openhps/fingerprinting";
import { SignalStrenghts } from "./AccuracyModel";
import { Absolute3DPosition, AbsolutePosition, Vector3 } from "@openhps/core";

export interface FingerprintObject { rssi: SignalStrenghts, x: number, y: number, z: number };

export class Fingerprinting {
    private fingerprints = Array<FingerprintObject>();
    private tree: KDTree;
    
    load(data: Array<FingerprintObject>) {
        this.fingerprints = data;
        this.tree = new KDTree(this.fingerprints.map(f => {
            const fingerprint = new Fingerprint();
            fingerprint.vector = Object.values(f.rssi);
            fingerprint.setPosition(new Absolute3DPosition(f.x, f.y, f.z));
            return fingerprint;
        }), DistanceFunction.EUCLIDEAN);
    }

    calc(k: number, data: SignalStrenghts, weighted: boolean = false): Vector3 {
        const dataObjectPoint: number[] = Object.values(data);
        // Perform reverse fingerprinting
        const results = this.tree.nearest(dataObjectPoint, k);
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
}
