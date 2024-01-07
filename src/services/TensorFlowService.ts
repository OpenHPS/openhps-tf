import { Service } from "@openhps/core";
import * as tf from '@tensorflow/tfjs';
import { TFPreProcessingFn, TensorFlowOptions } from "../types";

/**
 * TensorFlowService class for managing TensorFlow models.
 */
export class TensorFlowService extends Service {
    /**
     * TensorFlow layers model.
     * @type {tf.LayersModel}
     */
    layersModel: tf.LayersModel;
    /**
     * TensorFlow service options.
     * @type {TensorFlowServiceOptions}
     */
    protected options: TensorFlowServiceOptions;

    /**
     * Create a new TensorFlow service.
     * @param {TensorFlowServiceOptions} options - TensorFlow service options.
     */
    constructor(options: TensorFlowServiceOptions) {
        super();
        this.options = options;
    }
    
    /**
     * Load a TensorFlow model from a file or URL.
     * @param {string} fileOrUrl - File path or URL of the model.
     * @returns {Promise<void>} Promise that resolves when the model is loaded.
     */
    loadModel(fileOrUrl: string): Promise<void> {
        return new Promise((resolve, reject) => {
            tf.loadLayersModel(fileOrUrl).then(model => {
                this.layersModel = model;
                resolve();
            }).catch(reject);
        });
    }

    /**
     * Save the TensorFlow model to a file or URL.
     * @param {string} fileOrUrl - File path or URL to save the model.
     * @returns {Promise<void>} Promise that resolves when the model is saved.
     */
    saveModel(fileOrUrl: string): Promise<void> {
        return new Promise((resolve, reject) => {
            this.layersModel.save(fileOrUrl).then(() => {
                resolve();
            }).catch(reject);
        });
    }

}

/**
 * TensorFlow service options.
 */
export interface TensorFlowServiceOptions extends TensorFlowOptions {
    
}
