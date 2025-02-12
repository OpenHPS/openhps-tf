import { Service, ServiceOptions } from "@openhps/core";
import * as tf from '@tensorflow/tfjs';
import { TensorFlowOptions } from "../types";

export interface TFModel {
    /**
     * TensorFlow layers model.
     */
    model?: tf.LayersModel;
    /**
     * Model name.
     */
    name: string;
    /**
     * TensorFlow options.
     */
    options?: TensorFlowOptions;
    /**
     * File path or URL of the model.
     */
    fileOrUrl?: string;
}

/**
 * TensorFlowService class for managing TensorFlow models.
 */
export class TensorFlowService extends Service {
    /**
     * TensorFlow layers model.
     */
    models?: Map<string, TFModel> = new Map();
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
        if (this.options.model) {
            this.addModel({
                name: this.options.name,
                fileOrUrl: this.options.model
            });
        }

        this.once('build', this._onBuild.bind(this));
    }

    private _onBuild(): Promise<void> {
        return new Promise((resolve, reject) => {
            let promise = Promise.resolve();
            this.models.forEach((model: TFModel) => {
                promise = promise.then(() => {
                    if (model.fileOrUrl) {
                        return this.loadModel(model);
                    } else {
                        return Promise.resolve();
                    }
                });
            });
            promise.then(() => {
                resolve();
            }).catch(reject);
        });
    }

    /**
     * Has a TensorFlow model.
     * @param name TensorFlow model name.
     * @returns {boolean} Returns true if the model exists.
     */
    hasModel(name: string): boolean {
        return this.models.has(name);
    }

    getModel(name: string): TFModel {
        return this.models.get(name);
    }
    
    /**
     * Add a TensorFlow model to the service.
     * @param model Model to add.
     * @returns {this} Returns itself.
     */
    addModel(model: TFModel): this {
        this.models.set(model.name, model);
        return this;
    }

    /**
     * Load a TensorFlow model from a file or URL.
     * @param {TFModel} model - Model to load.
     * @returns {Promise<void>} Promise that resolves when the model is loaded.
     */
    loadModel(model: TFModel): Promise<void> {
        return new Promise((resolve, reject) => {
            if (!model.fileOrUrl) {
                reject(new Error(`Model ${model.name} does not have a file or URL`));
            }
            tf.loadLayersModel(model.fileOrUrl).then(layersModel => {
                this.models.set(model.name ?? model.fileOrUrl, {
                    model: layersModel,
                    name: model.name ?? model.fileOrUrl,
                    fileOrUrl: model.fileOrUrl
                });
                resolve();
            }).catch(reject);
        });
    }

    /**
     * Save the TensorFlow model to a file or URL.
     * @param {string} name - Model name.
     * @param {string} [fileOrUrl] - File path or URL to save the model.
     * @returns {Promise<void>} Promise that resolves when the model is saved.
     */
    saveModel(name: string, fileOrUrl?: string): Promise<void> {
        return new Promise((resolve, reject) => {
            const model = this.models.get(name);
            if (model) {
                model.model.save(fileOrUrl ?? model.fileOrUrl).then(() => {
                    model.fileOrUrl = fileOrUrl ?? model.fileOrUrl;
                    resolve();
                }).catch(reject);
            } else {
                reject(new Error(`Model ${name} not found`));
            }
        });
    }

}

/**
 * TensorFlow service options.
 */
export interface TensorFlowServiceOptions extends ServiceOptions {
    /**
     * Model name
     */
    name?: string;
    /**
     * Model path or URL.
     */
    model?: string;
    /**
     * TensorFlow options.
     */
    tensorFlow?: TensorFlowOptions;
}
