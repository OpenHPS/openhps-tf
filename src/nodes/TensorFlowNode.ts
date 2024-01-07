import { DataFrame, ProcessingNode, ProcessingNodeOptions, PushOptions } from "@openhps/core";
import * as tf from '@tensorflow/tfjs';
import { TFPreProcessingFn, TFPostProcessingFn, TensorFlowOptions } from "../types";


/**
 * TensorFlowNode is a processing node that uses TensorFlow models for data processing.
 * It extends the ProcessingNode class and provides methods for loading and saving models,
 * as well as processing input data using the loaded model.
 *
 * @typeparam In The input DataFrame type.
 * @typeparam Out The output DataFrame type.
 */
export class TensorFlowNode<In extends DataFrame, Out extends DataFrame>
    extends ProcessingNode<In, Out> {
    protected options: TensorFlowNodeOptions;
    layersModel: tf.LayersModel;
    protected preProcessing: TFPreProcessingFn<In>;
    protected postProcessing: TFPostProcessingFn<In, Out>;
    isTraining: boolean = false;
    
    /**
     * Creates an instance of TensorFlowNode.
     * 
     * @param {TFPreProcessingFn<In>} preProcessing - The pre-processing function.
     * @param {TFPostProcessingFn<In, Out>} postProcessing - The post-processing function.
     * @param {TensorFlowNodeOptions} [options] - The options for the TensorFlowNode.
     */
    constructor(
        preProcessing: TFPreProcessingFn<In>, 
        postProcessing: TFPostProcessingFn<In, Out>, 
        options?: TensorFlowNodeOptions) {
        super(options);

        // Default options
        this.options.prediction = this.options.prediction ?? {};

        this.preProcessing = preProcessing;
        this.postProcessing = postProcessing;
    }

    /**
     * Create a new TensorFlow model
     * 
     * @param {Function} callback Callback to create the model
     * @returns {TensorFlowNode<In, Out>} Returns itself
     */
    createModel(callback: (model: tf.LayersModel) => void): this {
        this.layersModel = tf.sequential();
        callback(this.layersModel);
        return this;
    }

    /**
     * Load a tensorflow model
     * @param {string} fileOrUrl File or URL to load
     * @returns 
     */
    loadModel(fileOrUrl: string): Promise<void> {
        return new Promise((resolve, reject) => {
            tf.loadLayersModel(fileOrUrl).then(model => {
                this.layersModel = model;
                resolve();
            }).catch(reject);
        });
    }

    
    saveModel(fileOrUrl: string): Promise<void> {
        return new Promise((resolve, reject) => {
            this.layersModel.save(fileOrUrl).then(() => {
                resolve();
            }).catch(reject);
        });
    }

    process(frame: In, options?: PushOptions): Promise<Out> {
        return new Promise((resolve, reject) => {
            const data = this.preProcessing(frame);
            if (this.options.training && this.isTraining) {
                const labels = this.options.training.preProcessing(frame);
                this.layersModel.fit(data, labels, this.options.training)
                    .then(() => {
                        resolve(frame as unknown as Out);
                    }).catch(reject);
            } else {
                const output = this.layersModel.predict(data, this.options.prediction);
                const outputFrame = this.postProcessing(frame, output);
                resolve(outputFrame);
            }
        });
    }

}

export interface TensorFlowNodeOptions extends ProcessingNodeOptions, TensorFlowOptions {
}
