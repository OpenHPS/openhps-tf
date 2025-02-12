import { DataFrame, ProcessingNode, ProcessingNodeOptions, PushOptions } from "@openhps/core";
import * as tf from '@tensorflow/tfjs';
import { TFPreProcessingFn, TFPostProcessingFn, TensorFlowOptions } from "../types";
import { TensorFlowService } from "../services";


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
    protected options: TensorFlowNodeOptions<In, Out>;
    layersModel: tf.LayersModel;
    protected preProcessing?: TFPreProcessingFn<In>;
    protected postProcessing?: TFPostProcessingFn<In, Out>;
    isTraining: boolean = false;
    
    /**
     * Creates an instance of TensorFlowNode.
     * 
     * @param {TensorFlowNodeOptions} [options] - The options for the TensorFlowNode.
     */
    constructor(
        options?: TensorFlowNodeOptions<In, Out>) {
        super(options);

        // Pre processing and post processing
        this.preProcessing = this.options.preProcessing ?? ((frame: any) => frame);
        this.postProcessing = this.options.postProcessing ?? ((_: any, output: any) => output);

        this.once('build', this._onBuild.bind(this));
    }
    
    private _onBuild(): Promise<void> {
        return new Promise((resolve, reject) => {
            let modelPromise = Promise.resolve();
            if (this.options.model && typeof this.options.model === 'string') {
                modelPromise = this.loadModel(this.options.model);
            } else if (this.options.model && typeof this.options.model === 'object') {
                const name = (this.options.model as TensorFlowServiceModel).name;
                const services = this.model.findAllServices(this.options.model.service ?? TensorFlowService);
                const service: TensorFlowService = services.find((service) => service.hasModel(name));
                if(service) {
                    const model = service.getModel(name);
                    this.layersModel = model.model;
                    this.options.tensorFlow = this.options.tensorFlow ?? model.options;
                    this.options.fileOrUrl = this.options.fileOrUrl ?? model.fileOrUrl;
                }
                modelPromise = Promise.resolve();
            } else {
                modelPromise = Promise.resolve();
            }
            // Check if nodes need to be chained
            modelPromise.then(() => {
                // Default options
                this.options.tensorFlow = this.options.tensorFlow ?? {};
                this.options.tensorFlow.prediction = this.options.tensorFlow.prediction ?? {};

                this.inlets.forEach(inlet => {
                    // inlet.inletNode
                });
                resolve();
            }).catch(reject);
        });
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
     * @param {string} [fileOrUrl] File or URL to load
     * @returns 
     */
    loadModel(fileOrUrl?: string): Promise<void> {
        return new Promise((resolve, reject) => {
            tf.loadLayersModel(fileOrUrl ?? this.options.fileOrUrl).then(model => {
                this.layersModel = model;
                resolve();
            }).catch(reject);
        });
    }

    saveModel(fileOrUrl?: string): Promise<void> {
        return new Promise((resolve, reject) => {
            this.layersModel.save(fileOrUrl ?? this.options.fileOrUrl).then(() => {
                resolve();
            }).catch(reject);
        });
    }

    process(frame: In, options?: PushOptions): Promise<Out> {
        return new Promise((resolve, reject) => {
            const data = this.preProcessing(frame);
            if (this.options.tensorFlow.training && this.isTraining) {
                const labels = this.options.tensorFlow.training.preProcessing(frame);
                this.layersModel.fit(data, labels, this.options.tensorFlow.training)
                    .then(() => {
                        resolve(frame as unknown as Out);
                    }).catch(reject);
            } else {
                const output = this.layersModel.predict(data, this.options.tensorFlow.prediction);
                const outputFrame = this.postProcessing(frame, output);
                resolve(outputFrame);
            }
        });
    }

}

interface TensorFlowServiceModel {
    /**
     * TensorFlow service type
     */
    service?: new () => TensorFlowService;
    /**
     * Model name
     */
    name: string;
};

export interface TensorFlowNodeOptions<In extends DataFrame, Out extends DataFrame> extends ProcessingNodeOptions {
    /**
     * TensorFlow model to use for processing.
     */
    model?: TensorFlowServiceModel;
    /**
     * File or URL to load the model from.
     */
    fileOrUrl?: string;
    /**
     * Pre processing function to transform a data frame to a tensor.
     * When undefined the node is considered to be chained to another tensor flow node.
     */
    preProcessing?: TFPreProcessingFn<In>, 
    /**
     * Post processing function to transform a tensor to a data frame.
     * When undefined the node is considered to be chained to another tensor flow node.
     */
    postProcessing?: TFPostProcessingFn<In, Out>, 
    /**
     * TensorFlow options.
     */
    tensorFlow?: TensorFlowOptions;
}
