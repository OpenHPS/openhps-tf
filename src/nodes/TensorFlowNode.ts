import { DataFrame, ProcessingNode, ProcessingNodeOptions, PushOptions } from "@openhps/core";
import * as tf from '@tensorflow/tfjs';

export class TensorFlowNode<In extends DataFrame, Out extends DataFrame>
    extends ProcessingNode<In, Out> {
    protected options: TensorFlowOptions;
    layersModel: tf.LayersModel;
    protected preProcessing: TFPreProcessingFn<In>;
    protected postProcessing: TFPostProcessingFn<In, Out>;
    isTraining: boolean = false;

    constructor(
        preProcessing: TFPreProcessingFn<In>, 
        postProcessing: TFPostProcessingFn<In, Out>, 
        options?: TensorFlowOptions) {
        super(options);

        // Default options
        this.options.prediction = this.options.prediction ?? {};

        this.preProcessing = preProcessing;
        this.postProcessing = postProcessing;
    }

    loadModel(fileOrUrl: string): Promise<void> {
        return new Promise((resolve, reject) => {
            tf.loadLayersModel(fileOrUrl, {

            }).then(model => {
                this.layersModel = model;
                resolve();
            }).catch(reject);
        });
    }

    saveModel(fileOrUrl: string): Promise<void> {
        return new Promise((resolve, reject) => {
            this.layersModel.save(fileOrUrl).then(value => {

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

export type TFPreProcessingFn<T extends DataFrame> = (frame: T) => tf.Tensor[];
export type TFPostProcessingFn<In extends DataFrame, Out extends DataFrame> = (frame: In, output: tf.Tensor[] | tf.Tensor) => Out;

export interface TensorFlowOptions extends ProcessingNodeOptions {
    model?: string;
    training?: {
        preProcessing: TFPreProcessingFn<any>;
    } & tf.ModelFitArgs;
    prediction?: tf.ModelPredictConfig;
}
