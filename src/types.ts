import { DataFrame } from "@openhps/core";
import * as tf from '@tensorflow/tfjs';

export type TFPreProcessingFn<T extends DataFrame> = (frame: T) => tf.Tensor[];
export type TFPostProcessingFn<In extends DataFrame, Out extends DataFrame> = (frame: In, output: tf.Tensor[] | tf.Tensor) => Out;

/**
 * TensorFlow options.
 */
export interface TensorFlowOptions {
    /**
     * Model path or URL.
     */
    model?: string;
    /**
     * Training options.
     */
    training?: {
        /**
         * Pre-processing function.
         */
        preProcessing: TFPreProcessingFn<any>;
    } & tf.ModelFitArgs;
    /**
     * Prediction options.
     */
    prediction?: tf.ModelPredictConfig;
}
