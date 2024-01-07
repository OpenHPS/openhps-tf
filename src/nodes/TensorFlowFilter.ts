import { DataFrame, DataObject, FilterProcessingNode, FilterProcessingOptions } from "@openhps/core";

export class TensorFlowFilter<InOut extends DataFrame> extends FilterProcessingNode<InOut> {

    initFilter(object: DataObject, frame: InOut, options?: FilterProcessingOptions): Promise<any> {
        throw new Error("Method not implemented.");
    }
    
    filter(object: DataObject, frame: InOut, filter: any, options?: FilterProcessingOptions): Promise<DataObject> {
        throw new Error("Method not implemented.");
    }

}
