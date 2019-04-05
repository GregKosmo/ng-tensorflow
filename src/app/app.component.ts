import { Component, OnInit } from '@angular/core';
import * as tf from '@tensorflow/tfjs';
import { Sequential, History, Tensor } from '@tensorflow/tfjs';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.scss']
})
export class AppComponent {
    model: Sequential;
    sourceData: any[];
    dataProperties: string[];
    dataLoaded: boolean;
    numberOfSourceRecords: number;
    modelTaught: boolean;
    modelLoading: boolean;
    tensorData: TensorData;

    learnProperty: string;
    estimateProperty: string;
    testBatchSize = 28;
    testIterations = 50;

    readSource(sourceFiles: File[]) {
        let reader = new FileReader();
        reader.addEventListener('load', () => {
            if(typeof reader.result === 'string') {
                this.sourceData = JSON.parse(reader.result);
                this.numberOfSourceRecords = this.sourceData.length;

                this.dataProperties = [];
                for(let obj of this.sourceData) {
                    for(let prop in obj) {
                        if(this.dataProperties.indexOf(prop) === -1) {
                            this.dataProperties.push(prop);
                        }
                    }
                }
            }

            this.dataLoaded = true;
        });
        reader.readAsText(sourceFiles[0]);
    }

    async teachModel() {
        this.modelTaught = false;
        this.modelLoading = true;
        this.createModel();
        this.tensorData = this.convertToTensor(this.sourceData);
        await this.trainModel();
        this.modelLoading = false;
        this.modelTaught = true;
    }

    createModel() {
        let model = tf.sequential();
        model.add(tf.layers.dense({
            inputShape: [1], 
            units: 1, 
            useBias: true
        }));

        model.add(tf.layers.dense({
            units: 1,
            useBias: true
        }));

        this.model = model;
    }

    convertToTensor(data: any[]): TensorData {
        return tf.tidy(() => {
            tf.util.shuffle(data);

            let inputs = data.map(d => d[this.learnProperty]);
            let labels = data.map(d => d[this.estimateProperty]);

            let inputTensor = tf.tensor2d(inputs, [inputs.length, 1]);
            let labelTensor = tf.tensor2d(labels, [labels.length, 1]);

            let inputMax = inputTensor.max();
            let inputMin = inputTensor.min();
            let labelMax = labelTensor.max();
            let labelMin = labelTensor.min();
            let normalizedInputs = inputTensor.sub(inputMin).div(inputMax.sub(inputMin));
            let normalizedLabels = labelTensor.sub(labelMin).div(labelMax.sub(labelMin));

            return {
                inputs: normalizedInputs,
                labels: normalizedLabels,
                inputMax: inputMax,
                inputMin: inputMin,
                labelMax: labelMax,
                labelMin: labelMin,
            }
        })
    }

    trainModel(): Promise<History> {
        this.model.compile({
            optimizer: tf.train.adam(),
            loss: tf.losses.meanSquaredError,
            metrics: ['mse']
        });

        return this.model.fit(this.tensorData.inputs, this.tensorData.labels, {
            batchSize: this.testBatchSize,
            epochs: this.testIterations,
            shuffle: true
        })
    }

    testModel() {
        let data = tf.tidy(() => {
            let xs = tf.linspace(0, 1, 100);
            let preds = this.model.predict(xs.reshape([100, 1]));

            if(!(preds instanceof Array)) {
                let unNormXs = xs
                    .mul(this.tensorData.inputMax.sub(this.tensorData.inputMin))
                    .add(this.tensorData.inputMin);

                let unNormPreds = preds
                    .mul(this.tensorData.labelMax.sub(this.tensorData.labelMin))
                    .add(this.tensorData.labelMin);

                return {
                    xs: unNormXs.dataSync(), 
                    preds: unNormPreds.dataSync()
                };
            }
        });

        let predictedPoints = Array.from(data.xs).map((val, i) => {
            return {x: val, y: data.preds[i]};
        });
        console.log(predictedPoints);
    }
}

class TensorData {
    inputs: Tensor;
    labels: Tensor;
    inputMax: Tensor;
    inputMin: Tensor;
    labelMax: Tensor;
    labelMin: Tensor;
}
