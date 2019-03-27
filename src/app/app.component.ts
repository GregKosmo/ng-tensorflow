import { Component, OnInit } from '@angular/core';
import * as tf from '@tensorflow/tfjs';
import { Sequential, Tensor, Rank, History } from '@tensorflow/tfjs';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.scss']
})
export class AppComponent implements OnInit {
    model: Sequential;
    sourceData: any[];
    dataProperties: string[];
    dataLoaded: boolean;
    numberOfSourceRecords: number;
    modelTaught: boolean;

    learnProperty: string;
    estimateProperty: string;
    testBatchSize = 28;
    testIterations = 50;
    normalizedInputs: Tensor<Rank>;
    normalizedLabels: Tensor<Rank>;
    inputMax: Tensor<Rank>;
    inputMin: Tensor<Rank>;
    labelMax: Tensor<Rank>;
    labelMin: Tensor<Rank>;

    ngOnInit() {
        this.createModel();
    }

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
        this.setTensors(this.sourceData);
        await this.trainModel();
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

    setTensors(data: any[]): any {
        return tf.tidy(() => {
            tf.util.shuffle(data);

            let inputs = data.map(d => d[this.learnProperty]);
            let labels = data.map(d => d[this.estimateProperty]);

            let inputTensor = tf.tensor2d(inputs, [inputs.length, 1]);
            let labelTensor = tf.tensor2d(labels, [labels.length, 1]);

            this.inputMax = inputTensor.max();
            this.inputMin = inputTensor.min();
            this.labelMax = labelTensor.max();
            this.labelMin = labelTensor.min();
            this.normalizedInputs = inputTensor.sub(this.inputMin).div(this.inputMax.sub(this.inputMin));
            this.normalizedLabels = labelTensor.sub(this.labelMin).div(this.labelMax.sub(this.labelMin));
        })
    }

    async trainModel(): Promise<History> {
        this.model.compile({
            optimizer: tf.train.adam(),
            loss: tf.losses.meanSquaredError,
            metrics: ['mse']
        });

        return await this.model.fit(this.normalizedInputs, this.normalizedLabels, {
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
                    .mul(this.inputMax.sub(this.inputMin))
                    .add(this.inputMin);

                let unNormPreds = preds
                    .mul(this.labelMax.sub(this.labelMin))
                    .add(this.labelMin);

                return {
                    xs: unNormXs.dataSync(), 
                    preds: unNormPreds.dataSync()
                };
            }
        });

        let predictedPoints = Array.from(data.xs).map((val, i) => {
            return {x: val, y: data.preds[i]};
        });
    }
}
