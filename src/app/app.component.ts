import { Component } from '@angular/core';
import * as tf from '@tensorflow/tfjs';
import { Sequential, History, Tensor, Rank } from '@tensorflow/tfjs';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.scss']
})
export class AppComponent {
    trainingResults: History;
    model: Sequential;
    data: Car[];
    tensorData: TensorData;

    async runTraining() {
        this.data = await this.getCars();
        this.model = this.createModel();

        this.tensorData = this.convertToTensor(this.data);
        this.trainingResults = await this.trainModel(this.model, this.tensorData.inputs, this.tensorData.labels)

        console.log('Training Results:');
        console.log(this.trainingResults);
        console.log('');
    }

    runTesting() {
        this.testModel(this.model, this.data, this.tensorData);
    }

    async getCars(): Promise<Car[]> {
        let request = await fetch('https://storage.googleapis.com/tfjs-tutorials/carsData.json');
        let carData: any[] = await request.json();
        let cars: Car[] = carData.map(car => {
            let newCar = new Car();
            newCar.mpg = car.Miles_per_Gallon;
            newCar.horsepower = car.Horsepower;
            return newCar;
        })
        .filter(car => (car.mpg !== null && car.horsepower !== null));

        return cars;
    }

    createModel(): Sequential {
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

        return model;
    }

    convertToTensor(data: Car[]): TensorData {
        return tf.tidy(() => {
            tf.util.shuffle(data);

            let inputs = data.map(d => d.horsepower);
            let labels = data.map(d => d.mpg);

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
                inputMax,
                inputMin,
                labelMax,
                labelMin,
            }
        })
    }

    async trainModel(model: Sequential, inputs: Tensor<Rank>, labels: Tensor<Rank>): Promise<History> {
        model.compile({
            optimizer: tf.train.adam(),
            loss: tf.losses.meanSquaredError,
            metrics: ['mse']
        });

        let batchSize = 28;
        let epochs = 50;

        return await model.fit(inputs, labels, {
            batchSize,
            epochs,
            shuffle: true
        })
    }

    testModel(model: Sequential, inputData: Car[], normalizationData: TensorData) {

        let data = tf.tidy(() => {
            let xs = tf.linspace(0, 1, 100);
            let preds = model.predict(xs.reshape([100, 1]));

            if(!(preds instanceof Array)) {
                let unNormXs = xs
                    .mul(normalizationData.inputMax.sub(normalizationData.inputMin))
                    .add(normalizationData.inputMin);

                let unNormPreds = preds
                    .mul(normalizationData.labelMax.sub(normalizationData.labelMin))
                    .add(normalizationData.labelMin);

                return {
                    xs: unNormXs.dataSync(), 
                    preds: unNormPreds.dataSync()
                };
            }
        });

        let predictedPoints = Array.from(data.xs).map((val, i) => {
            return {x: val, y: data.preds[i]};
        });

        let originalPoints = inputData.map(d => ({
            x: d.horsepower, y: d.mpg
        }));

        console.log('Predicted Points:');
        console.log(predictedPoints);
        console.log('');
        console.log('Original Points');
        console.log(originalPoints);
    }
}

export class Car {
    horsepower: number;
    mpg: number
}

export class TensorData {
    inputs: Tensor<Rank>;
    labels: Tensor<Rank>;
    inputMax: Tensor<Rank>;
    inputMin: Tensor<Rank>;
    labelMax: Tensor<Rank>;
    labelMin: Tensor<Rank>;
}
