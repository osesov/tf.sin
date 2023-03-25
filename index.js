// @ts-check

const SIN = {
    /** @type {number} */
    argMin: -Math.PI,

    /** @type {number} */
    argMax: +Math.PI,

    /** @type {number} */
    resMin: -1,

    /** @type {number} */
    resMax: +1,

    name: 'Sin',
    /**
     * @param {Tensor1D} value
     * @returns {Tensor1D}
     */
    do: (value) => tf.sin(value)
}

const TAN_LIMIT = 100;

const TAN = {
    /** @type {number} */
    argMin: Math.atan(-TAN_LIMIT), // -Math.PI/4,

    /** @type {number} */
    argMax: Math.atan(+TAN_LIMIT), // +Math.PI/4,

    /** @type {number} */
    resMin: -TAN_LIMIT, // Math.tan(-Math.PI/4),

    /** @type {number} */
    resMax: +TAN_LIMIT, // Math.tan(+Math.PI/4),

    name: 'Tan',

    /**
     * @param {Tensor1D} value
     * @returns {Tensor1D}
     */
    do: (value) => tf.tan(value)
}

const fn = SIN;

/** @typedef Track
 * @property {Tensor1D} x
 * @property {Tensor1D} y
 */

/** @typedef Limits
 * @property {Tensor1D} min
 * @property {Tensor1D} max
 */
/** @typedef SourceDataSet
 * @property {Track} train
 * @property {Track} validation
 * @property {Limits} limitX
 * @property {Limits} limitY
 */

/** @returns {Promise<SourceDataSet>} */
async function getData()
{
    const dataPoints = 1000;
    const validationPoints = 100;

    const dataX = tf.linspace(fn.argMin, fn.argMax, dataPoints);
    // clamp values?
    const dataY = fn.do(dataX);

    const validationX = tf.randomUniform([validationPoints, 1], fn.argMin, fn.argMax);
    // clamp values?
    const validationY = fn.do(validationX);

    /**
     * @param {number|undefined} fnMin
     * @param {number|undefined} fnMax
     * @param {Tensor1D} data
     * @param {Tensor1D} validation
     * @returns {Promise<Limits>}
     */
    async function getLimit(fnMin, fnMax, data, validation)
    {
        return {
            min: tf.tensor1d([fnMin !== undefined ? fnMin : Math.min((await data.min().data())[0], (await validation.min().data())[0])]),
            max: tf.tensor1d([fnMax !== undefined ? fnMax : Math.max((await data.max().data())[0], (await validation.max().data())[0])]),
        }
    }

    return {
        train: {x:dataX, y:dataY },
        validation: { x: validationX, y: validationY},
        limitX: { min: tf.tensor1d([fn.argMin]), max: tf.tensor1d([fn.argMax])},
        limitY: await getLimit(fn.resMin, fn.resMax, dataY, validationY),
    }
}

/**
 * @typedef DataSet
 * @type {object}
 * @property {Tensor1D} x
 * @property {Tensor1D} y
 */

/**
 * @typedef ModelDataSet
 * @property {Sequential} model
 * @property {DataSet} train
 * @property {DataSet} validation
 * @property {Limits} limitX
 * @property {Limits} limitY
 */

/**
 * @param {Tensor1D} data
 * @param {Limits} limits
 * @returns {Tensor1D}
 */
function normalizeTensor(data, limits)
{
    const delta = limits.max.sub(limits.min);
    return data.sub(limits.min).div(delta);
}

/**
 * @param {Tensor1D} data
 * @param {Limits} limits
 * @returns {Tensor1D}
 */
function denormalizeTensor(data, limits)
{
    const delta = limits.max.sub(limits.min);
    return data.mul(delta).add(limits.min);
}

/** @returns {Promise<() => Promise<ModelDataSet>>} */
async function createModel()
{
    // Create a simple model.
    const model = tf.sequential();
    model.add(tf.layers.inputLayer({inputShape: [1]}))
    model.add(tf.layers.dense({units: 50, activation: 'relu', name: 'hidden1'}));
    model.add(tf.layers.dense({units: 1,  activation: 'linear', name: 'output'}));

    // Prepare the model for training: Specify the loss and the optimizer.
    model.compile({
        // sin
        loss: tf.losses.meanSquaredError,
        optimizer: tf.train.adam(),
        // metrics: ['mse'],
        //
        // loss: tf.losses.meanSquaredError,
        // optimizer: 'sgd',
        // metrics: ['accuracy'],
    });

    const surface = { name: 'Model Summary', tab: 'Model Inspection'};
    tfvis.show.modelSummary(surface, model);

    const data = await getData();

    /**
     * @param {Track} track
     * @returns {DataSet}
     */
    function normalize(track)
    {
        const normalizedX = normalizeTensor(track.x, data.limitX);
        const normalizedY = normalizeTensor(track.y, data.limitY);
        return { x: normalizedX, y: normalizedY }
    }

    /**
     *
     * @param {Track} track
     * @returns {Promise<Point2D[]>}
     */
    async function getScatterPoints(track)
    {
        const xs = await track.x.data();
        const ys = await track.y.data();

        return Array.from(xs).map( (value, index) => ({x: value, y: ys[index]}) );
    }

    const train = normalize(data.train);
    const validation = normalize(data.validation);

    const trainPoints = await getScatterPoints(train);
    const validationPoints = await getScatterPoints(validation);

    tfvis.render.scatterplot(
        { name: 'Original Data' },
        { values: [trainPoints, validationPoints], series: ["train", "validation"] },
        {
            xLabel: 'normalized (0..1)',
            yLabel: 'normalized (0..1)',
            height: 300
        }
    );

    // Train the model using the data.
    const batchSize = 32;
    const epochs = 2000;

    const fitCallbacks = tfvis.show.fitCallbacks(
        { name: 'Training Performance', tab: 'Train' },
        [ 'mse', 'loss', 'val_loss', 'acc', 'val_acc'],
        { height: 200, callbacks: ['onEpochEnd'] }
    );

    return async () => {
        const input = await denormalizeTensor(validation.x, data.limitX).data();
        await model.fit(train.x, train.y, {
            batchSize,
            epochs,
            shuffle: true,
            validationData: [validation.x, validation.y],
            callbacks: {
                ... fitCallbacks,
                onEpochEnd: async function(number, logs) {
                    console.log(`End of epoch: ${number}`)

                    const predictedValues = model.predict(validation.x);
                    if (Array.isArray(predictedValues))
                        throw new Error("Unexpected result");

                    // const output = denormalizeTensor(predictedValues, data.limitY).dataSync();
                    // const expectedPoints = Array.from(input).map((value) => ({x: value, y: fn.do(value).dataSync()[0]}))
                    // const predictedPoints = Array.from(input).map( (value, index) => ({x: value, y: output[index]}) );

                    const expectedPoints = await getScatterPoints(validation);
                    const predictedPoints = await getScatterPoints({x: validation.x, y: predictedValues});

                    tfvis.render.scatterplot(
                        { name: 'Training', tab: 'Train' },
                        { values: [expectedPoints, predictedPoints], series: ["expected", "predicted"] },
                        {
                            xLabel: 'normalized (0..1)',
                            yLabel: 'normalized (0..1)',
                            height: 300,
                            xAxisDomain: [0, 1],
                            yAxisDomain: [-0.1, 1.1 ],
                        }
                    );

                    return fitCallbacks.onEpochEnd ? fitCallbacks.onEpochEnd.call(fitCallbacks, number, logs) : undefined;
                }
            }
        });
        console.log("training done");
        return {model, train, validation, limitX: data.limitX, limitY: data.limitY};
    }
}


/**
 *
 * @param {ModelDataSet} data
 */
async function testModel(data)
{
    const { model, train, validation, limitX, limitY } = data;
    const inputValues = tf.randomUniform([100,1], 0, 1);

    /** @type {Tensor<Rank> | Tensor<Rank>[]} */
    const predictedValues = model.predict(inputValues);
    if (Array.isArray(predictedValues))
        throw new Error("Unexpected result");

    const input = await denormalizeTensor(inputValues, limitX).data();
    const output = await denormalizeTensor(predictedValues, limitY).data();
    const expected = Array.from(input).map((value) => ({x: value, y: fn.do(value)}))

    const predictedPoints = Array.from(input).map( (value, index) => ({x: value, y: output[index]}) );
    const expectedPoints = Array.from(output).map( (value, index) => ({x: value, y: expected[index]}) );

    tfvis.render.scatterplot(
        { name: 'Model Predictions vs Original Data' },
        { values: [expectedPoints, predictedPoints], series: ['original', 'predicted'] },
        {
          xLabel: 'normalized (0..1)',
          yLabel: 'normalized (0..1)',
          height: 300
        }
    );

}

async function run()
{
    if (WebAssembly !== undefined) {
        // linspace is not supported?
        // await tf.setBackend('wasm');
    }

    const fn = await createModel();

    $("#run").click( () =>
        setTimeout( async () => {
            const data = await fn();
            testModel(data);
        }, 3000)
    );
}


$( run );
