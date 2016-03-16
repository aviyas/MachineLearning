package homework1;

import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;

public class LinearRegression extends Classifier{

    private int m_ClassIndex;
    private int m_truNumAttributes;
    private double[] m_coefficients;
    private double m_alpha;

    //the method which runs to train the linear regression predictor, i.e.
    //finds its weights.
    @Override
    public void buildClassifier(Instances trainingData) throws Exception {
        trainingData = new Instances(trainingData);
        m_ClassIndex = trainingData.classIndex();
        //since class attribute is also an attribuite we subtract 1
        m_truNumAttributes = trainingData.numAttributes() - 1;
        setAlpha();
        m_coefficients = gradientDescent(trainingData);

    }

    private void setAlpha(){

    }

    /**
     * An implementation of the gradient descent algorithm which should try
     * to return the weights of a linear regression predictor which minimizes
     * the average squared error.
     * @param trainingData
     * @return
     * @throws Exception
     */
    public double[] gradientDescent(Instances trainingData)
            throws Exception {

        double[] coefficients = new double[trainingData.numAttributes()];

        // 1. Guess the weights
        for (int i = 0; i < coefficients.length; i++) {
            coefficients[i] = 1;
        }

        // TODO: calculate error
        double error = 100;

        while (error > 0.03) {

            double[] temp_coefficients = new double[coefficients.length];

            // 2. Calculates error for all weights
            for (int i = 0; i < temp_coefficients.length; i++) {

                double result = calculateSigma(trainingData, coefficients);

                temp_coefficients[i] = coefficients[i] - ((m_alpha * result) / trainingData.numInstances());

            }

            // 3. Updates weights accordingly
            for (int i = 0; i < coefficients.length; i++) {

                error += (coefficients[i] - temp_coefficients[i]);
                coefficients[i] = temp_coefficients[i];

            }

            error /= coefficients.length;
        }

        //

        // 2. For each instance - calculate error rate
            // * error rate = average squared error:
            //   using weights and all features but the last one, calculate the derivative by current weight



        // 3. Update weights

        // Continue if error is still too big.

        //

        return coefficients;
    }

    private double calculateSigma(Instances trainingData, double[] coefficients) {

        double sum = 0;
        trainingData.setClassIndex(trainingData.numAttributes() - 1);

        // Calculates the derivative of the error function by the current weight
        for (int i = 1; i < coefficients.length; i++) {

            // Goes through all instances
            for (int j = 0; j < trainingData.numInstances(); j++) {

                double innerSum = coefficients[0];

                // Add products of all coefficients and all attributes (but the last) of current instance
                for (int k = 0; k < trainingData.numAttributes() - 1; k++) {

                    innerSum += coefficients[i] * trainingData.instance(j).value(k);

                }

                innerSum -= trainingData.instance(j).classValue();
                innerSum *= trainingData.instance(j).value(i);

                sum += innerSum;

            }


        }

        return sum;
    }

    /**
     * Returns the prediction of a linear regression predictor with weights
     * given by coefficients on a single instance.
     * @param instance
     * @param coefficients
     * @return
     * @throws Exception
     */
    public double regressionPrediction(Instance instance, double[] coefficients) throws Exception {
        return 0;
    }

    /**
     * Calculates the total squared error over the test data on a linear regression
     * predictor with weights given by coefficients.
     * @param testData
     * @param coefficients
     * @return
     * @throws Exception
     */
    public double calculateSE(Instances testData, double[] coefficients) throws Exception {

        return 0;
    }

    /**
     * Finds the closed form solution to linear regression with one variable.
     * Should return the coefficient that is to be multiplied
     * by the input attribute.
     * @param data
     * @return
     */
    public double findClosedForm1D(Instances data){
        return 0;
    }

}
