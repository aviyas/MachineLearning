package homework1;

import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;

import java.util.Arrays;

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
        trainingData.setClassIndex(trainingData.numAttributes() - 1);

        m_ClassIndex = trainingData.classIndex();

        //since class attribute is also an attribuite we subtract 1
        m_truNumAttributes = trainingData.numAttributes() - 1;

        System.out.println("Calculating alpha.");
        setAlpha(trainingData);
        System.out.printf("Alpha = %f\n.", m_alpha);


        m_coefficients = new double[m_truNumAttributes + 1];
        m_coefficients = gradientDescent(trainingData);

        // Print coefficients
        System.out.println("Printing coefficients:");
        for (int i = 0; i < m_coefficients.length; i++) {
            System.out.println(" tetha_" + i + ": " + m_coefficients[i]);
        }

        // TODO: scale features, normalize mean

    }

    private static final int ITERATIONS_NUM = 20000;

    // Choose learning rate alpha:
    private void setAlpha(Instances trainingData) throws Exception {
        double minError = Double.MAX_VALUE;
        double[] coefficients = new double[m_truNumAttributes + 1];

        double alpha = Double.MIN_VALUE;

        for (int i = 0; i < coefficients.length; i++) {
            coefficients[i] = 1;
        }

        for (int i = -17; i < 2; i++)
        {
            m_alpha = Math.pow(3, i);

            for (int j = 0; j < ITERATIONS_NUM; j++) {
                coefficients = gradientDecentIteration(coefficients, trainingData);
            }

            double error = calculateSE(trainingData, coefficients);

            if (error < minError) {
                minError = error;
                alpha = m_alpha;
            }
        }

        m_alpha = alpha;
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

        double[] coefficients = new double[m_truNumAttributes + 1];

        // 1. Guess the weights
        for (int i = 0; i < coefficients.length; i++) {
            coefficients[i] = 1;
        }

        // Improvement rate measurement variables
        double prevAvgSqrErr = 0;
        double curAvgSqrErr = calculateSE(trainingData, coefficients);
        double improvement = Math.abs(curAvgSqrErr - prevAvgSqrErr);

        long counter;

        for (counter = 1; improvement > 0.003 ; counter++) {

            // 2. Calculates new weights according to the gradient of the error function
            coefficients = gradientDecentIteration(coefficients, trainingData);

            // 4. Calculates average squared error every 100 iterations to check improvement rate
            if (counter % 100 == 0) {
                prevAvgSqrErr = curAvgSqrErr;
                curAvgSqrErr = calculateSE(trainingData, coefficients);
                improvement = Math.abs(curAvgSqrErr - prevAvgSqrErr);
            }
        }

        System.out.printf("Ran for %d iterations\n", counter);

        return coefficients;
    }

    /**
     * Perform one iteration of gradient decent.
     * @param coefficients
     * @param trainingData
     * @return
     */
    private double[] gradientDecentIteration(double[] coefficients, Instances trainingData) {
        double[] roundCoefficients = new double[m_truNumAttributes + 1];

        for (int i = 0; i < coefficients.length; i++) {
            double partialDerivative = calculatePartialDerivative(trainingData, coefficients, i);
            roundCoefficients[i] = coefficients[i] - m_alpha * partialDerivative;
        }

        return roundCoefficients;
    }

    /**
     * Calculates the derivative of the error function by the current weight.
     * @param trainingData
     * @param coefficients
     * @param i - current weight index.
     * @return
     */
    private double calculatePartialDerivative(Instances trainingData, double[] coefficients, int i) {
        double sum = 0;

        // Goes through all instances
        for (int j = 0; j < trainingData.numInstances(); j++) {

            // Calculate h(x).
            double innerSum = regressionPrediction(trainingData.instance(j), coefficients); // Holds theta * x

            // Measures difference from actual value
            innerSum -= trainingData.instance(j).classValue();

            // Doubles by the weight that the derivative is calculated by (Don't multiple by x0 for the bias).
            if (i > 0) {
                innerSum *= trainingData.instance(j).value(i - 1);
            }

            // Adds calculation of current instance to the general sum.
            sum += innerSum;
        }

        return sum / trainingData.numInstances();
    }

    /**
     * Returns the prediction of a linear regression predictor with weights
     * given by coefficients on a single instance.
     * @param instance
     * @return
     * @throws Exception
     */
    public double regressionPrediction(Instance instance) throws Exception {
        return regressionPrediction(instance, m_coefficients);
    }

    private static double regressionPrediction(Instance instance, double[] coefficients) {
        // Add the bias.
        double result = coefficients[0];

        for (int i = 1; i < instance.numAttributes(); i++) {
            result += coefficients[i] * instance.value(i - 1);
        }

        return result;
    }

    /**
     * Calculates the total squared error over the test data on a linear regression
     * predictor with weights given by coefficients.
     * @param data
     * @return
     * @throws Exception
     */
    public double calculateSE(Instances data) throws Exception {

        return calculateSE(data, m_coefficients);
    }

    private static double calculateSE(Instances data, double[] coefficients) throws Exception {
        if (Arrays.stream(coefficients).anyMatch(Double::isNaN)) {
            return Double.MAX_VALUE;
        }

        int sum = 0;

        for (int i = 0; i < data.numInstances(); i++) {
            Instance cur = data.instance(i);
            sum += Math.pow(cur.classValue() - regressionPrediction(cur, coefficients), 2);
        }

        return sum / (2 * data.numInstances());
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

    public double[] getCoefficients() {
        return m_coefficients;
    }
}