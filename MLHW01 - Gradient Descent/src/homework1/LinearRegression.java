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
        trainingData.setClassIndex(trainingData.numAttributes() - 1);

        m_ClassIndex = trainingData.classIndex();

        //since class attribute is also an attribuite we subtract 1
        m_truNumAttributes = trainingData.numAttributes() - 1;
        setAlpha(trainingData);
        m_coefficients = gradientDescent(trainingData);

        // Print coefficients
        for (int i = 0; i < m_coefficients.length; i++) {
            System.out.println(m_coefficients[i]);
        }

        // Print error
        System.out.println(calculateSE(trainingData, m_coefficients));

        // TODO: scale features, normalize mean, choose learning rate

    }

    // Choose learning rate alpha:
    private void setAlpha(Instances trainingData){

        System.out.println("Setting Alpha....");

        double[] trialCoefficients = null;
        double minError = 100;
        int pow = 5;

//        // 1. Try different values for alpha
//        for (int i = -17; i < 2; i++) {
//
//            m_alpha = Math.pow(3, i);
//
//            try {
//
//            for (int j = 0; j < 20000; j++) {
//                trialCoefficients = gradientDescent(trainingData);
//            }
//
//            double error = calculateSE(trainingData, trialCoefficients);
//            System.out.println("Tried with alpha 3^" + i + ", received error: " + error);
//
//            if (error < minError) {
//                minError = error;
//                pow = i;
//            }
//
//            } catch (Exception e) {
//                System.out.println("Exception in setAlpha: ");
//                e.printStackTrace();
//            }
//
//        }

        m_alpha = Math.pow(3, pow);

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
            coefficients[i] = Math.random();
        }

        // Improvement rate measurement variables
        double prevAvgSqrErr = 0;
        double curAvgSqrErr = calculateSE(trainingData, coefficients);
        double improvement = Math.abs(curAvgSqrErr - prevAvgSqrErr);

        for (int counter = 1; improvement > 0.003 ; counter++) {

            double[] tempCoefficients = new double[coefficients.length];

            // 2. Calculates new weights according to the gradient of the error function
            for (int i = 0; i < tempCoefficients.length; i++) {
                double partialDerivative = calculatePartialDerivative(trainingData, coefficients, i);
                tempCoefficients[i] = coefficients[i] - (m_alpha * partialDerivative);
            }

            // 3. Updates weights accordingly
            for (int i = 0; i < coefficients.length; i++) {
                coefficients[i] = tempCoefficients[i];
            }

            // 4. Calculates average squared error every 100 iterations to check improvement rate
            if (counter % 100 == 0) {
                prevAvgSqrErr = curAvgSqrErr;
                curAvgSqrErr = calculateSE(trainingData, coefficients);
                improvement = Math.abs(curAvgSqrErr - prevAvgSqrErr);
            }
        }

        return coefficients;
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

            double innerSum = coefficients[0]; // Holds theta * x

            // Add products of all coefficients and all attributes (but the Class Attribute) of current instance
            for (int k = 1; k < trainingData.numAttributes(); k++) {
                innerSum += coefficients[k] * trainingData.instance(j).value(k - 1);
            }

            // Measures difference from actual value
            innerSum -= trainingData.instance(j).classValue();

            // Doubles by the weight that the derivative is calculated by
            innerSum *= trainingData.instance(j).value(i);

            // Adds calculation of current instance to general sum
            sum += innerSum;

        }

        return sum / trainingData.numInstances();
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

        double result = -1;

        for (int i = 1; i < instance.numAttributes(); i++) {
            result += coefficients[i] * instance.value(i - 1);
        }

        return result;
    }

    /**
     * Calculates the total squared error over the test data on a linear regression
     * predictor with weights given by coefficients.
     * @param data
     * @param coefficients
     * @return
     * @throws Exception
     */
    public double calculateSE(Instances data, double[] coefficients) throws Exception {

        int sum = 0;
        Instance cur;

        for (int i = 0; i < data.numAttributes(); i++) {
            cur = data.instance(i);
            sum += Math.pow(Math.abs(cur.classValue() - regressionPrediction(cur, coefficients)), 2);
        }

        return sum / data.numAttributes();
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
