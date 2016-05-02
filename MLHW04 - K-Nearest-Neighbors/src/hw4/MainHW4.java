package hw4;

import java.io.*;

import weka.core.Instances;

public class MainHW4 {

    public static BufferedReader readDataFile(String filename) {
        BufferedReader inputReader = null;

        try {
            inputReader = new BufferedReader(new FileReader(filename));
        } catch (FileNotFoundException ex) {
            System.err.println("File not found: " + filename);
        }

        return inputReader;
    }

    /**
     * Sets the class index as the last attribute.
     *
     * @param fileName
     * @return Instances data
     * @throws IOException
     */
    public Instances loadData(String fileName) throws IOException {
        BufferedReader datafile = readDataFile(fileName);

        Instances data = new Instances(datafile);
        data.setClassIndex(data.numAttributes() - 1);
        return data;
    }

    public static void main(String[] args) throws Exception {

        System.out.println("Working Directory = " +
                System.getProperty("user.dir"));

        // 1. Loads data
        MainHW4 homework = new MainHW4();
        Instances glassData = homework.loadData("glass.txt");
        Instances cancerData = homework.loadData("cancer.txt");

        // 2. Prints the cross validation error with the best parameters for the both -

        // 2.1. Glass data
        // We remove the useless ID attribute.
        glassData.deleteAttributeAt(0);


        double[] glassBestParams = homework.findBestParameters(glassData);

        System.out.printf("Cross validation error with K = %d" +
                " p = %d, vote function = %s " +
                "for glass data is: %f\n",
                (int)glassBestParams[0],
                (int)glassBestParams[1],
                glassBestParams[2] == 0? "weighted" : "uniform",
                glassBestParams[3]);



        // 2.2. Cancer data
        double[] cancerBestParams = homework.findBestParameters(cancerData);

        System.out.printf("Cross validation error with K = %d" +
                        " p = %d, vote function = %s " +
                        "for cancer data is: %f\n",
                (int)cancerBestParams[0],
                (int)cancerBestParams[1],
                cancerBestParams[2] == 0? "weighted" : "uniform",
                cancerBestParams[3]);

        // 3. Compares the three Edited-Knn algorithms,
        // by calculating their error and measure the average elapsed time it takes to do so, excluding training time.
        // System.nanoTime();

        Knn knn = new Knn();
        knn.setParameters((long)glassBestParams[0], glassBestParams[1], glassBestParams[2]);
        knn.setM_MODE("none");
        double[] result = knn.crossValidationError(glassData);
        System.out.printf("Cross validation error of non-edited knn on glass dataset is " +
                "%f and the average elapsed time is %f\n",
                result[0], result[1]);

        knn = new Knn();
        knn.setParameters((long)glassBestParams[0], glassBestParams[1], glassBestParams[2]);
        knn.setM_MODE("forward");
        result = knn.crossValidationError(glassData);
        System.out.printf("Cross validation error of forwards-edited knn on glass dataset is " +
                        "%f and the average elapsed time is %f\n",
                result[0], result[1]);

        knn = new Knn();
        knn.setParameters((long)glassBestParams[0], glassBestParams[1], glassBestParams[2]);
        knn.setM_MODE("backward");
        result = knn.crossValidationError(glassData);
        System.out.printf("Cross validation error of backwards-edited knn on glass dataset is " +
                        "%f and the average elapsed time is %f\n",
                result[0], result[1]);


    }

    /**
     * Checks k = 1,...,30 , p = 1, 2, 3, infinity and both getClassVoteResult() - 0 and getWeightedClassVoteResult() - 1,
     * to find the parameters that give the lowest cross validation error and returns them in an array.
     *
     * @param data to find best parameters for.
     * @return k, p, votingMethod with lowest cross validation error found.
     * @throws Exception
     */
    private double[] findBestParameters(Instances data) throws Exception {

        double[] bestParams = new double[4];
        bestParams[3] = Double.MAX_VALUE;
        Knn knn = new Knn();

        for (int k = 1; k <= 30; k++) {
            for (double p = 1; p <= 4; p++) {
                for (double votingMethod = 0; votingMethod <= 1; votingMethod++) {

                    // Goes through p = 1, 2, 3 and infinity
                    p = (p == 4) ? Double.MAX_VALUE : p;

                    //System.out.printf("New iteration: K=%d, P=%f, Voting=%f\n", k, p, votingMethod);

                    // Calculates best k and p using votingMethod
                    knn.setParameters(k, p, votingMethod);

                    // Sets best parameters to those with lowest error
                    double crossValidationError = knn.crossValidationError(data)[0];
                    if (crossValidationError < bestParams[3]) {
                        bestParams[0] = k;
                        bestParams[1] = p;
                        bestParams[2] = votingMethod;
                        bestParams[3] = crossValidationError;
                    }
                }
            }
        }
        return bestParams;
    }

}
