package hw4;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;

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
        double[] bestParams = homework.findBestParameters(glassData);

//		System.out.println("Best parameters for glass.txt are: ");
//		System.out.println("1. k = " + bestParams[0]);
//		System.out.println("2. p = " + bestParams[1]);
//		System.out.println("3. votingMethod = " + bestParams[2]);
//		System.out.println("With lowest Cross Validation Error of " + bestParams[3]);

        // 2.2. Cancer data
        bestParams = homework.findBestParameters(cancerData);

        System.out.println("Best parameters for cancer.txt are: ");
        System.out.println("1. k = " + bestParams[0]);
        System.out.println("2. p = " + bestParams[1]);
        System.out.println("3. votingMethod = " + bestParams[2]);
        System.out.println("With lowest Cross Validation Error of " + bestParams[3]);

        // 3. Compares the three Edited-Knn algorithms,
        // by calculating their error and measure the average elapsed time it takes to do so, excluding training time.
        // System.nanoTime();

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
        Knn knn = new Knn();

        for (int k = 1; k <= 30; k++) {
            for (double p = 1; p <= 4; p++) {
                for (double votingMethod = 0; votingMethod <= 1; votingMethod++) {

                    // Goes through p = 1, 2, 3 and infinity
                    p = (p == 4) ? Double.MAX_VALUE : p;

                    // Calculates best k and p using votingMethod
                    knn.setParameters(k, p, votingMethod);
                    knn.buildClassifier(data);

                    // Sets best parameters to those with lowest error
                    if (knn.crossValidationError(data) > bestParams[3]) {
                        bestParams[3] = knn.crossValidationError(data);
                        bestParams[0] = k;
                        bestParams[1] = p;
                        bestParams[2] = votingMethod;
                    }
                }
            }
        }
        return bestParams;
    }

}
