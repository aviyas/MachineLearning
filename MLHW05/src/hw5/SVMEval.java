package hw5;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;


import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.Kernel;
import weka.classifiers.functions.supportVector.PolyKernel;
import weka.classifiers.functions.supportVector.RBFKernel;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

public class SVMEval {

    private SMO smo;

    public SVMEval() {
        this.smo = new SMO();
    }

    void buildClassifier(Instances instances) throws Exception {
        this.smo.buildClassifier(instances);
    }

    /**
     * Chooses the best kernel
     * Action: chooses best kernel from the options: 13 possible RBF kernels with parameter gamma 2^i for -10<=i<=2,
     * 3 possible polynomial kernels of degree i for 2<=i<=4.
     * The chosen kernel is the kernel that yields the best cross validation error for the SVM model that uses it.
     Output: the method sets the kernel of your SVM classifier.
     */
    void chooseKernel(Instances instances) throws Exception {
        Kernel bestKernel = null;
        double minError = Double.MAX_VALUE;
        // RBF kernel checking.
        for (int i = -10; i <= 2; i++) {
            RBFKernel rbfKernel = new RBFKernel();
            rbfKernel.setGamma(Math.pow(2, i));
            this.smo.setKernel(rbfKernel);
            double error = this.calcCrossValidationError(instances);
            if (error < minError) {
                minError = error;
                bestKernel = rbfKernel;
            }
        }

        // Poly kernel checking.
        for (int i = 2; i <= 4 ; i++) {
            PolyKernel polyKernel = new PolyKernel();
            polyKernel.setExponent(i);
            this.smo.setKernel(polyKernel);
            double error = this.calcCrossValidationError(instances);
            if (error < minError) {
                minError = error;
                bestKernel = polyKernel;
            }
        }

        this.smo.setKernel(bestKernel);
    }

    private ArrayList<Integer> attributesToRemove = new ArrayList<>();

    Instances removeAttributes(Instances instances) throws Exception {
        return removeIndexes(instances, attributesToRemove.stream().mapToInt(i -> i).toArray());
    }

    /**
     * performs the backwards wrapper feature selection algorithm as explained above.
     *
     * @param threshold
     * @param minAttributeNum
     * @param instances
     * @return New Instances object with the chosen subset of the original features, indices of chosen features.
     */
    Instances backwardsWrapper(double threshold, int minAttributeNum, Instances instances) throws Exception {

        double errorDiff = 0;
        double originalError = calcCrossValidationError(instances);
        Instances resultInstances = new Instances(instances);

        while (resultInstances.numAttributes() - 1 > minAttributeNum && errorDiff < threshold) {
            int minimalIndex = 1;
            Instances innerInstances = removeIndexes(resultInstances, 1);
            Instances minimalInstances = innerInstances;
            double minimalError = calcCrossValidationError(innerInstances);

            for (int i = 2; i < resultInstances.numAttributes() - 1; i++) {
                innerInstances = removeIndexes(resultInstances, i);
                double newError = calcCrossValidationError(innerInstances);
                if (newError < minimalError) {
                    minimalError = newError;
                    minimalIndex = i;
                    minimalInstances = innerInstances;
                }
            }

            errorDiff = minimalError - originalError;
            if (errorDiff < threshold) {
                this.attributesToRemove.add(instances.attribute(resultInstances.attribute(minimalIndex).name()).index());
                resultInstances = minimalInstances;
            }
        }

        return resultInstances;
    }

    private Instances removeIndexes(Instances instances, int... indexes) throws Exception {
        Remove remove = new Remove();
        remove.setAttributeIndicesArray(indexes);
        remove.setInvertSelection(false);
        remove.setInputFormat(instances);
        return Filter.useFilter(instances, remove);
    }

    /**
     * @param instances
     * @return The number of prediction mistakes the classifier makes divided by the number of instances.
     */
    double calcAvgError(Instances instances) throws Exception {
        double mistakes = 0;
        for (int i = 0; i < instances.numInstances(); i++) {
            Instance instance = instances.instance(i);
            mistakes += this.smo.classifyInstance(instance) == instance.classValue()? 0 : 1;
        }

        return mistakes / instances.numInstances();
    }

    private int FOLD_NUM = 3;

    /**
     * @param instances
     * @return The cross validation estimate of the avg error.
     */
    double calcCrossValidationError(Instances instances) throws Exception {
        Instances randInstances = new Instances(instances);
        randInstances.randomize(new Random());

        double totalError = 0;
        Instances[] splitData;
        for (int i = 0; i < FOLD_NUM; i++) {
            splitData = splitDataBy(randInstances, i);
            Instances learning = splitData[0];
            Instances validation = splitData[1];

            buildClassifier(learning);
            double error = calcAvgError(validation);
            totalError += error;
        }

        return totalError / FOLD_NUM;
    }

    /**
     * Splits the dataset into 5 subsets by index i.
     *
     * @param dataset to split.
     * @param index   of fold to set as validation.
     * @return an array where arr[0] = 4 learning folds and arr[1] = 1 validation fold.
     */
    private Instances[] splitDataBy(Instances dataset, int index) {

        Instances[] splitData = new Instances[2];

        int foldSize = dataset.numInstances() / FOLD_NUM;
        int splitIndex = foldSize * index;

        // 1. Copies the relevant validation section
        splitData[1] = new Instances(dataset, splitIndex, foldSize);

        // 2. Copies the relevant training sections:
        // 2.1. Pre validation set
        splitData[0] = new Instances(dataset, 0, splitIndex);

        // 2.2. Post validation set
        for (int i = splitIndex + foldSize; i < dataset.numInstances(); i++) {
            splitData[0].add(dataset.instance(i));
        }
        return splitData;
    }

    public static BufferedReader readDataFile(String filename) {
        BufferedReader inputReader = null;

        try {
            inputReader = new BufferedReader(new FileReader(filename));
        } catch (FileNotFoundException ex) {
            System.err.println("File not found: " + filename);
        }

        return inputReader;
    }

    public Instances loadData(String fileName) throws IOException {
        BufferedReader dataFile = readDataFile(fileName);
        Instances instances = new Instances(dataFile);
        instances.setClassIndex(0);
        return instances;
    }

    public static void main(String[] args) throws Exception {
        // Loads data.
        SVMEval homework = new SVMEval();
        Instances testData = homework.loadData("ElectionsData_test.txt");
        Instances trainData = homework.loadData("ElectionsData_train.txt"); // validation

        // Choose the best kernel method.
        homework.chooseKernel(trainData);

        // Perform features selection using backwardWrapper.
        double threshold = 0.05;
        int minNumberOfFeatures = 5;
        Instances trainDataChosenFeatures = homework.backwardsWrapper(threshold, minNumberOfFeatures, trainData);

        homework.buildClassifier(trainDataChosenFeatures);

        // Moving to test data.
        // Leave only selected features.
        Instances testDataChosenFeatures = homework.removeAttributes(testData);

        // Calculate average error using the trained classifier.
        double avgError = homework.calcAvgError(testDataChosenFeatures);
        System.out.println(avgError);
    }
}
