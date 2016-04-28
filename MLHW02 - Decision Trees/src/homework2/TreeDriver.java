package homework2;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.Enumeration;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import weka.attributeSelection.ChiSquaredAttributeEval;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

public class TreeDriver {

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
     * @param fileName
     * @return Instances data
     * @throws IOException
     */
    public Instances loadData(String fileName) throws IOException{
        BufferedReader datafile = readDataFile(fileName);

        Instances data = new Instances(datafile);
        data.setClassIndex(data.numAttributes() - 1);
        return data;
    }


    public static void main(String[] args) throws Exception {

        // Load data
        TreeDriver homework = new TreeDriver();
        Instances training = homework.loadData("cancer_train.txt");
        Instances testing = homework.loadData("cancer_test.txt");

//        Instances training = homework.loadData("mushrooms_training.txt");
//        Instances testing = homework.loadData("mushroom_test.txt");

        // Train classifier
        DecisionTree decisionTree = new DecisionTree();
        decisionTree.buildClassifier(training);

        Stream<String> lines = Stream.of("");

        double averageErrorTrain = decisionTree.calcAvgError(training);
        System.out.printf("Average Error, Training: %f\n" ,averageErrorTrain);
        lines = Stream.concat(lines,
                Stream.of(String.format("The average train error of the decision tree is %f", averageErrorTrain)));

        double averageErrorTest = decisionTree.calcAvgError(testing);
        System.out.printf("Average Error, Testing: %f\n" ,averageErrorTest);
        lines = Stream.concat(lines,
                Stream.of(String.format("The average test error of the decision tree is %f", averageErrorTest)));

        // Calculates error with pruning and displays in "hw02.txt"
        decisionTree.setPruningMode(true);
        //decisionTree.setThreshold(11.591);
        decisionTree.postPruneTree();

        averageErrorTrain = decisionTree.calcAvgError(training);
        System.out.printf("Average Error (pruning), Training: %f\n" ,averageErrorTrain);
        lines = Stream.concat(lines,
                Stream.of(String.format("The average train error of the decision tree with pruning is %f", averageErrorTrain)));

        averageErrorTest = decisionTree.calcAvgError(testing);
        System.out.printf("Average Error (pruning), Testing: %f\n" ,averageErrorTest);
        lines = Stream.concat(lines,
                Stream.of(String.format("The average test error of the decision tree with pruning is %f", averageErrorTest)));

        try {
            Files.write(Paths.get("hw2.txt"), lines.collect(Collectors.toList()));
        } catch (IOException e) {
            e.printStackTrace();
        }

    }
}