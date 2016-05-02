package hw4;

import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;

import java.util.*;
import java.util.concurrent.ThreadLocalRandom;
import java.util.stream.*;


public class Knn extends Classifier {

    private String M_MODE = "";
    private Instances m_trainingInstances;

    private long k;
    private double p;
    private double votingMethod;

    public String getM_MODE() {
        return M_MODE;
    }

    public void setM_MODE(String m_MODE) {
        M_MODE = m_MODE;
    }

    public void setParameters(long k, double p, double votingMethod) {
        this.k = k;
        this.p = p;
        this.votingMethod = votingMethod;
    }

    @Override
    public void buildClassifier(Instances arg0) throws Exception {
        switch (M_MODE) {
            case "none":
                noEdit(arg0);
                break;
            case "forward":
                editedForward(arg0);
                break;
            case "backward":
                editedBackward(arg0);
                break;
            default:
                noEdit(arg0);
                break;
        }
    }

    /**
     * Trains the algorithm using the Edited-Knn Forwards Algorithm shown in class.
     *
     * @param instances of training data.
     */
    private void editedForward(Instances instances) {
        m_trainingInstances = new Instances(instances, instances.numInstances());

        for (int i = 0; i < instances.numInstances(); i++) {
            Instance instance = instances.instance(i);
            if (classify(instance) != instance.classValue()) {
                m_trainingInstances.add(instance);
            }
        }
    }

    /**
     * Trains the algorithm using the Edited-Knn Backwards Algorithm shown in class.
     *
     * @param instances of training data.
     */
    private void editedBackward(Instances instances) {
        m_trainingInstances =  new Instances(instances);

        for (int i = 0; i < m_trainingInstances.numInstances(); i++) {
            Instance instance = m_trainingInstances.instance(i);
            m_trainingInstances.delete(i);
            if (classify(instance) != instance.classValue()) {
                m_trainingInstances.add(instance);
            }
        }
    }

    /**
     * Trains the algorithm using the Standard Knn Algorithm shown in class, storing all of the instances in memory.
     *
     * @param instances of training data.
     */
    private void noEdit(Instances instances) {
        m_trainingInstances = new Instances(instances);
    }

    /**
     * Predicts the class value of a given instance.
     *
     * @param newInstance to classify.
     * @return the classification.
     */
    public double classify(Instance newInstance) {
        Map<Double, Instance> nearestNeighbors = findNearestNeighbors(newInstance);

        try {
            if (votingMethod == 1) {
                return getWeightedClassVoteResult(nearestNeighbors);
            } else {
                return getClassVoteResult(nearestNeighbors);
            }
        } catch (IllegalStateException e) {
            // The instances list is empty so let choose a random class.
            double c = ThreadLocalRandom.current().nextInt(0, newInstance.numClasses());
            //System.out.printf("Chose random class: %f\n", c);
            return c;
        }
    }

    /**
     * Finds the given k nearest neighbors of a given instance, using the standard method.
     *
     * @param newInstance to be examined.
     * @return k nearest neighbors and their distances.
     */
    public Map<Double, Instance> findNearestNeighbors(Instance newInstance) {
        // Calculates the distance from all instances and put in a TreeMap
        // and finds k mappings with k lowest key values
        //System.out.printf("------- newInstance: %s -----\n", newInstance);
        Map<Double, Instance> collect = Knn.<Instance>enumerationAsStream(m_trainingInstances.enumerateInstances())
                .map(instance -> new AbstractMap.SimpleImmutableEntry<>(distance(instance, newInstance), instance))
                .sorted(Comparator.comparing(Map.Entry::getKey))
                .limit(k)
                //.peek(System.out::println)
                .collect(Collectors.toMap(Map.Entry::getKey, Map.Entry::getValue, (a,b) -> a));
        //System.out.println("------------------");
        return collect;
    }

    /**
     * Takes a vote on what the class of the neighbors are and determines the final result accordingly.
     *
     * @param nearestNeighbors to base the vote on.
     * @return the class value with the most votes.
     */
    public double getClassVoteResult(Map<Double, Instance> nearestNeighbors) {
        return getClassVoteResult(nearestNeighbors,
                Collectors.summingDouble(entry -> 1d));
    }

    /**
     * Takes a vote on what the class of the neighbors are, weighted according with their distances from the newInstance,
     * and determines the final result accordingly.
     *
     * @param nearestNeighbors to base the vote on.
     * @return the class value with the most votes.
     */
    public double getWeightedClassVoteResult(Map<Double, Instance> nearestNeighbors) {
        return getClassVoteResult(nearestNeighbors,
                Collectors.summingDouble(entry -> 1d/Math.pow(entry.getKey(), 2)));
    }

    private double getClassVoteResult(Map<Double, Instance> nearestNeighbors,
                                      Collector<Map.Entry<Double, Instance>, ?, Double> summaryMethod) {
        return nearestNeighbors.entrySet().stream()
                .collect(Collectors.groupingBy(
                        entry -> entry.getValue().classValue(),
                        summaryMethod))
                .entrySet().stream()
                .max(Map.Entry.comparingByValue())
                .orElseThrow(IllegalStateException::new)
                .getKey();
    }

    /**
     * Calculates the distance between two instances, based on the chosen distance function.
     *
     * @param thingOne first instance
     * @param thingTwo second instance
     * @return the distance between the instances.
     */
    public double distance(Instance thingOne, Instance thingTwo) {
        return (p == Double.MAX_VALUE) ?
                lInfinityDistance(thingOne, thingTwo) : lPDistance(thingOne, thingTwo);
    }

    /**
     * Calculates the l-p distance between two instances.
     *
     * @param thingOne first instance
     * @param thingTwo second instance
     * @return the l-p distance between the instances.
     */
    public double lPDistance(Instance thingOne, Instance thingTwo) {

        double distance = 0;

        // Sums all differences at the power of p
        for (int i = 0; i < thingOne.numAttributes() - 1; i++) {
            distance += Math.abs(Math.pow(thingOne.value(i) - thingTwo.value(i), p));
        }

        return root(distance, p);
    }

    /**
     * Calculates the nth root of a number.
     *
     * @param number the value under the root.
     * @param n      the degree of the calculation.
     * @return the nth root of value.
     */
    private double root(double number, double n) {
        return Math.pow(number, 1d / n);
    }

    /**
     * Calculates the l-infinity distance between two instances.
     *
     * @param thingOne first instance
     * @param thingTwo second instance
     * @return the l-infinity distance between the instances.
     */
    public double lInfinityDistance(Instance thingOne, Instance thingTwo) {
        // Takes the maximum difference measured between all attributes
        return IntStream.range(0, thingOne.numAttributes() - 1)
                .mapToDouble(i -> Math.abs(thingOne.value(i) - thingTwo.value(i)))
                .max()
                .orElseThrow(IllegalStateException::new);
    }

    /**
     * Calculates the average error on given instances: # mistakes / # instances.
     *
     * @return the error on the instances.
     */
    public double calcAverageError(Instances instances) {
        double mistakes = Knn.<Instance>enumerationAsStream(instances.enumerateInstances())
                .filter(instance -> classify(instance) != instance.classValue())
                .count();

        return mistakes / instances.numInstances();
    }

    private int FOLD_NUM = 10;

    /**
     * Calculates the Cross Validation Error, using 10 folds.
     *
     * @param dataset of instances.
     * @return the cross validation error on the dataset.
     */
    public double[] crossValidationError(Instances dataset) throws Exception {
        double totalError = 0;
        Instances[] splitData;
        long totalTime = 0;
        for (int i = 0; i < FOLD_NUM; i++) {
            splitData = splitDataBy(dataset, i);
            Instances learning = splitData[0];
            Instances validation = splitData[1];

            buildClassifier(learning);
            long startTime = System.nanoTime();
            double error = calcAverageError(validation);
            totalTime += System.nanoTime() - startTime;
            totalError += error;
        }

        return new double[] {totalError / FOLD_NUM, totalTime / FOLD_NUM};
    }

    /**
     * Splits the dataset into 10 subsets by index i.
     *
     * @param dataset to split.
     * @param index   of fold to set as validation.
     * @return an array where arr[0] = 9 learning folds and arr[1] = 1 validation fold.
     */
    private Instances[] splitDataBy(Instances dataset, int index) {

        Instances[] splitData = new Instances[2];

        int foldSize = dataset.numInstances() / FOLD_NUM;
        int splitIndex = foldSize * index;

        // 1. Copies the relevant validation section
        splitData[1] = new Instances(dataset, splitIndex, foldSize);

        // 2. Copies the relevant training sections:
        // 		2.1. Pre validation set
        splitData[0] = new Instances(dataset, 0, splitIndex);

        // 		2.2. Post validation set
        for (int i = splitIndex + foldSize; i < dataset.numInstances(); i++) {
            splitData[0].add(dataset.instance(i));
        }
        return splitData;
    }

    /**
     * Helper method for making an Enumeration being able to being used by Java 8 streams.
     * @param e
     * @param <T>
     * @return
     */
    @SuppressWarnings("unchecked")
    private static <T> Stream<T> enumerationAsStream(Enumeration e) {
        return StreamSupport.stream(
                Spliterators.spliteratorUnknownSize(
                        new Iterator<T>() {
                            public T next() {
                                return (T)e.nextElement();
                            }
                            public boolean hasNext() {
                                return e.hasMoreElements();
                            }
                        },
                        Spliterator.ORDERED), false);
    }

}