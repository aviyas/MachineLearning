package hw7;

import com.sun.tools.doclets.formats.html.SourceToHTMLConverter;
import weka.core.Debug;
import weka.core.Instances;
import weka.core.Instance;

import java.util.*;

public class KMeans {

    private int k;
    private double[][] centroids;
    private static int NUM_ITERATIONS;
    private static double THRESHOLD;

    public void setK(int k) {
        this.k = k;
    }

    /**
     * Initializes any parameters and runs the K-Means algorithm.
     * It runs everything necessary to find the K clusters centroids on the input instances.
     *
     * @param datset - to run the algorithm on.
     */
    public void buildClusterModel(Instances datset) {
        centroids = new double[k][4];
        initializeCentroids(datset);
        findKMeansCentroids(datset);
    }

    /**
     * Initializes the centroids by selecting k random instances from the training set
     * and setting the centers as those instances.
     * @param dataset - the training set.
     */
    public void initializeCentroids(Instances dataset) {

        HashSet<Integer> randomIndices = randomizeKIntegers(dataset.numInstances() + 1);

        int centroidIndex = 0;

        // Copies the instances at the random indices RGBA values
        for (int instanceIndex : randomIndices) {
            for (int j = 0; j < 4; j++) {
                centroids[centroidIndex][j] = dataset.instance(instanceIndex).value(j);
            }
            centroidIndex++;
        }
    }

    /**
     * Randomizes k integers in a given range.
     * @param range - the given range.
     * @return the randmoize integers.
     */
    private HashSet<Integer> randomizeKIntegers(int range) {

        // Randomizes k distinct indices
        Random random = new Random();
        HashSet<Integer> randomIndices = new HashSet<>();

        while (randomIndices.size() < k) {
            int randomInteger = random.nextInt(range);
            randomIndices.add(randomInteger);
        }

        return randomIndices;
    }

    /**
     * Finds the centroids according with the k-means algorithm, that converges to a minimum of the cost function.
     *
     * Stopping condition of iterations is "not moved much from previous location" (cost function / centroids), or
     * running for a certain number of iterations (40 is said to be good).
     *
     * @param dataset - to run the algorithm on.
     */
    public void findKMeansCentroids(Instances dataset) {

        for (int i = 0; i < 40; i++) {

            // 1. Assigns all instances to their nearest centroid
            ArrayList<Instance>[] currentClusters = assignClusters(dataset);

            // 2. Recomputes centroids using their assigned cluster
            recomputeCentroids(currentClusters);

        }

    }

    /**
     * Helper method - assigns instances to their closest centroid.
     * @param dataset
     * @return
     */
    private ArrayList<Instance>[] assignClusters(Instances dataset) {

        ArrayList<Instance>[] currentClusters = (ArrayList<Instance>[])new ArrayList[k];

        // 1. Initialize the current clusters
        for (int i = 0; i < k; i++) {
            currentClusters[i] = new ArrayList<>();
        }

        // 2. Assigns all instances to their nearest centroid
        int closestCentroid;
        Instance curInstance;

        for (int i = 0; i < dataset.numInstances(); i++) {
            curInstance = dataset.instance(i);
            closestCentroid = findClosestCentroid(curInstance);

            // Adds the instance to the cluster of its closest centroids'
            currentClusters[closestCentroid].add(curInstance);
        }
        return currentClusters;
    }

    /**
     * Recomputes centroids using their assigned cluster.
     * @param currentClusters - the assigned clusters.
     */
    private void recomputeCentroids(ArrayList<Instance>[] currentClusters) {

        // For each centroid,
        for (int centroidIndex = 0; centroidIndex < k; centroidIndex++) {

            if (currentClusters[centroidIndex].isEmpty()) {
                continue;
            }

            double[] newCentroid = new double[4];

            // Iterates through the instances of its cluster and sums their RGBA values
            ArrayList<Instance> cluster = currentClusters[centroidIndex];

            for (Instance instance : cluster) {
                for (int i = 0; i < 4; i++) {
                    newCentroid[i] += instance.value(i);
                }
            }

            // Divides by the number of instances in the cluster to get the average
            for (int i = 0; i < 4; i++) {
                newCentroid[i] /= cluster.size();
            }

            // Sets the new centroid that was found
            centroids[centroidIndex] = newCentroid;
        }
    }

    /**
     * Calculates the squared distance between the input instance and the input centroid.
     * @param instance - the given instance.
     * @param centroid - the array of centroids.
     * @return the squared Euclidian distance between the instance to the centroid.
     */
    public double calcSquaredDistance(Instance instance, double[] centroid) {

        double squaredSum = 0;

        // Sums the squares of the differences between each RGBA values
        for (int i = 0; i < 4; i++) {
            squaredSum += Math.pow(instance.value(i) - centroid[i], 2);
        }

        // Returns the root of the sum
        return Math.sqrt(squaredSum);
    }

    /**
     * Finds the closest centroid to the input instance.
     * @param instance - the given instance.
     * @return the index of the closest centroid.
     */
    public int findClosestCentroid(Instance instance) {

        double currentDistance;
        double minDistance = Double.MAX_VALUE;
        int minCentroidIndex = 0;

        // Iterates over the centroids, calculates the distances, and saves the index of the min distance found
        for (int curCentroid = 0; curCentroid < k; curCentroid++) {
            currentDistance = calcSquaredDistance(instance, centroids[curCentroid]);

            if (currentDistance < minDistance) {
                minCentroidIndex = curCentroid;
                minDistance = currentDistance;
            }
        }

        return minCentroidIndex;
    }

    /**
     * Replaces every instance in Instances to the centroid to which it is closest to and return the new Instances object.
     * @param dataset - the given instances.
     * @return the clusters according to the centroids replacing each instance.
     */
    public Instances quantize(Instances dataset) {

        Instances quantizedInstances = new Instances(dataset);
        double[] closestCentroid;

        for (int i = 0; i < dataset.numInstances(); i++) {
            closestCentroid = centroids[findClosestCentroid(dataset.instance(i))];
            for (int j = 0; j < 4; j++) {
                quantizedInstances.instance(i).setValue(j, closestCentroid[j]);
            }
        }
        return quantizedInstances;
    }

    /**
     * Calculates the cost function,
     * which is the square root of the sum of the squared distances of every instance from the closest centroid to it.
     * @param dataset
     * @return the average error.
     */
    public double calcAvgESSSE(Instances dataset) {

        double distanceSum = 0;

        for (int i = 0; i < dataset.numInstances(); i++) {
            distanceSum += calcSquaredDistance(dataset.instance(i), centroids[findClosestCentroid(dataset.instance(i))]);
        }

        return distanceSum / dataset.numInstances();
    }

}