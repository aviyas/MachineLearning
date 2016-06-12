package hw7;

import com.sun.tools.doclets.formats.html.SourceToHTMLConverter;
import weka.core.Debug;
import weka.core.Instances;
import weka.core.Instance;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.Random;

public class KMeans {

    private int k;
    private double[][] centroids;

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
        initializeCentroids(datset);
        findKMeansCentroids(datset);
    }

    /**
     * Initializes the centroids by selecting k random instances from the training set
     * and setting the centers as those instances.
     * @param dataset - the training set.
     */
    public void initializeCentroids(Instances dataset) {
        centroids = new double[k][4];

        Random random = new Random();
        int range = dataset.numAttributes() + 1;
        Instance randomInstance;

        // Randomizes k instances
        for (int i = 0; i < k; i++) {
            randomInstance = dataset.instance(random.nextInt(range));

            // Copies their RGBA values
            for (int j = 0; j < 4; j++) {
                centroids[i][j] = randomInstance.value(j);
            }
        }

        // Prints centroids for debugging purposes
        System.out.println("Initalized: ");
        for (int i = 0; i < k; i++) {
            System.out.print(Arrays.toString(centroids[i]) + " , ");
        }
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
            findKMeansCentroidsIteration(dataset);
        }

    }

    public void findKMeansCentroidsIteration(Instances dataset) {

        ArrayList<Instance>[] clusters = (ArrayList<Instance>[])new ArrayList[k];

        for (int i = 0; i < k; i++) {
            clusters[i] = new ArrayList<>();
        }

        // 1. Assigns all instances to their nearest centroid:
        int closestCentroid;
        Instance curInstance;

        for (int i = 0; i < dataset.numInstances(); i++) {
            curInstance = dataset.instance(i);
            closestCentroid = findClosestCentroid(curInstance);

            // Adds the instance to the cluster of its closest centroids'
            clusters[closestCentroid].add(curInstance);
        }

        // 2. Recomputes centroids using their assigned cluster:

        // For each centroid,
        for (int centroidIndex = 0; centroidIndex < k; centroidIndex++) {

            if (clusters[centroidIndex].isEmpty()) {
                continue;
            }

            double[] newCentroid = new double[4];

            // Iterates through the instances of its cluster and sums their RGBA values
            ArrayList<Instance> cluster = clusters[centroidIndex];

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

        // Prints centroids for debugging purposes
        for (int j = 0; j < k; j++) {
            System.out.print(Arrays.toString(centroids[j]) + " , ");
            System.out.println(clusters[j].size());
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