package hw7;

import weka.core.Instances;
import weka.core.Instance;


public class KMeans {

    int k;
    double[] centroids;

    public void setK(int k) {
        this.k = k;
    }

    /**
     * Initializes any parameters (therefore should call initializecentroids)
     * and runs the K-Means algorithm (which means to call findKMeansCentroids methods).
     *
     * It runs everything necessary to find the K clusters centroids on the input instances.
     *
     * @param datset - to run the algorithm on.
     */
    public void buildClusterModel(Instances datset) {

        initializeCentroids(datset);
        findKMeansCentroids(datset);

    }


    /**
     * Initializes the centroids by selecting k random instances from the training set and setting the centers as those instances.
     * @param dataset - the training set.
     */
    public void initializeCentroids(Instances dataset) {

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

    }


    /**
     * Calculates the squared distance between the input instance and the input centroid.
     * @param instance - the given instance.
     * @param centroid - the array of centroids.
     * @return the squared euclidian distance between the instance to the centroid.
     */
    public double calcSquaredDistance(Instance instance, double centroid) {


        return -1.0;
    }


    /**
     * Finds the closest centroid to the input instance.
     * @param instance - the given instance.
     * @return the index of the closest centroid.
     */
    public int findClosestCentroid(Instance instance) {


        return -1;
    }

    /**
     * Replaces every instance in Instances to the centroid to which it is closest to and return the new Instances object.
     * @param dataset - the given instances.
     * @return the clusters according to the centroids replacing each instance.
     */
    public Instances quantize(Instances dataset) {


        return null;
    }


    /**
     * Calculates the cost function.
     * Meaning, calculates the square root of the sum of the squared distances of every instance from the closest centroid to it.
     * @param dataset
     * @return
     */
    public double calcAvgESSSE(Instances dataset) {


        return -1.0;
    }

}
