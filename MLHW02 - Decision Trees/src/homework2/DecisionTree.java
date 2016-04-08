package homework2;

import weka.classifiers.Classifier;
import weka.core.Instances;

public class DecisionTree extends Classifier {

    private boolean m_pruningMode = false;

    /* buildClassifier
    Input: Instances object of the training data.
    Description: Builds a decision tree from the training data as learned in class.
    Note: BuildClassifier is separated from BuildTree in order to allow you to do extra preprocessing before calling buildTree.
          Also this is the only method we provide the signature for because the signature for this method is determined by WEKA.
     */
    @Override
    public void buildClassifier(Instances arg0) throws Exception {

    }

    public void setPruningMode(boolean pruningMode) {
        m_pruningMode = pruningMode;
    }

    /* buildTree
    Input: Training data (or a subset of it if using a recursive algorithm), other variables if you need them
    Description: Builds the decision tree on given data set using either a recursive or queue algorithm as learned in class.
    Output: a decision tree.
     */
    private Node growTree(Instances dataset) {

        // Leaf, if Class Value is the same for all data
        if (isHomogenious(dataset)) {
            new Node(dataset.getInstance(i).getClassValue);
        }

        // Chooses best attribute and splits data accordingly

        // For each new child, growTree on it

        // Returns the final tree
        return new Node(-1);
    }

    // Checks whether class value is the same for all instances
    private boolean isHomogenious(Instances dataset) {

        int value = dataset.getInstance(0).getClassValue;

        for (int i = 1; i < dataset.numInstances() - 1; i++) {
            if (dataset.getInstance(i).getClassValue != value) {
                return false;
            }
        }

        return true;
    }

    /* calcInfoGain
    Input: A subset of the training data, attribute index, extra variables you need.
    Description: The method calculates the information gain of splitting the input data according to the attribute.
    Output: The information gain (double).
    */

    /* calcEntropy
    Input: a set of probabilities
    Description: Calculates the entropy of a random variable where all the probabilities of all of the possible values it can take are given as input.
    Output: The entropy (double).
    Note: using a log base e will work but you should use log base 2.
     */

    /* calcChiSquare
    Input: A subset of the training data, attribute index
    Description: Calculates the chi square statistic of splitting the data according to this attribute as learned in class.
    Output: The chi square score (double).
    Note:  This method is required only for the version of the algorithm with the pruning (section 6).
           When deciding whether to prune or not to prune a branch you preform the chi squared test and then you compare
           the number you get to the chi squared chart. The number you should compare to for the cancer data is 2.733 .
           This number comes from the chi squared chart in the row for 8 degrees of freedom
           (which is the number of attributes in the cancer data minus 1) and the column for 0.95 confidence level.

            If you use the mushroom data you should use the number 11.591 which is 0.95 confidence column and 21 dof.
            If your chi squared statistic is greater than the threshold, you continue building. If it is smaller, you prune.

    Note on pruning: If you prune a branch and make a leaf node instead of it, what should that leaf node return?
                     It should return the majority class value of the data associated with that node.
     */

    /* Classify
    Input: Instance object
    Description: Returns a classification for the instance.
    Output: Classification - 0/1 (double)
     */

    /* calcAvgError
    Input: Instances object of testing data set
    Description: Counts the total number of classification mistakes on the testing data set
                 and divides that by the number of data instances.
    Output: Average error (double).
    Note: For example - if on a test set you classified 10 examples correctly, and 10 incorrectly, the output of this method should be 0.5.
     */

    /**
     * Implements Tree
     */
    private class Node {

        private List<Node> children = new ArrayList<Node>();
        private Node parent = null;
        private int attributeIndex = -1;

        public Node(int attributeIndex) {
            this.attributeIndex = attributeIndex;
        }

        public Node(int attributeIndex, Node parent) {
            this.attributeIndex = attributeIndex;
            this.parent = parent;
        }

        public List<Node> getChildren() {
            return this.children;
        }

        public Node getParent() {
            return this.parent;
        }

        public int getAttributeIndex() {
            return this.attributeIndex;
        }

        public void addChild(int attributeIndex) {
            Node child = new Node(attributeIndex);
            child.setParent(this);
            this.children.add(child);
        }

        public void addChild(Node child) {
            child.setParent(this);
            this.children.add(child);
        }

        public boolean isRoot() {
            return (this.parent == null);
        }

        public boolean isLeaf() {
            return (this.children.size() == 0);
        }

    }

}