package homework2;

import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

import java.security.InvalidParameterException;
import java.util.*;
import java.util.stream.*;

public class DecisionTree extends Classifier{

    private boolean pruningMode = false;
    private Node rootNode = null;
    private double threshold = 2.733; // Default to cancer threshold.

    /**
     * Builds a decision tree from the training data as learned in class.
     * @param instances Instances object of the training data.
     * @throws Exception
     */
    @Override
    public void buildClassifier(Instances instances) throws Exception {
        instances.setClassIndex(instances.numAttributes() - 1);

        System.out.println("Building root node.");
        this.rootNode = buildTree(instances, new ArrayList<>());

        if (pruningMode) {
            postPruneTree();
        }
    }

    public void setPruningMode(boolean pruningMode){
        this.pruningMode = pruningMode;
    }

    public void setThreshold(double threshold) {
        this.threshold = threshold;
    }

    private Node buildTree(String value, List<Integer> alreadyUsedAttributes, Instances trainingData) {
        System.out.format("Building tree for %s.\n", value);
        return buildTree(trainingData, alreadyUsedAttributes);
    }

    private double calculateCommonClassValue(Instances instances) {
        return Math.round(DecisionTree.<Instance>enumerationAsStream(instances.enumerateInstances())
                .mapToDouble(Instance::classValue).average().orElseThrow(InvalidParameterException::new));
    }

    /**
     * Builds a decision tree on a given dataset using the recursive algorithm as learned in class.
     * @param trainingData
     * @param alreadyUsedAttributes
     * @return a decision tree
     */
    private Node buildTree(Instances trainingData, List<Integer> alreadyUsedAttributes) {
        final Node root = new Node(alreadyUsedAttributes, trainingData);

        System.out.format("Building tree with %d instances.\n", trainingData.numInstances());

        // Leaf, if Class Value is the same for all data
        if (isHomogeneous(trainingData)) {
            System.out.format("Found homogeneous data, setting as leaf for class value: %f\n",
                    trainingData.instance(0).classValue());

            root.setData(trainingData.instance(0).classValue());
            return root;
        }

        // Chooses best attribute and splits data accordingly
        int chosenAttributeIndex = chooseBestAttribute(trainingData, root.getAlreadyUsedAttributes());
        System.out.format("Best attribute chosen is %d.\n", chosenAttributeIndex);

        if (chosenAttributeIndex == -1)
        {
            double val = calculateCommonClassValue(trainingData);
                System.out.format("No more info gain, setting as leaf for class value according to average: %f\n",
                        val);
                root.setData(val);

                return root;
        }

        root.setData(chosenAttributeIndex);

        // Creating a new list to be used in the children, so we won't check the same attribute twice.
        List<Integer> newAlreadyUsedAttributes = root.getAlreadyUsedAttributes().stream().collect(Collectors.toList());
        newAlreadyUsedAttributes.add(chosenAttributeIndex);

        System.out.println("Splitting data according to the chosen index.");
        Map<String, Instances> split = splitData(trainingData, chosenAttributeIndex);

        Attribute attribute = trainingData.attribute(chosenAttributeIndex);

        System.out.format("Data splitted to %d sections.\n", split.size());

        // Builds the branch for each new child
        split.entrySet().forEach(entry -> root.addChild(entry.getKey(),
                buildTree(entry.getKey(), newAlreadyUsedAttributes, entry.getValue())));

        // In case we only covered part of the options for the attribute we should add leafs for the other options.
        if (split.size() != attribute.numValues()) {
            double commonClassValue = calculateCommonClassValue(trainingData);

            DecisionTree.<String>enumerationAsStream(attribute.enumerateValues())
                    .filter(value -> !split.containsKey(value))
                    .forEach(value -> root.addChild(value, commonClassValue));
        }

        // Returns the final tree
        return root;
    }

    /**
     * Post-prunes a given tree.
     *
     * Note I: Chi Squared Error is used to decide whether to prune or not,
     *         comparing to the threshold 2.733 for the cancer data and 11.591 for the mushroom data.
     *         If the error is lower than threshold, we prune.
     *
     * Note II: When pruning a branch, we replace it by a leaf containing the majority class value of the data associated with it.
     *
     * @param root
     * @return
     */
    public void postPruneTree() {
        // Let's go all the way down to the leafs.
        List<Node> leafs = getLeafs();

        for (Node currentNode : leafs) {
            Node parent = currentNode.getParent();

            while (parent != null && !parent.isPruned()) {
                double chiSquare = calcChiSquare(parent.getTrainingData(), (int) parent.getData());

                if (chiSquare < threshold) {
                    // we prune.
                    double v = calculateCommonClassValue(parent.getTrainingData());
                    parent.makeLeaf(v);
                }

                parent.setPruned(true);
                parent = parent.getParent();
            }
            currentNode.setPruned(true);
        }
    }

    private List<Node> getLeafs() {
        ArrayList<Node> leafs = new ArrayList<>();
        getLeafs(rootNode, leafs);
        return leafs;
    }

    private void getLeafs(Node root, List<Node> leafs) {
        for (Node child : root.getChildren().values()) {
            if (child.isLeaf()) {
                leafs.add(child);
            } else {
                getLeafs(child, leafs);
            }
        }
    }

    /**
     * Helper method to split the data according to the given attribute index, and removes any values of the attribute
     * that did not have any representation in the trainingData.
     * @param trainingData
     * @param chosenAttributeIndex
     * @return
     */
    private HashMap<String, Instances> splitData(Instances trainingData, int chosenAttributeIndex) {

        HashMap<String, Instances> split = new HashMap<>();

        Map<String, List<Instance>> collect =
                DecisionTree.<Instance>enumerationAsStream(trainingData.enumerateInstances())
                .collect(Collectors.groupingBy(instance -> instance.stringValue(chosenAttributeIndex)));

        collect.forEach((k, v) -> v.forEach(instance -> {
            split.putIfAbsent(k, new Instances(trainingData, 0));
            split.get(k).add(instance);
        }));

        return split;
    }

    /** Helper method to choose the best attribute to split data by
     *
     * @param trainingData
     * @param alreadyUsedAttributes
     * @return
     */
    private int chooseBestAttribute(Instances trainingData, List<Integer> alreadyUsedAttributes) {
        System.out.format("Choosing best attribute from %d instances.\n", trainingData.numInstances());

        double[] infoGains = IntStream.range(0, trainingData.numAttributes() - 1)
                .boxed()
                .mapToDouble(i -> alreadyUsedAttributes.contains(i)? 0 : calcInfoGain(trainingData, i)).toArray();

        Optional<Integer> bestAttributeIndex = IntStream.range(0, trainingData.numAttributes() - 1)
                .boxed().max(Comparator.comparing(i -> infoGains[i]));

        bestAttributeIndex.orElseThrow(RuntimeException::new);

        if (bestAttributeIndex.isPresent()) {
            int index = bestAttributeIndex.get();
            if (infoGains[index] == 0) {
                return -1;
            }

            return bestAttributeIndex.get();
        } else {
            return Integer.MIN_VALUE;
        }
    }

    /** Helper method to check whether all given instances in a data set share the same class value
     * @param trainingData all instances in current node
     * @return whether this node should be considered as a leaf
     */
    private boolean isHomogeneous(Instances trainingData) {
        return DecisionTree.<Instance>enumerationAsStream(trainingData.enumerateInstances())
                .map(Instance::classValue)
                .distinct()
                .limit(2)
                .count() == 1;
    }

    /**
     * Calculates the information gain of splitting the input data according to the given attribute.
     * @param trainingSet
     * @return the information gain
     */
    private double calcInfoGain(Instances trainingSet, int attributeIndex) {
        double trainingSetEntropy = calcEntropy(trainingSet);

        // Sums entropy for each value of the attribute
        Map<String, Instances> split = splitData(trainingSet, attributeIndex);
        double total = trainingSet.numInstances();

        double splittedEntropy = split.values().stream()
                .mapToDouble(instances -> (instances.numInstances() / total) * calcEntropy(instances))
                .sum();

        double infoGain = trainingSetEntropy - splittedEntropy;

        System.out.format("InfoGain for attribute %d is %f.\n", attributeIndex, infoGain);

        return infoGain;
    }

    /** Helper method that calculates the probability to get each class value in a given dataset.
     *
     * @param instances
     * @return
     */
    private Collection<Double> calcProbabilities(Instances instances) {
        Map<Double, Double> counts =
                DecisionTree.<Instance>enumerationAsStream(instances.enumerateInstances())
                        .collect(Collectors.groupingBy(Instance::classValue,
                                Collectors.reducing(0D, value -> 1D, Double::sum)));

        counts.replaceAll((k, v) -> v / (double)instances.numInstances());

        return counts.values();
    }

    /**
     * Calculates the entropy of a certain attribute, using log base 2.
     * @param instances - The instances to calculate the entropy for.
     * @return the entropy
     */
    private double calcEntropy(Instances instances) {
        return calcProbabilities(instances).stream()
                .mapToDouble(probability -> -(probability * (Math.log(probability) / Math.log(2))))
                .sum();
    }

    /**
     * Calculates the Chi Square Error statistic of splitting the given data according to the given attribute as learned in class.
     *
     * @param trainingDataSet - the given Instances set that is to be splitted
     * @param attributeIndex - the given attribute to decide whether to split by
     * @return the Chi Square Error
     */
    public double calcChiSquare(Instances trainingDataSet, int attributeIndex) {
        // Calculate P(Y = 0).
        double py0 = DecisionTree.<Instance>enumerationAsStream(trainingDataSet.enumerateInstances())
                .mapToDouble(instance -> instance.classValue() == 0 ? 1 : 0).average().orElseThrow(RuntimeException::new);
        double py1 = 1 - py0;

        Attribute attribute = trainingDataSet.attribute(attributeIndex);

        double chiSquare = 0;

        for (int i = 0; i < attribute.numValues(); i++) {
            String f = attribute.value(i);

            long df = DecisionTree.<Instance>enumerationAsStream(trainingDataSet.enumerateInstances())
                    .map(instance -> instance.stringValue(attributeIndex))
                    .filter(f::equals)
                    .count();

            long pf = DecisionTree.<Instance>enumerationAsStream(trainingDataSet.enumerateInstances())
                    .filter(instance -> instance.stringValue(attributeIndex).equals(f))
                    .filter(instance -> instance.classValue() == 0)
                    .count();

            long nf = DecisionTree.<Instance>enumerationAsStream(trainingDataSet.enumerateInstances())
                    .filter(instance -> instance.stringValue(attributeIndex).equals(f))
                    .filter(instance -> instance.classValue() == 1)
                    .count();

            double e0 = df * py0;
            double e1 = df * py1;
            if (e0 != 0 && e1 != 0) {
                chiSquare += Math.pow(e0 - pf, 2) / e0 + Math.pow(e1 - nf, 2) / e1;
            }
        }

        return chiSquare;
    }

    /**
     * Classifies an instance according to the decision tree built.
     * @param instance - the given Instance to be classified
     * @return classification (0 or 1)
     */
    public double classify(Instance instance) {

        Node currentNode = rootNode;
        while (!currentNode.isLeaf()) {
            int currentAttributeToFollow = (int)currentNode.getData();
            // Gets the instance value of the chosen attribute
            String instanceAttributeValue = instance.stringValue(currentAttributeToFollow);

            currentNode = currentNode.getChildren().get(instanceAttributeValue);
        }

        return currentNode.data;
    }

    /**
     * Counts the total number of classification mistakes on the given dataset and divides it by the number of instances.
     * @param instances
     * @return the average error
     */
    public double calcAvgError(Instances instances) {

        return DecisionTree.<Instance>enumerationAsStream(instances.enumerateInstances())
                .mapToDouble(instance -> this.classify(instance) != instance.classValue()? 1 : 0).average()
                .orElseThrow(() -> new IllegalArgumentException("Error calculating average."));
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

    /**
     * Creates Node instances that allows one to build a tree.
     */
    private class Node {

        private Instances trainingData = null;
        private Map<String, Node> children = new HashMap<>();
        private Node parent = null;

        // Holds bestAttributeIndex in case of general node, classification in case of leaf
        private double data = -1;

        private List<Integer> alreadyUsedAttributes = new ArrayList<>();
        private boolean pruned;


        public Node(List<Integer> alreadyUsedAttributes) {
            this.alreadyUsedAttributes = alreadyUsedAttributes;
        }

        public Node(double data, List<Integer> alreadyUsedAttributes) {
            this.data = data;
            this.alreadyUsedAttributes = alreadyUsedAttributes;
        }

        public Node(double data) {
            this.data = data;
        }

        public Node(List<Integer> alreadyUsedAttributes, Instances trainingData) {
            this(alreadyUsedAttributes);
            this.trainingData = trainingData;
        }

        public Map<String, Node> getChildren() {
            return this.children;
        }

        public Node getParent() {
            return this.parent;
        }

        public double getData() {
            return this.data;
        }

        public void addChild(String attribute, int data, List<Integer> alreadyUsedAttribute) {
            Node child = new Node(data, alreadyUsedAttribute);
            child.setParent(this);
            this.children.put(attribute, child);
        }

        public void addChild(String attribute, Node child) {
            child.setParent(this);
            this.children.put(attribute, child);
        }

        public void addChild(String attribute, double data) {
            Node child = new Node(data);
            this.children.put(attribute, child);
        }

        public void setParent(Node parent) {
            this.parent = parent;
        }

        public void setData(double data) {
            this.data = data;
        }

        public boolean isRoot() {
            return (this.parent == null);
        }

        public boolean isLeaf() {
            return (this.children.size() == 0);
        }

        public List<Integer> getAlreadyUsedAttributes() {
            return alreadyUsedAttributes;
        }

        public Instances getTrainingData() {
            return trainingData;
        }

        public void makeLeaf(double v) {
            data = v;
            children = new HashMap<>();
        }

        public boolean isPruned() {
            return pruned;
        }

        public void setPruned(boolean pruned) {
            this.pruned = pruned;
        }
    }
}