package hw7;

import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;
import java.util.ArrayList;
import java.util.List;

import javax.imageio.ImageIO;

import weka.classifiers.functions.LinearRegression;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
//import weka.filters.unsupervised.attribute.PrincipalComponents;
import weka.filters.unsupervised.attribute.Remove;

public class Hw7Main {
	public static BufferedReader readDataFile(String filename) {
		BufferedReader inputReader = null;

		try {
			inputReader = new BufferedReader(new FileReader(filename));
		} catch (FileNotFoundException ex) {
			System.err.println("File not found: " + filename);
		}

		return inputReader;
	}

	public static Instances convertImgToInstances(BufferedImage image) {
		Attribute attribute1 = new Attribute("alpha");
		Attribute attribute2 = new Attribute("red");
		Attribute attribute3 = new Attribute("green");
		Attribute attribute4 = new Attribute("blue");
		FastVector fvWekaAttributes = new FastVector(4);
		fvWekaAttributes.addElement(attribute1);
		fvWekaAttributes.addElement(attribute2);
		fvWekaAttributes.addElement(attribute3);
		fvWekaAttributes.addElement(attribute4);
		Instances imageInstances = new Instances("Image", fvWekaAttributes, image.getHeight() * image.getWidth());

		int[][] result = new int[image.getHeight()][image.getWidth()];
		int[][][] resultARGB = new int[image.getHeight()][image.getWidth()][4];

		for (int col = 0; col < image.getWidth(); col++) {
			for (int row = 0; row < image.getHeight(); row++) {
				int pixel = image.getRGB(col, row);

				int alpha = (pixel >> 24) & 0xff;
				int red = (pixel >> 16) & 0xff;
				int green = (pixel >> 8) & 0xff;
				int blue = (pixel) & 0xff;
				result[row][col] = pixel;
				resultARGB[row][col][0] = alpha;
				resultARGB[row][col][1] = red;
				resultARGB[row][col][2] = green;
				resultARGB[row][col][3] = blue;

				Instance iExample = new weka.core.DenseInstance(4);
				DenseInstance a = new DenseInstance(0);
				iExample.setValue((Attribute) fvWekaAttributes.elementAt(0), alpha);// alpha
				iExample.setValue((Attribute) fvWekaAttributes.elementAt(1), red);// red
				iExample.setValue((Attribute) fvWekaAttributes.elementAt(2), green);// green
				iExample.setValue((Attribute) fvWekaAttributes.elementAt(3), blue);// blue
				imageInstances.add(iExample);
			}
		}

		return imageInstances;

	}

	public static BufferedImage convertInstancesToImg(Instances instancesImage, int width, int height) {
		final BufferedImage image = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
		int index = 0;
		for (int col = 0; col < width; ++col) {
			for (int row = 0; row < height; ++row) {
				Instance instancePixel = instancesImage.instance(index);
				int pixel = ((int) instancePixel.value(0) << 24) | (int) instancePixel.value(1) << 16
						| (int) instancePixel.value(2) << 8 | (int) instancePixel.value(3);
				image.setRGB(col, row, pixel);
				index++;
			}
		}
		return image;
	}

	public static void main(String[] args) throws Exception {

		System.out.println("Started running");

		// 1. Create instances object from image
		BufferedImage image = ImageIO.read(new File("baboon_face.jpg"));
		int origHeight = image.getHeight();
		int origWidth = image.getWidth();
		Instances pixels = convertImgToInstances(image);
		Instances quantizedPixels;

		// 2. Quantize instance object using K-Means, for k = 2,3,5,10,25,50,100,256
		// int[] kValues = {2, 3, 5, 10, 25, 50, 100, 256};
		int[] kValues = {100};

		for (int value : kValues) {
			KMeans model = new KMeans();
			model.setK(value);
			model.buildClusterModel(pixels);

			quantizedPixels = model.quantize(pixels);

			// 3. Convert back to an image and save the result
			BufferedImage resultImage = convertInstancesToImg(quantizedPixels, image.getWidth(), image.getHeight());
			File result = new File("resultFor" + value + ".jpg");
			ImageIO.write(resultImage, "jpg", result);
		}

		// 4. Runs PCA looping over number i of principal components
		//    and prints the average euclidean distance (between transformed and original instances)

	}

	/**
	 * Iterates over the instances in the transformed and original set and for each corresponding pair of instances,
	 * it measures the Euclidean Distance between them and averages over the number of instances.
	 * @param original - original dataset.
	 * @param transformed - transformed dataset.
     */
	public double calcAvgDistance(Instances original, Instances transformed) {
//		double distanceSum = 0;
//		double squaredSum;
//
//		for (int i = 0; i < original.numInstances(); i++) {
//			squaredSum = 0;
//
//			for (int j = 0; j < original.numAttributes(); j++) {
//				squaredSum += Math.pow(original.instance(i).value(j) - transformed.instance(i).value(j), 2);
//			}
//			distanceSum += Math.sqrt(squaredSum);
//		}
//		return distanceSum / original.numAttributes();
		return -1.0;
	}

}
