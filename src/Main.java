/*
 * TODO: create and train a neural net to classify handwritten digits from the MNIST data set
 * 
 * choose eps based on ulp?
 * 		eps = x0*sqrt(ulp)		(if x0 != 0)

 * restructure net to have weights/biases in one array
 * 
 * figure out how to find the gradient of a function
 * 		use forward difference rather than central to compute partial derivatives so we need only compute the cost function once per derivative
 * 		dC/dw_i = (C_{w_i + eps} - C_{w_i})/eps
 * 		C(w,b) = Sum_x ||y(x)-a||^2/(2n)
 * 			where a is the vector of outputs when x is input, y(x) is the desired output, and n is the numbe4 of training inputs
 * 		let V denote the current vector of weights and biases for the net
 * 		V' = V - eta*grad(C(V))
 */

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.Random;

public class Main {
	public static int nodesInLayer0 = 28*28;
	public static int nodesInLayer1 = 30;
	public static int nodesInLayer2 = 10;
	public static int numTrainingImages = 60000;
	public static int numTestingImages = 10000;
	public static double learningRate = 3.0;
	public static int epochs = 30;
	public static int miniBatchSize = 10;
	
	
	//a, z, delta, b, w
	private static double[] a0 = new double[nodesInLayer0];
	private static double[] a1 = new double[nodesInLayer1];
	private static double[] a2 = new double[nodesInLayer2];
	
	private static double[] z1 = new double[nodesInLayer1];
	private static double[] z2 = new double[nodesInLayer2];
	
	private static double[] b1 = new double[nodesInLayer1];
	private static double[] b2 = new double[nodesInLayer2];
	
	private static double[][] w1 = new double[nodesInLayer1][nodesInLayer0];
	private static double[][] w2 = new double[nodesInLayer2][nodesInLayer1];
	
	private static double[] delta1 = new double[nodesInLayer1];
	private static double[] delta2 = new double[nodesInLayer2];
	
	//training images and labels
	private static char[][] trainingImages = new char[numTrainingImages][nodesInLayer0];
	private static char[] trainingLabels = new char[numTrainingImages];
	
	//test images and labels
	private static char[][] testingImages = new char[numTestingImages][nodesInLayer0];
	private static char[] testingLabels = new char[numTestingImages];
	
	//used for accuracy statistics
	private static int[] labelCounts = new int[nodesInLayer2];
	private static int[] correctCounts = new int[nodesInLayer2];
	
	private static Random rand = new Random();

	
	
	public static void main(String[] args) throws IOException{
		//while(true) {
			//initialize b,w with random numbers
			initializeBiasesAndWeights();
			
			//load training and testing images and labels
			loadData();
			
			trainNet();
			
			printAccuracy(1);
			
			//print intro and user input options
				//1. train network
				//2. load pre-trained network
				//3. display network accuracy on training data
				//4. display network accuracy on testing data
				//5. save network state to file
				//6. exit
			
			//wait for user input
			//switch on user input to do the thing
		//}
	}
	
	//Math.random() gives a uniformly random number between 0.0 and 1.0
	private static void initializeBiasesAndWeights() {
		//b1		
		for(int i=0; i<b1.length; i++) {
			b1[i] = rand.nextGaussian();
		}
		
		//b2
		for(int i=0; i<b2.length; i++) {
			b2[i] = rand.nextGaussian();
		}
		
		//w1
		for(int k=0; k<a0.length; k++) {
			for(int j=0; j<a1.length; j++) {
				w1[j][k] = rand.nextGaussian();
			}
		}
		
		//w2
		for(int k=0; k<a1.length; k++) {
			for(int j=0; j<a2.length; j++) {
				w2[j][k] = rand.nextGaussian();
			}
		}
		
	}
	
	private static void loadData() {
		FileReader fr;
		BufferedReader br;
				
		try {
			//training images
			fr = null;
			br = null;
			fr = new FileReader("train-images.idx3-ubyte");
			br = new BufferedReader(fr);
			br.skip(16);	//skip over the images file header
			for(int i=0; i<trainingImages.length; i++) {
				br.read(trainingImages[i], 0, trainingImages[0].length);
			}
			if(br != null) {
				br.close();
			}
			if(fr != null) {
				fr.close();
			}
			
			//training labels
			fr = null;
			br = null;
			fr = new FileReader("train-labels.idx1-ubyte");
			br = new BufferedReader(fr);
			br.skip(8);	//skip over the images file header
			br.read(trainingLabels, 0, trainingLabels.length);
			if(br != null) {
				br.close();
			}
			if(fr != null) {
				fr.close();
			}
			
			//testing images
			fr = null;
			br = null;
			fr = new FileReader("t10k-images.idx3-ubyte");
			br = new BufferedReader(fr);
			br.skip(16);	//skip over the images file header
			for(int i=0; i<testingImages.length; i++) {
				br.read(testingImages[i], 0, testingImages[0].length);
			}
			if(br != null) {
				br.close();
			}
			if(fr != null) {
				fr.close();
			}
					
			//testing labels
			fr = null;
			br = null;
			fr = new FileReader("t10k-labels.idx1-ubyte");
			br = new BufferedReader(fr);
			br.skip(8);	//skip over the images file header
			br.read(testingLabels, 0, testingLabels.length);
			if(br != null) {
				br.close();
			}
			if(fr != null) {
				fr.close();
			}
			
		} catch (Exception e) {
			System.out.println(e.getMessage());
			e.printStackTrace();
		}
	}
	
	private static void trainNet() {		
		//create a list that we can shuffle in order to randomize our mini-batches
		int[] shuffledList = new int[numTrainingImages];
		for(int i=0; i<shuffledList.length; i++) {
			shuffledList[i] = i;
		}
		
		for(int epoch=0; epoch<epochs; epoch++) {	//for each epoch
			shuffle(shuffledList);		//shuffle the training set order		
			setToZero(labelCounts);
			setToZero(correctCounts);
			
			for(int miniBatch=0; miniBatch<shuffledList.length/miniBatchSize; miniBatch++) {	//for each miniBatch
				//initialize delta1 and delta2 to 0
				setToZero(delta1);
				setToZero(delta2);
				
				for(int input=miniBatch*miniBatchSize; input<(miniBatch+1)*miniBatchSize; input++) {	//for each input image
					//load a0
					for(int i=0; i<a0.length; i++) {
						a0[i] = (double)trainingImages[shuffledList[input]][i];
					}
					
					//feed forward
					feedForward();
					
					//see if it classified correctly
					checkOutputAccuracy((int)trainingLabels[shuffledList[input]]);
					
					//backpropagate
					backpropagate(shuffledList[input]);
				}
				//update weights/biases
				updateWeightsAndBiases();
			}
			System.out.println("epoch " + epoch + ": ");
			printAccuracyStatistics();
			System.out.println("");
		}
	}
	
	private static void feedForward() {
		//calculate z1 and a1
		for(int j=0; j<a1.length; j++) {	//for each neuron in layer1
			double tmpZ = 0;
			for(int k=0; k<a0.length; k++) {	//for each neuron in layer0
				tmpZ = tmpZ + a0[k]*w1[j][k];		//add weighted output
			}
			tmpZ = tmpZ + b1[j];				//add bias
			z1[j] = tmpZ;						//store z value
			a1[j] = sigma(z1[j]);				//store a value
		}
		
		//calculate z2 and a2
		for(int j=0; j<a2.length; j++) {	//for each neuron in layer2
			double tmpZ = 0;
			for(int k=0; k<a1.length; k++) {	//for each neuron in layer1
				tmpZ = tmpZ + a1[k]*w2[j][k];			//add weighted output
			}
			tmpZ = tmpZ + b2[j];				//add bias
			z2[j] = tmpZ;						//store z value
			a2[j] = sigma(z2[j]);				//store a value
		}
	}
	
	private static void backpropagate(int inputImage) {
		//compute error for layer2
		//delta2j = (a-y)a(1-a)
		for(int j=0; j<a2.length; j++) {
			//set y according to if this is the correct classification
			int y = 0;
			if(j == (int)trainingLabels[inputImage]) {
				y = 1;
			}
			
			delta2[j] = delta2[j] + (a2[j]-y)*a2[j]*(1-a2[j]);	//this is divided by miniBatchSize later
		}
		
		//compute error for layer1
		//delta1k = sum_j delta2j*w2jk*ak*(1-ak)
		for(int k=0; k<a1.length; k++) {
			for(int j=0; j<a2.length; j++) {
				delta1[k] = delta1[k] + delta2[j]*w2[j][k]*a1[k]*(1-a1[k]);	//this is divided by miniBatchSize later
			}
		}
	}
	
	//gradients are divided by miniBatchSize because they are the sum of all gradients over the miniBatch
	private static void updateWeightsAndBiases() {
		//dC/db2 = delta2
		for(int j=0; j<a2.length; j++) {
			b2[j] = b2[j] - learningRate*delta2[j]/miniBatchSize;
		}
		
		//dC/db1 = delta1
		for(int k=0; k<a1.length; k++) {
			b1[k] = b1[k] - learningRate*delta1[k]/miniBatchSize;
		}
		
		//dC/dw2jk = delta2j*a1k
		for(int j=0; j<a2.length; j++) {
			for(int k=0; k<a1.length; k++) {
				w2[j][k] = w2[j][k] - learningRate*delta2[j]*a1[k]/miniBatchSize;
			}
		}
		
		//dC/dw1jk = delta1j*a0k
		for(int j=0; j<a1.length; j++) {
			for(int k=0; k<a0.length; k++) {
				w1[j][k] = w1[j][k] - learningRate*delta1[j]*a0[k]/miniBatchSize;
			}
		}
	}
	
	private static double sigma(double z) {
		return 1.0/(1.0+Math.pow(Math.E, -z));
	}
	
	//shuffles a list
	private static void shuffle(int[] list) {
		Random rand = new Random();
		for(int i=list.length-1; i>=0; i--) {
			int index = rand.nextInt(i+1);
			int tmp = list[index];
			list[index] = list[i];
			list[i] = tmp;
		}
	}
	
	private static void printAccuracy(int dataSet) {
		//set arrays to hold statistics to zero
		setToZero(labelCounts);
		setToZero(correctCounts);
		
		char[][] imagesArray;
		char[] labelsArray;
		int numImages;
		
		if(dataSet == 0) {	//test against training data
			imagesArray = trainingImages;
			labelsArray = trainingLabels;
			numImages = numTrainingImages;
		}
		else {	//test against testing data
			imagesArray = testingImages;
			labelsArray = testingLabels;
			numImages = numTestingImages;
		}
			
		for(int image=0; image<numImages; image++) {	//for each image
			//load a0
			for(int pixel=0; pixel<a0.length; pixel++) {
				a0[pixel] = (double)imagesArray[image][pixel];
			}
			
			//feed forward
			feedForward();
			
			//the correct answer
			int correctClassification = (int)labelsArray[image];
			
			//compare with net output
			checkOutputAccuracy(correctClassification);	
		}
		
		printAccuracyStatistics();

	}
	
	private static void printAccuracyStatistics() {
		int total = 0;
		int totalCorrect = 0;
		
		for(int i=0; i<labelCounts.length; i++) {
			total += labelCounts[i];
			totalCorrect += correctCounts[i];
			
			System.out.print(i + ":" + correctCounts[i] + "/" + labelCounts[i] + " ");
		}
		System.out.println("");
		System.out.println("Accuracy = " + totalCorrect + "/" + total + " = " + (double)totalCorrect/(double)total);
	}
	
	private static void checkOutputAccuracy(int correctClassification) {
		//find highest activation in output layer
		double highest = 0;
		int classification = 0;	//arbitrary
		for(int j=0; j<a2.length; j++) {
			if(a2[j] > highest) {
				highest = a2[j];
				classification = j;
			}
		}
		
		//increment the counter for this digit
		labelCounts[correctClassification]++;
		
		//see if the net classified correctly
		if(classification == correctClassification) {
			correctCounts[correctClassification]++;
		}
	}
	
	private static void setToZero(double[] x) {
		for(int i=0; i<x.length; i++) {
			x[i] = 0;
		}
	}
	
	private static void setToZero(int[] x) {
		for(int i=0; i<x.length; i++) {
			x[i] = 0;
		}
	}
}
