import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.Random;
import java.util.Scanner;

public class Main {
	public static int nodesInLayer0 = 28*28;
	public static int nodesInLayer1 = 30;
	public static int nodesInLayer2 = 10;
	public static int numTrainingImages = 50000;
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
	
	private static double[] gb1 = new double[nodesInLayer1];
	private static double[] gb2 = new double[nodesInLayer2];
	private static double[][] gw1 = new double [nodesInLayer1][nodesInLayer0];	
	private static double[][] gw2 = new double [nodesInLayer2][nodesInLayer1];
	
	//training images and labels
	private static int[][] trainingImages = new int[numTrainingImages][nodesInLayer0];
	private static int[] trainingLabels = new int[numTrainingImages];
	
	//test images and labels
	private static int[][] testingImages = new int[numTestingImages][nodesInLayer0];
	private static int[] testingLabels = new int[numTestingImages];
	
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
				
			//printData();
			
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
	
	static void printData() {
		for(int i=0; i<numTrainingImages; i++) {
			for(int j=0; j<28; j++) {
				for(int k=0; k<28; k++) {
					int num = (int)(trainingImages[i][j*28 + k]);
					char c;
					if(j == 0 || j == 27) c = '-';
					else if(k == 0 || k == 27) c = '|';
					else if(num < 10) c = ' ';
					else if (num < 50) c = '.';
					else if (num < 100) c = 'l';
					else if (num < 150) c = 'h';
					else if (num < 200) c = '&';
					else c = 'X';
					System.out.print(c);
				}
				System.out.println("");
			}
			System.out.println((int)trainingLabels[i]);
		}
	}
	
	//nextDouble() gives a uniformly random number between 0.0 and 1.0
	//(rand.nextDouble()- 0.5)*2 gives a uniformly random number between -1.0 and 1.0
	private static void initializeBiasesAndWeights() {
		//b1		
		for(int i=0; i<b1.length; i++) {
			b1[i] = (rand.nextDouble()- 0.5);
		}
		
		//b2
		for(int i=0; i<b2.length; i++) {
			b2[i] = (rand.nextDouble()- 0.5);
		}
		
		//w1
		for(int k=0; k<a0.length; k++) {
			for(int j=0; j<a1.length; j++) {
				w1[j][k] = (rand.nextDouble()- 0.5);
			}
		}
		
		//w2
		for(int k=0; k<a1.length; k++) {
			for(int j=0; j<a2.length; j++) {
				w2[j][k] = (rand.nextDouble()- 0.5);
			}
		}
		
	}
	
	private static void readFromFile(String fileName) {
		int[] labels;
		int[][] images;
		int numImages;
		
		if(fileName == "mnist_train.csv") {
			labels = trainingLabels;
			images = trainingImages;
			numImages = numTrainingImages;
		}
		else if(fileName == "mnist_test.csv") {
			labels = testingLabels;
			images = testingImages;
			numImages = numTestingImages;
		}
		else return;
		
		Scanner scanner = null;
		try {
			scanner = new Scanner(new File(fileName));
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
		
		for(int image=0; image<numImages; image++) {
			String line = scanner.nextLine();				//get a line of data, which has the label preceding the image data
			String[] vals = line.split(",");			//split the line on the delimiter ","
			labels[image] = Integer.parseInt(vals[0]);	//get the label
			for(int j=1; j<a0.length+1; j++) {			//the next 28*28 values are greyscale values 0-255
				images[image][j-1] = Integer.parseInt(vals[j]);
			}
		}
		scanner.close();
	}
	
	//use csv and Scanner
	private static void loadData() {
		//get training data
		readFromFile("mnist_train.csv");
		
		//get testing data
		readFromFile("mnist_test.csv");
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
				setToZero(gb1);
				setToZero(gb2);
				setToZero(gw1);
				setToZero(gw2);
				
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
		
		//to use for calculating gradients later
		double[] tmpDelta1 = new double[a1.length];
		double[] tmpDelta2 = new double[a2.length];
		
		for(int j=0; j<a2.length; j++) {
			//set y according to if this is the correct classification
			int y = 0;
			if(j == (int)trainingLabels[inputImage]) {
				y = 1;
			}
			
			tmpDelta2[j] = (a2[j]-y)*a2[j]*(1-a2[j]);
		}
		
		//compute error for layer1
		//delta1k = sum_j delta2j*w2jk*ak*(1-ak)
		for(int k=0; k<a1.length; k++) {
			for(int j=0; j<a2.length; j++) {
				tmpDelta1[k] = tmpDelta1[k] + tmpDelta2[j]*w2[j][k]*a1[k]*(1-a1[k]);
			}
		}
		
		//calculate gradients
		for(int j=0; j<a2.length; j++) {	//dC/db2
			gb2[j] = gb2[j] + tmpDelta2[j];
		}
		for(int k=0; k<a1.length; k++) {	//dC/db1
			gb1[k] = gb1[k] + tmpDelta1[k];
		}
		for(int j=0; j<a2.length; j++) {	//dC/dw2
			for(int k=0; k<a1.length; k++) {
				gw2[j][k] = gw2[j][k] + tmpDelta2[j]*a1[k];
			}
		}
		for(int j=0; j<a1.length; j++) {	//dC/dw1
			for(int k=0; k<a0.length; k++) {
				gw1[j][k] = gw1[j][k] + tmpDelta1[j]*a0[k];
			}
		}
		
	}
	
	//gradients are divided by miniBatchSize because they are the sum of all gradients over the miniBatch
	private static void updateWeightsAndBiases() {
		//dC/db2 = delta2
		for(int j=0; j<a2.length; j++) {
			b2[j] = b2[j] - learningRate*gb2[j]/miniBatchSize;
		}
		
		//dC/db1 = delta1
		for(int k=0; k<a1.length; k++) {
			b1[k] = b1[k] - learningRate*gb1[k]/miniBatchSize;
		}
		
		//dC/dw2jk = delta2j*a1k
		for(int j=0; j<a2.length; j++) {
			for(int k=0; k<a1.length; k++) {
				w2[j][k] = w2[j][k] - learningRate*gw2[j][k]/miniBatchSize;
			}
		}
		
		//dC/dw1jk = delta1j*a0k
		for(int j=0; j<a1.length; j++) {
			for(int k=0; k<a0.length; k++) {
				w1[j][k] = w1[j][k] - learningRate*gw1[j][k]/miniBatchSize;
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
		
		int[][] imagesArray;
		int[] labelsArray;
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
	
	private static void setToZero(double[][] x) {
		for(int i=0; i<x.length; i++) {
			for(int j=0; j<x[0].length; j++) {
				x[i][j] = 0;
			}
		}
	}
	
	private static void setToZero(int[] x) {
		for(int i=0; i<x.length; i++) {
			x[i] = 0;
		}
	}
}
