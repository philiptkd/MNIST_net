import java.io.BufferedReader;
import java.io.FileReader;
import java.io.InputStreamReader;
import java.io.RandomAccessFile;
import java.util.Random;
import java.io.File;

/*
 * Philip Raeisghasem
 * 102 08 738
 * 17 October 2017
 * CSC 475
 * Assignment 2
 * 
 * This program implements a neural network to classify handwritten digits from the MNIST data set.
 * The network is trained with stochastic gradient descent and backpropagation.
 */

public class Main {
	//network constants
	public static boolean shuffling = true;
	public static int nodesInLayer0 = 28*28;
	public static int nodesInLayer1 = 30;
	public static int nodesInLayer2 = 10;
	public static int numTrainingImages = 50000;
	public static int numTestingImages = 10000;
	public static double initialLearningRate = 3.0;
	public static double finalLearningRate = 3.0;		//set this lower than initial to have a linear learning rate schedule 
	public static int epochs = 30;
	public static int miniBatchSize = 10;
	private static double lambda = 0.1;
	
	
	//arrays to hold b, w, dC/db, dC/dw, z, and a
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
	
	//arrays to hold training images and labels
	private static int[][] trainingImages = new int[numTrainingImages][nodesInLayer0];
	private static int[] trainingLabels = new int[numTrainingImages];
	
	//arrays to hold test images and labels
	private static int[][] testingImages = new int[numTestingImages][nodesInLayer0];
	private static int[] testingLabels = new int[numTestingImages];
	
	//used for accuracy statistics
	private static int[] labelCounts = new int[nodesInLayer2];
	private static int[] correctCounts = new int[nodesInLayer2];
	
	private static Random rand = new Random();
	private static boolean fancy = false;
	private static boolean earlyStopping = false;
	private static double lastAccuracy = 0.0;
	private static int numEpochsThatCanSuck = 3;
	private static int numEpochsThatSucked = 0;
	
	//global variables for cmd line arguments
	public static String trainCSVFileString;
	public static String testCSVFileString;
	public static String weightsFileString;
	
	public static void main(String[] args) {
		
		boolean argsGood = getArgs(args);
		if(!argsGood) {
			return;
		}
		
		boolean trained = false;
		System.out.println("This is a program that demonstrates a basic neural network that classifies handwritten digits from the MNIST data set. " + 
				"\nIt uses stochastic gradient descent with backpropagation to train the network.\n");
		while(true) {
			//print intro and user input options
			System.out.println("Choose from the options below.\n");
			System.out.println("[1] Train network");
			System.out.println("[2] Load pre-trained network");
			if(trained) {
				System.out.println("[3] Display network accuracy on training data");
				System.out.println("[4] Display network accuracy on testing data");
				System.out.println("[5] Save network state to file");
			}
			System.out.println("[6] Exit");
			
			//load data
			loadData();
			
			//wait for user input
			int selection = 0;
			BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
			try {
				selection = Integer.parseInt(br.readLine());
			} catch (Exception e) {
				System.out.println("Something went wrong. Please choose from the options below.");
			}
			
			//switch on user input to do the thing
			switch(selection) {
			case 0: //do nothing
				break;
			case 1: //train network
				fancyWeights();
				trainNet();
				trained = true;
				break;
			case 2: //load pre-trained network
				readFromFile();
				trained = true;
				break;
			case 3: //display network accuracy on training data
				if(trained) {
					printAccuracy(0);
				}
				break;
			case 4: //display network accuracy on testing data
				if(trained) {
					printAccuracy(1);
				}
				break;
			case 5: //save network state to file
				if(trained) {
					writeToFile();
				}
				break;
			case 6: //exit
				return;
			}
		}
	}
	
	//to check if the command line arguments are good
	private static boolean getArgs(String[] args) {
		//ensure there are three command line arguments
		if(args.length != 3) {
			System.out.println("This takes 3 arguments: trainCSVFile, testCSVFile, and weightsFile");
			return false;
		}
		
		//get command line arguments
		trainCSVFileString = args[0];
		testCSVFileString = args[1];
		weightsFileString = args[2];
		
		//see if files exist
		File trainFile = new File(trainCSVFileString);
		File testFile = new File(testCSVFileString);
		File weightsFile = new File(weightsFileString);
		
		//build error string to print
		String errorString = "";
		if(!trainFile.exists()) {
			errorString += trainCSVFileString + " not found.\n";
		}
		if(!testFile.exists()) {
			errorString += testCSVFileString + " not found.\n";
		}
		if(!weightsFile.exists()) {
			errorString += weightsFileString + " not found.\n";
		}
		
		if(errorString != "") {
			System.out.println(errorString);
			return false;
		}
		
		return true;
	}
	
	//reads the training and testing data from CSV files and into the static arrays defined above
	//also separates labels and image data
	private static void loadData() {
		BufferedReader br = null;
		String line;
		String[] splitLine;
		
		try {
			br = new BufferedReader(new FileReader(trainCSVFileString));
			for(int i=0; i<numTrainingImages; i++) {
				line = br.readLine();
				splitLine = line.split(",");
				trainingLabels[i] = Integer.parseInt(splitLine[0]);
				for(int j=0; j<trainingImages[0].length; j++) {
					trainingImages[i][j] = Integer.parseInt(splitLine[j+1]);
				}
			}
			br.close();
			
			br = new BufferedReader(new FileReader(testCSVFileString));
			for(int i=0; i<numTestingImages; i++) {
				line = br.readLine();
				splitLine = line.split(",");
				testingLabels[i] = Integer.parseInt(splitLine[0]);
				for(int j=0; j<testingImages[0].length; j++) {
					testingImages[i][j] = Integer.parseInt(splitLine[j+1]);
				}
			}
			br.close();
			
		}
		catch(Exception e) {
			System.out.println(e.getMessage());
		}
	}
	
	//a method for troubleshooting. 
	//prints all input images with ASCII text
	private static void printData() {
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
					else if (num < 225) c = 'x';
					else c = 'X';
					System.out.print(c);
				}
				System.out.println("");
			}
			System.out.println((int)trainingLabels[i]);
		}
	}
	
	//a method for troubleshooting
	//prints in ASCII text only the image currently loaded into the input layer
	//also prints the correct classification
	private static void printInputAndLabel(int correctClassification) {
		for(int j=0; j<28; j++) {
			for(int k=0; k<28; k++) {
				int num = (int)(a0[j*28 + k]*255.0);
				char c;
				if(j == 0 || j == 27) c = '-';
				else if(k == 0 || k == 27) c = '|';
				else if(num < 10) c = ' ';
				else if (num < 50) c = '.';
				else if (num < 100) c = 'l';
				else if (num < 150) c = 'h';
				else if (num < 200) c = '&';
				else if (num < 225) c = 'x';
				else c = 'X';
				System.out.print(c);
			}
			System.out.println("");
		}
		System.out.println(correctClassification);
	}
	
	//a method for troubleshooting
	//independently verifies a gradient in w1 using the image currently in the input layer
	private static void gradientChecking(int correctClassification) {
		double eps = 0.0001;
		//dC/db1
		w1[5][350] = w1[5][350] + eps;
		feedForward();
		double C1 = cost(correctClassification);
		
		w1[5][350] = w1[5][350] - 2*eps;
		feedForward();
		double C2 = cost(correctClassification);
		
		w1[5][350] = w1[5][350] + eps;
		
		
		System.out.println("dC/dw1[5][350] = " + (C1-C2)/(2*eps));
		System.out.println("gw1[5][350] = " + gw1[5][350]);
		System.out.println("");
	}
	
	//a method for troubleshooting
	//calculates the quadratic cost function for the image currently in the input layer
	//assumes already fed forward
	private static double cost(int correctClassification) {
		int[] y = new int[a2.length];
		y[correctClassification] = 1;
		
		double C = 0;
		for(int i=0; i<a2.length; i++) {
			C = C + Math.pow(y[i]-a2[i], 2);
		}
		C = C/2;
		
		return C;
	}
	
	//the old way of initializing weights and biases
	//nextDouble() gives a uniformly random number between 0.0 and 1.0
	//(rand.nextDouble()- 0.5)*2 gives a uniformly random number between -1.0 and 1.0
	private static void initializeBiasesAndWeights() {
		//b1		
		for(int i=0; i<b1.length; i++) {
			b1[i] = (rand.nextDouble()- 0.5)*2;
		}
		
		//b2
		for(int i=0; i<b2.length; i++) {
			b2[i] = (rand.nextDouble()- 0.5)*2;
		}
		
		//w1
		for(int k=0; k<a0.length; k++) {
			for(int j=0; j<a1.length; j++) {
				w1[j][k] = (rand.nextDouble()- 0.5)*2;
			}
		}
		
		//w2
		for(int k=0; k<a1.length; k++) {
			for(int j=0; j<a2.length; j++) {
				w2[j][k] = (rand.nextDouble()- 0.5)*2;
			}
		}
		
	}
	
	//a better way of initializing weights
	//we lower the variance for each weight in order to lower the variance of the activations in the next layer
	//how you initialize biases doesn't matter as much, so I left those the same
	private static void fancyWeights() {
		//b1 is initialized the same		
				for(int i=0; i<b1.length; i++) {
					b1[i] = (rand.nextDouble()- 0.5)*2;
				}
				
		//b2 is initialized the same
		for(int i=0; i<b2.length; i++) {
			b2[i] = (rand.nextDouble()- 0.5)*2;
		}
		
		//w1
		for(int k=0; k<a0.length; k++) {
			for(int j=0; j<a1.length; j++) {
				w1[j][k] = rand.nextGaussian()/Math.sqrt(a0.length);	//std dev. = 1/numInputNodes
			}
		}
		
		//w2
		for(int k=0; k<a1.length; k++) {
			for(int j=0; j<a2.length; j++) {
				w2[j][k] = rand.nextGaussian()/Math.sqrt(a1.length);	//std dev. = 1/humHiddenNodes
			}
		}
	}
	
	//the method that implements stochastic gradient descent
	private static void trainNet() {		
		//create a list that we can shuffle in order to randomize our mini-batches
		int[] shuffledList = new int[numTrainingImages];
		for(int i=0; i<shuffledList.length; i++) {
			shuffledList[i] = i;
		}
		
		for(int epoch=0; epoch<epochs; epoch++) {	//for each epoch
			if(shuffling) {
				shuffle(shuffledList);		//shuffle the training set order	
			}
			setToZero(labelCounts);
			setToZero(correctCounts);
			
			for(int miniBatch=0; miniBatch<shuffledList.length/miniBatchSize; miniBatch++) {	//for each miniBatch
				//initialize gradients to zero
				setToZero(gb1);
				setToZero(gb2);
				setToZero(gw1);
				setToZero(gw2);
				
				for(int input=miniBatch*miniBatchSize; input<(miniBatch+1)*miniBatchSize; input++) {	//for each input image
					//load a0
					for(int i=0; i<a0.length; i++) {
						a0[i] = (double)(trainingImages[shuffledList[input]][i])/255.0;	//scale to 0-1
						if(a0[i] < 0 || a0[i] > 1)		//for troubleshooting. if this is true, something about our input is wrong
						{
							System.out.println(a0[i]);
						}
					}
					//load correct classification
					int correctClassification = (int)trainingLabels[shuffledList[input]];
					
					//feed forward
					feedForward();
					
					//see if it classified correctly
					checkOutputAccuracy(correctClassification);
					
					//backpropagate
					backpropagate(correctClassification);
					
					//check gradient against numerical calculation. only valid for first input in minibatch
//					if(input == miniBatch*miniBatchSize) {
//						gradientChecking((int)trainingLabels[shuffledList[input]]);
//					}

				}
				//update weights/biases
				if(fancy) {
					updateFancyWeights(epoch);
				}
				else {
					updateWeightsAndBiases(epoch);
				}
				
				//printWeightsAndBiases();
			}
			System.out.println("epoch " + epoch + ": ");
			printAccuracyStatistics();
			System.out.println("");
			
			//implements early stopping. 
			//when the classification accuracy against the test set stops improving, we stop training
			//prevents overfitting
			if(earlyStopping) {
				double testAccuracy = checkTestAccuracy();
				if(testAccuracy > lastAccuracy) {
					lastAccuracy = testAccuracy;
					numEpochsThatSucked = 0;
				}
				else {
					numEpochsThatSucked++;
					if(numEpochsThatSucked > numEpochsThatCanSuck) {
						return;
					}
				}
			}

		}
	}
	
	//a method for troubleshooting
	//prints out the current weights and biases of the network
	private static void printWeightsAndBiases() {
		//b2
		for(int j=0; j<a2.length; j++) {
			System.out.print(b2[j] + " ");
		}
		System.out.println("");
		
		//b1
		for(int j=0; j<a1.length; j++) {
			System.out.print(b1[j] + " ");
		}
		System.out.println("");
		
		//w2
//		for(int j=0; j<a2.length; j++) {
//			for(int k=0; k<a1.length; k++) {
//				System.out.print(w2[j][k] + " ");
//			}
//			System.out.println("");
//		}
//		System.out.println("");
//		
//		//w1
//		for(int j=0; j<a1.length; j++) {
//			for(int k=0; k<a0.length; k++) {
//				System.out.print(w1[j][k] + " ");
//			}
//			System.out.println("");
//		}
//		System.out.println("");
	}

	//calculates a and z for the hidden layer and the output layer given an input image
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
	
	//computes the gradients of the weights and biases
	//uses a quadratic cost function
	private static void backpropagate(int correctClassification) {		
		//to use for calculating gradients later
		double[] tmpDelta1 = new double[a1.length];
		double[] tmpDelta2 = new double[a2.length];
		
		//set the one-hot vector
		int[] y = new int[a2.length];	//initializes to all 0s
		y[correctClassification] = 1;
		
		//compute error for layer2
		//delta2j = (a-y)a(1-a)
		for(int j=0; j<a2.length; j++) {
			tmpDelta2[j] = (a2[j]-y[j])*a2[j]*(1-a2[j]);
		}
		
		//compute error for layer1
		//here, the convention I've been using for indices is switched
			//that is, j indexes into a1 and k into a2
			//this is so I can have it match Nielsen's online book, chpt 2, eq. 45
		//delta1j = sum_k delta2k*w2kj*aj*(1-aj)
		for(int j=0; j<a1.length; j++) {
			for(int k=0; k<a2.length; k++) {	//sum over k
				tmpDelta1[j] = tmpDelta1[j] + tmpDelta2[k]*w2[k][j];
			}
			tmpDelta1[j] = tmpDelta1[j]*a1[j]*(1-a1[j]);	//multiply by sigma prime
		}
		
		//calculate gradients
		//they are added to the existing arrays, because we want the sum of the gradient over the whole minibatch
		//the gradient arrays are zeroed before every minibatch
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
	
	//a method for calculating weight and bias gradients using a cross-entropy cost function
	private static void fancyBackpropagate(int correctClassification) {
		//to use for calculating gradients later
		double[] tmpDelta1 = new double[a1.length];
		double[] tmpDelta2 = new double[a2.length];
		
		//set the one-hot vector
		int[] y = new int[a2.length];	//initializes to all 0s
		y[correctClassification] = 1;
		
		//compute error for layer2
		//delta2j = (a-y)
		for(int j=0; j<a2.length; j++) {
			tmpDelta2[j] = (a2[j]-y[j]);
		}
		
		//compute error for layer1
		//here, the convention I've been using for indices is switched
			//that is, j indexes into a1 and k into a2
			//this is so I can have it match Nielsen's online book, chpt 2, eq. 45
		//delta1j = sum_k delta2k*w2kj*aj*(1-aj)
		for(int j=0; j<a1.length; j++) {
			for(int k=0; k<a2.length; k++) {	//sum over k
				tmpDelta1[j] = tmpDelta1[j] + tmpDelta2[k]*w2[k][j];
			}
			tmpDelta1[j] = tmpDelta1[j]*a1[j]*(1-a1[j]);	//multiply by sigma prime
		}
		
		//calculate gradients
		//they are added to the existing arrays, because we want the sum of the gradient over the whole minibatch
		//the gradient arrays are zeroed before every minibatch
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
	
	//method for learning without regularization
	//gradients are divided by miniBatchSize because they are the sum of all gradients over the miniBatch
	private static void updateWeightsAndBiases(int epoch) {
		//learning rate schedule
		double learningRate = finalLearningRate - (finalLearningRate - initialLearningRate)*((double)epoch)/((double)epochs);
		
		//b2
		for(int j=0; j<a2.length; j++) {
			b2[j] = b2[j] - learningRate*gb2[j]/miniBatchSize;
		}
		
		//b1
		for(int k=0; k<a1.length; k++) {
			b1[k] = b1[k] - learningRate*gb1[k]/miniBatchSize;
		}
		
		//w2
		for(int j=0; j<a2.length; j++) {
			for(int k=0; k<a1.length; k++) {
				w2[j][k] = w2[j][k] - learningRate*gw2[j][k]/miniBatchSize;
			}
		}
		
		//w1
		for(int j=0; j<a1.length; j++) {
			for(int k=0; k<a0.length; k++) {
				w1[j][k] = w1[j][k] - learningRate*gw1[j][k]/miniBatchSize;
			}
		}
	}
	
	//the new method to update weights and biases
	//uses L2 regularization to add weight decay during the update
	private static void updateFancyWeights(int epoch) {
		updateWeightsAndBiases(epoch);	//do the regular update
		double learningRate = finalLearningRate - (finalLearningRate - initialLearningRate)*((double)epoch)/((double)epochs);
		double weightDecay = learningRate*lambda/numTrainingImages;
		
		//w2. add L2 regularization
		for(int j=0; j<a2.length; j++) {
			for(int k=0; k<a1.length; k++) {
				w2[j][k] = w2[j][k] - weightDecay*w2[j][k];
			}
		}
		
		//w1. add L2 regularization
		for(int j=0; j<a1.length; j++) {
			for(int k=0; k<a0.length; k++) {
				w1[j][k] = w1[j][k] - weightDecay*w1[j][k];
			}
		}
		
	}
	
	//used to load saved network parameters from a file
	private static void readFromFile() {
		RandomAccessFile reader = null;
		try {
			reader = new RandomAccessFile(weightsFileString, "r");
			
			//b1
			for(int k=0; k<b1.length; k++) {
				b1[k] = reader.readDouble();
			}
			
			//b2
			for(int j=0; j<b2.length; j++) {
				b2[j] = reader.readDouble();
			}
			
			//w1
			for(int j=0; j<a1.length; j++) {
				for(int k=0; k<a0.length; k++) {
					w1[j][k] = reader.readDouble();
				}
			}
			
			//w2
			for(int j=0; j<a2.length; j++) {
				for(int k=0; k<a1.length; k++) {
					w2[j][k] = reader.readDouble();
				}
			}
			
			reader.close();
		}
		catch(Exception e) {
			System.out.println(e.getMessage());
		}
	}
	
	//used to save network parameters to a file
	private static void writeToFile() {
		RandomAccessFile writer = null;
		try {
			writer = new RandomAccessFile(weightsFileString, "rw");
			
			//b1
			for(int k=0; k<b1.length; k++) {
				writer.writeDouble(b1[k]);
			}
			
			//b2
			for(int j=0; j<b2.length; j++) {
				writer.writeDouble(b2[j]);
			}
			
			//w1
			for(int j=0; j<a1.length; j++) {
				for(int k=0; k<a0.length; k++) {
					writer.writeDouble(w1[j][k]);
				}
			}
			
			//w2
			for(int j=0; j<a2.length; j++) {
				for(int k=0; k<a1.length; k++) {
					writer.writeDouble(w2[j][k]);
				}
			}
			
			writer.close();
		}
		catch(Exception e) {
			System.out.println(e.getMessage());
		}
	}
	
	//the sigmoid function
	private static double sigma(double z) {
		return 1.0/(1.0+Math.pow(Math.E, -z));
	}
	
	//shuffles a list
	private static void shuffle(int[] list) {
		for(int i=list.length-1; i>=0; i--) {
			int index = rand.nextInt(i+1);
			int tmp = list[index];
			list[index] = list[i];
			list[i] = tmp;
		}
	}
	
	//method used to check classification accuracy against the test set. 
	//for early stopping
	private static double checkTestAccuracy() {
		int numCorrect = 0;
		
		for(int image=0; image<numTestingImages; image++) {	//for each image
			//load a0
			for(int pixel=0; pixel<a0.length; pixel++) {
				a0[pixel] = (double)testingImages[image][pixel]/255.0;	//scale to 0-1
			}
			
			//feed forward
			feedForward();
			
			//the correct answer
			int correctClassification = (int)testingLabels[image];
			
			//the computed answer
			//find highest activation in output layer
			double highest = Double.NEGATIVE_INFINITY;
			int classification = 0;	//arbitrary
			for(int j=0; j<a2.length; j++) {
				if(a2[j] > highest) {
					highest = a2[j];
					classification = j;
				}
			}
			
			//compare with net output
			if(classification == correctClassification) {
				numCorrect = numCorrect + 1;
			}
		}
		return (double)numCorrect/(double)numTestingImages;
	}
	
	
	//method used to check classification accuracy against either the training or testing set
	//also prints detailed accuracy statistics
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
				a0[pixel] = (double)imagesArray[image][pixel]/255.0;	//scale to 0-1
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
	
	//prints how well the network did in classifying each digit
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
	
	//increments the counts for each digit and for each correct classification
	private static void checkOutputAccuracy(int correctClassification) {
		//find highest activation in output layer
		double highest = Double.NEGATIVE_INFINITY;
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
	
	//sets an array to all 0s
	private static void setToZero(double[] x) {
		for(int i=0; i<x.length; i++) {
			x[i] = 0;
		}
	}
	
	//sets a 2-dimensional array to all 0s
	private static void setToZero(double[][] x) {
		for(int i=0; i<x.length; i++) {
			for(int j=0; j<x[0].length; j++) {
				x[i][j] = 0;
			}
		}
	}
	
	//sets an array of integers to all 0s
	private static void setToZero(int[] x) {
		for(int i=0; i<x.length; i++) {
			x[i] = 0;
		}
	}
}



