import java.io.BufferedReader;
import java.io.FileReader;
import java.util.Random;

public class Main {
	//commented are the example numbers. 
	//to match example, turn off shuffling, don't scale a0 to 0-1, and don't call loadData() or initializeBiasesAndWeights().
	public static boolean shuffling = true;
	public static int nodesInLayer0 = 28*28;//4;//
	public static int nodesInLayer1 = 30;//3;//
	public static int nodesInLayer2 = 10;//2;//
	public static int numTrainingImages = 60000;//4;//
	public static int numTestingImages = 10000;
	public static double learningRate = 3.0;//10.0;//
	public static int epochs = 30;//6;//
	public static int miniBatchSize = 10;//2;//
	
	
	//a, z, delta, b, w
	private static double[] a0 = new double[nodesInLayer0];
	private static double[] a1 = new double[nodesInLayer1];
	private static double[] a2 = new double[nodesInLayer2];
	
	private static double[] z1 = new double[nodesInLayer1];
	private static double[] z2 = new double[nodesInLayer2];
	
	private static double[] b1 = new double[nodesInLayer1];//{0.1,-0.36,-0.31};//
	private static double[] b2 = new double[nodesInLayer2];//{0.16, -0.46};//
	private static double[][] w1 = new double[nodesInLayer1][nodesInLayer0];//{{-0.21, 0.72, -0.25, 1},{-0.94, -0.41, -0.47, 0.63},{0.15, 0.55, -0.49, -0.75}};//
	private static double[][] w2 = new double[nodesInLayer2][nodesInLayer1];//{{0.76,0.48,-0.73},{0.34,0.89,-0.23}};//
	
	private static double[] gb1 = new double[nodesInLayer1];
	private static double[] gb2 = new double[nodesInLayer2];
	private static double[][] gw1 = new double [nodesInLayer1][nodesInLayer0];	
	private static double[][] gw2 = new double [nodesInLayer2][nodesInLayer1];
	
	//training images and labels
	private static char[][] trainingImages = new char[numTrainingImages][nodesInLayer0];//{{0,1,0,1},{1,0,1,0},{0,0,1,1},{1,1,0,0}};//
	private static char[] trainingLabels = new char[numTrainingImages];//{1,0,1,0};//
	
	//test images and labels
	private static char[][] testingImages = new char[numTestingImages][nodesInLayer0];
	private static char[] testingLabels = new char[numTestingImages];
	
	//used for accuracy statistics
	private static int[] labelCounts = new int[nodesInLayer2];
	private static int[] correctCounts = new int[nodesInLayer2];
	
	private static Random rand = new Random();
	
	
	public static void main(String[] args) {
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
					else if (num < 225) c = 'x';
					else c = 'X';
					System.out.print(c);
				}
				System.out.println("");
			}
			System.out.println((int)trainingLabels[i]);
		}
	}
	
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
			for(int i=0; i<trainingImages.length; i++) {		//for each image in the file
				br.read(trainingImages[i], 0, trainingImages[0].length);	//(array to store them in, offset in array, number of bytes to get)
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
						a0[i] = (double)trainingImages[shuffledList[input]][i]/255.0;	//scale to 0-1
					}
					
					//feed forward
					feedForward();
					
					//see if it classified correctly
					checkOutputAccuracy((int)trainingLabels[shuffledList[input]]);
					
					//backpropagate
					backpropagate((int)trainingLabels[shuffledList[input]]);
					
					//check gradient against numerical calculation. only valid for first input in minibatch
					if(input == miniBatch*miniBatchSize) {
						gradientChecking((int)trainingLabels[shuffledList[input]]);
					}

				}
				//update weights/biases
				updateWeightsAndBiases();
				//printWeightsAndBiases();
			}
			System.out.println("epoch " + epoch + ": ");
			printAccuracyStatistics();
			System.out.println("");
		}
	}
	
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
	
	//gradients are divided by miniBatchSize because they are the sum of all gradients over the miniBatch
	private static void updateWeightsAndBiases() {
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
