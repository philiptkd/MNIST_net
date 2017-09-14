/*
 * TODO: create and train a neural net to classify handwritten digits from the MNIST data set
 * 
 * choose eps based on ulp?
 * 		eps = x0*sqrt(ulp)		(if x0 != 0)
 * create function to evaluate cost function at given weights/bias
 * restructure net to have weights/biases in one array
 * 
 * figure out how to find the gradient of a function
 * 		use forward difference rather than central to compute partial derivatives so we need only compute the cost function once per derivative
 * 		dC/dw_i = (C_{w_i + eps} - C_{w_i})/eps
 * 		C(w,b) = Sum_x ||y(x)-a||^2/(2n)
 * 			where a is the vector of outputs when x is input, y(x) is the desired output, and n is the numbe4 of training inputs
 * 		let V denote the current vector of weights and biases for the net
 * 		V' = V - eta*grad(C(V))
 * learn how to randomly select arrayList elements for stochastic gradient descent
 * at completion of training, test on testing set
 */

import java.io.IOException;
import java.util.Random;
import java.lang.Double;

public class Main {
	public static void main(String[] args) throws IOException {
		
//		Net myNet = new Net(28*28, 15, 10);
//		trainNet(myNet, 30, 10, 3.0);
	}
	
	private static void trainNet(Net myNet, int epochs, int miniBatchSize, double learningRate) throws IOException{
		double EPSILON = Math.sqrt(Math.pow(2, -53));	//to be used to calculate step size in differentiation
		
		//create a list that we can shuffle in order to randomize our mini-batches
		int[] shuffledList = new int[60000];
		for(int i=0; i<shuffledList.length; i++) {
			shuffledList[i] = i;
		}
		
		int numWeights = myNet.inputLayer.length*myNet.hiddenLayer.length + myNet.hiddenLayer.length*myNet.outputLayer.length;
		int numBiases = myNet.hiddenLayer.length + myNet.outputLayer.length;
		
		for(int epoch=0; epoch<epochs; epoch++) {	//for each epoch
			shuffle(shuffledList);		//shuffle the training set order
			for(int miniBatch=0; miniBatch<shuffledList.length/miniBatchSize; miniBatch++) {	//for each miniBatch
				int[] gradientVectorSum = new int[numWeights + numBiases];	//create zero vector for weights/biases
				//calculate cost function for current weights/biases
				for(int input=miniBatch*miniBatchSize; input<(miniBatch+1)*miniBatchSize; input++) {	//for each training input
					//compute partial derivatives dC/dw and dC/db for all weights w and biases b
					//add this input's gradient vector to gradient vector sum
				}
				//divide gradient vector sum by miniBatchSize to get average
				//update point in weight/bias space
			}			
		}
			
		//save final weights and biases for testing
		
		//check output of net for each test image against its label
		//compute error rate
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
}
