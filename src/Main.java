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

import java.io.IOException;
import java.util.Random;

public class Main {
	public static void main(String[] args) throws IOException{
		
		Net myNet = new Net(28*28, 15, 10);
		trainNet(myNet, 30, 10, 3.0);
		System.out.println(myNet.getErrorRate());
	}
	
	private static void trainNet(Net myNet, int epochs, int miniBatchSize, double learningRate) {		
		//create a list that we can shuffle in order to randomize our mini-batches
		int[] shuffledList = new int[60000];
		for(int i=0; i<shuffledList.length; i++) {
			shuffledList[i] = i;
		}
			
		for(int epoch=0; epoch<epochs; epoch++) {	//for each epoch
			shuffle(shuffledList);		//shuffle the training set order
			for(int miniBatch=0; miniBatch<shuffledList.length/miniBatchSize; miniBatch++) {	//for each miniBatch
				myNet.SGD(miniBatch, miniBatchSize, shuffledList, learningRate);	//update weights/biases via stochastic gradient descent
			}			
		}
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
