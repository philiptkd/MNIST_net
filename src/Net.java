/*
 * TODO: create and train a neural net to classify handwritten digits from the MNIST data set
 * 
 * figure out how to get the data
 * 		all at once? one piece at a time?	(all at once. the file is only about 46MB)
 * learn to access the data				(done)
 * create the Net class				(done)
 * 		constructor parameters?
 * figure out how to find the gradient of a function
 * learn how to randomly select arrayList elements for stochastic gradient descent
 * write method for getting net's output with given input and weights/biases
 * write sigmoid neuron class
 * at completion of training, write final weights/biases to a file
 * test on testing set
 */

import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.channels.FileChannel.MapMode;
import java.nio.file.Path;
import java.nio.file.Paths;

public class Net {
	MappedByteBuffer buffer;
	byte[] inputLayer;
	Neuron[] hiddenLayer;
	Neuron[] outputLayer;
	
	public Net(int numInputNodes, int numHiddenNodes, int numOutputNodes) throws IOException{		
		
		//initialize input layer
		Path path = Paths.get("train-images.idx3-ubyte");	
		FileChannel file = FileChannel.open(path);			//open file
		buffer = file.map(MapMode.READ_ONLY, 16, 60000);		//skip 16 byte header
		inputLayer = new byte[28*28];						//initialize byte array
		loadInputLayer(0);									//read bytes from buffer
		
		//initialize hidden layer
		hiddenLayer = new Neuron[numHiddenNodes];
		for(int i=0; i<hiddenLayer.length; i++) {
			hiddenLayer[i] = new Neuron(numInputNodes);
			hiddenLayer[i].layer = NetLayer.HIDDENLAYER;
		}
		
		//initialize output layer
		outputLayer = new Neuron[numOutputNodes];
		for(int i=0; i<outputLayer.length; i++) {
			outputLayer[i] = new Neuron(numHiddenNodes);
			outputLayer[i].layer = NetLayer.OUTPUTLAYER;
		}
	}
	
	//method to read from the MappedByteBuffer of the images file to the inputLayer
	public void loadInputLayer(int offset) {
		buffer.get(inputLayer, offset, 28*28);
	}
	
	//taking these inner products could probably be done more efficiently
	public void calcNodeOutput(Neuron node) {
		double input = 0;
		if(node.layer == NetLayer.HIDDENLAYER) {		//if we want the output of a hidden layer neuron
			for(int i=0; i<inputLayer.length; i++) {
				input = input + (double)((char)inputLayer[i])*node.inWeights[i];	//inner product of previous layer and their associated weights
			}
		}
		else if(node.layer == NetLayer.OUTPUTLAYER) {	//if we want the output of an output layer neuron
			for(int i=0; i<hiddenLayer.length; i++) {
				input = input + hiddenLayer[i].output*node.inWeights[i];	//inner product of previous layer and their associated weights
			}
		}
		input = input + node.bias;
		node.output = sigmoidFunction(input);
	}
	
	public double sigmoidFunction(double input) {
		return 1/(1+Math.pow(Math.E, -input));
	}
	
	//updates the output values of all neurons in the net
	public void calcNetOutput() {
		for(Neuron node : hiddenLayer) {
			calcNodeOutput(node);
		}
		for(Neuron node : outputLayer) {
			calcNodeOutput(node);
		}
	}
}
