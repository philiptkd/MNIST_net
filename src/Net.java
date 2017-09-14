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
	public void loadInputLayer(int input) {
		buffer.get(inputLayer, input*28*28, 28*28);
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
		return 1.0/(1.0+Math.pow(Math.E, -input));
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
