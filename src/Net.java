import java.io.IOException;
import java.nio.ByteBuffer;
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
	ByteBuffer labels;
	Neuron[][] layers;
	
	public Net(int numInputNodes, int numHiddenNodes, int numOutputNodes) {		
		
		try {
			//initialize input layer
			Path imgPath = Paths.get("train-images.idx3-ubyte");	
			FileChannel imgFile = null;
			imgFile = FileChannel.open(imgPath);			//open file
			buffer = imgFile.map(MapMode.READ_ONLY, 16, 60000*28*28);		//skip 16 byte header
			inputLayer = new byte[28*28];						//initialize byte array
			if(imgFile != null) {
				imgFile.close(); 			//close file
			}
		} catch(Exception e) {
			System.out.println(e.toString());
		}
		
		
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
		
		try {
			//get correct labels for each input image
			labels = ByteBuffer.allocate(60000);	//sets the number of bytes to be held in the buffer
			Path labelPath = Paths.get("train-labels.idx1-ubyte");	
			FileChannel labelFile = null;
			labelFile = FileChannel.open(labelPath);			//open file
			labelFile.read(labels, 8);		//attempts to read as many bytes into labels ByteBuffer within its limit
			if(labelFile != null) {
				labelFile.close();
			}
		} catch(Exception e) {
			System.out.println(e.toString());
		}
		
		//initialize layers array
		layers = new Neuron[2][];
		layers[0] = hiddenLayer;
		layers[1] = outputLayer;
	}
	
	//method to read from the MappedByteBuffer of the images file to the inputLayer
	public void loadInputLayer(int input) {
		buffer.position(input*28*28);	//set next read position in buffer
		buffer.get(inputLayer);		//reads as many bytes as can fit in inputLayer array
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
	
	//C(w,b) = Sum_x ||y(x)-a||^2/(2n)
	public double miniBatchCost(int miniBatch, int miniBatchSize, int[] shuffledList) {
		double cost = 0.0;
		double workingVector[] = new double[outputLayer.length];	//vector for y(x) - a
		
		for(int input=miniBatch*miniBatchSize; input<(miniBatch+1)*miniBatchSize; input++) {	//for each training input, x
			loadInputLayer(input);	//puts training image data in input layer
			calcNetOutput();	//find output, a	(stored in output nodes)
			double sumSquares = 0.0;	//to hold ||y(x)-a||^2 for each input image, x
			
			for(int i=0; i<workingVector.length; i++) {	
				//y(x) - a
				//y(x) has 1.0 for expected output index and 0.0 for all others
				if(i == labels.get(input)) {
					workingVector[i] = 1.0 - outputLayer[i].output;
				}
				else { workingVector[i] = 0.0 - outputLayer[i].output; }
				
				//sum squares of difference vector to get ||y(x)-a||^2
				sumSquares = sumSquares + workingVector[i]*workingVector[i];
			}
			
			cost = cost + sumSquares;	//Sum_x ||y(x)-a||^2
		}
		cost = cost/(2*miniBatchSize);	//divide by 2n
		
		return cost;
	}
	
	//dC/dw_i = (C_{w_i + eps} - C_{w_i})/eps
	public void SGD(int miniBatch, int miniBatchSize, int[] shuffledList, double learningRate) {
		double cost = miniBatchCost(miniBatch, miniBatchSize, shuffledList);	//calculate cost function for current weights/biases and miniBatch
		
		//update each weight/bias
		for(Neuron[] layer : layers) {
			for(Neuron node : layer) {		//for each node in the net
				//for the node's weights
				for(int i=0; i<node.inWeights.length; i++) {
					double eps = calcEps(node.inWeights[i]);
					node.inWeights[i] = node.inWeights[i] + eps;	//change to w_i + eps
					double newCost = miniBatchCost(miniBatch, miniBatchSize, shuffledList);		//C_{w_i + eps}
					double partialW = (newCost - cost)/eps;		//dC/dw_i = (C_{w_i + eps} - C_{w_i})/eps
					node.inWeights[i] = node.inWeights[i] - eps - learningRate*partialW;	//w_i' = w_i - learningRate*dC/dw_i 
				}
				//for the node's bias
				double eps = calcEps(node.bias);
				node.bias = node.bias + eps;
				double newCost = miniBatchCost(miniBatch, miniBatchSize, shuffledList);		//C_{w_i + eps}
				double partialB = (newCost - cost)/eps;		//dC/dw_i = (C_{w_i + eps} - C_{w_i})/eps
				node.bias = node.bias - eps - learningRate*partialB;	//b_i' = b_i - learningRate*dC/db_i
			}
		}
	} 
	
	//EPSILON = x0*sqrt(Math.pow(2, -53))
	//ensures round-off and truncation errors are of the same order
	public static double calcEps(double x0) {
		double eps;
		double sqrtUlp = Math.sqrt(Math.pow(2, -53));	//the square root of the precision of the double type
		eps = x0*sqrtUlp;
		
		if(eps != 0.0) {
			return eps;
		}
		else {
			return 0.01*sqrtUlp;	//arbitrary
		}
	}
	
	public double getErrorRate() throws IOException {		
		//fill buffer with test set data
		int numTestImages = 10000;
		Path imgPath = Paths.get("t10k-images.idx3-ubyte");	
		FileChannel imgFile = null;
		imgFile = FileChannel.open(imgPath);			//open file
		buffer = imgFile.map(MapMode.READ_ONLY, 16, numTestImages*28*28);		//skip 16 byte header
		if(imgFile != null) {
			imgFile.close(); 			//close file
		}
		
		//get correct labels for each test image
		labels.limit(10000);		//sets the number of bytes to be held in the buffer
		Path labelPath = Paths.get("t10k-labels.idx1-ubyte");	
		FileChannel labelFile = null;
		labelFile = FileChannel.open(labelPath);			//open file
		labelFile.read(labels, 8);		//attempts to read as many bytes into the labels ByteBuffer within its limit
		if(labelFile != null) {
			labelFile.close();
		}
		
		int numWrong = 0;	//the number of test images the net classifies incorrectly
		
		for(int i=0; i<numTestImages; i++) {	//for each test image
			loadInputLayer(i);	//load test image into input layer
			calcNetOutput();	//get the output for this test image
			int output = 0;		//the index of the output node with highest activation energy
			double highest = 0.0; 	//said highest activation energy
			for(int j=0; j<outputLayer.length; j++) {	//for each output node
				if(outputLayer[j].output > highest) {	//find the one with highest activation energy
					highest = outputLayer[j].output;
					output = j;
				}
			}
			//check if it matches the label
			if(output != labels.get(i)) {
				numWrong++;
			}
		}
		
		return ((double)numWrong)/(double)numTestImages;
	}
}
