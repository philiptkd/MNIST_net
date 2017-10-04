import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.channels.FileChannel.MapMode;
import java.nio.file.Path;
import java.nio.file.Paths;
import org.ojalgo.matrix.*;
import org.ojalgo.matrix.BasicMatrix.Builder;
import org.ojalgo.random.Normal;

public class Net {
	MappedByteBuffer buffer;
	byte[] inputLayer;
	ByteBuffer labels;
	
	//factory to initialize arrays and matrices
    final BasicMatrix.Factory<PrimitiveMatrix> matrixFactory = PrimitiveMatrix.FACTORY;
	
	//arrays to hold a, z, b, and delta values
	PrimitiveMatrix  inputLayerA;
	
	PrimitiveMatrix  hiddenLayerA;
	PrimitiveMatrix  hiddenLayerZ;
	PrimitiveMatrix  hiddenLayerB;
	PrimitiveMatrix  hiddenLayerDelta;
	
	PrimitiveMatrix  outputLayerA;
	PrimitiveMatrix  outputLayerZ;
	PrimitiveMatrix  outputLayerB;	
	PrimitiveMatrix  outputLayerDelta;
	
	
	//matrices to hold weights
	PrimitiveMatrix  weights1;
	PrimitiveMatrix  weights2;
	
	public Net(int numInputNodes, int numHiddenNodes, int numOutputNodes) {		
		inputLayer = new byte[numInputNodes];						//initialize byte array

		try {
			//initialize input layer
			Path imgPath = Paths.get("train-images.idx3-ubyte");	
			FileChannel imgFile = null;
			imgFile = FileChannel.open(imgPath);			//open file
			buffer = imgFile.map(MapMode.READ_ONLY, 16, 60000*numInputNodes);		//skip 16 byte header
			if(imgFile != null) {
				imgFile.close(); 			//close file
			}
		} catch(Exception e) {
			System.out.println(e.toString());
		}
				
        //initialize input layer
        inputLayerA = matrixFactory.makeZero(numInputNodes, 1);
        
		//initialize hidden layer
		hiddenLayerA = matrixFactory.makeZero(numHiddenNodes, 1);
		hiddenLayerZ = matrixFactory.makeZero(numHiddenNodes, 1);
		hiddenLayerB = matrixFactory.makeFilled(numHiddenNodes, 1, new Normal());
		
		//initialize output layer
		outputLayerA = matrixFactory.makeZero(numOutputNodes, 1);
		outputLayerZ = matrixFactory.makeZero(numOutputNodes, 1);
		outputLayerB = matrixFactory.makeFilled(numOutputNodes, 1, new Normal());

		//initialize weight matrices
	    weights1 = matrixFactory.makeFilled(numHiddenNodes, numInputNodes, new Normal());
	    weights2 = matrixFactory.makeFilled(numOutputNodes, numHiddenNodes, new Normal());

		
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
	}
	
	//method to read from the MappedByteBuffer of the images file to the inputLayer
	public void loadInputLayer(int input) {
		buffer.position(input*inputLayer.length);	//set next read position in buffer
		buffer.get(inputLayer);		//reads as many bytes as can fit in inputLayer array
		
		//put input data into usable matrix 
		//there is probably a better way to get MappedByteBuffer into a PrimitiveMatrix
		Builder<PrimitiveMatrix> matrixBuilder = inputLayerA.copy();	//PrimitiveMatrix is immutable
		
		for(int i=0; i<inputLayer.length; i++) {
			matrixBuilder.add(i, 0, inputLayer[i]);
		}
		
		inputLayerA = matrixBuilder.build();
	}
	
	public double sigma(double input) {
		return 1.0/(1.0+Math.pow(Math.E, -input));
	}
	
	//updates the output values of all neurons in the net
	public void calcNetOutput() {
		//z_1 = w_1*a_0 + b_1
		hiddenLayerZ = weights1.multiply(inputLayerA);
		hiddenLayerZ = hiddenLayerZ.add(hiddenLayerB);
		
		Builder<PrimitiveMatrix> matrixBuilder = hiddenLayerA.copy();	//PrimitiveMatrix is immutable
		int numHiddenNodes = (int)hiddenLayerA.countRows();
		
		//a_1 = sigma(z_1)
		for(int i=0; i<numHiddenNodes; i++) {
			matrixBuilder.add(i, 0, sigma(hiddenLayerZ.get(i, 0)));
		}
		hiddenLayerA = matrixBuilder.build();
		
		//z_2 = w_2*a_1 + b_2
		outputLayerZ = weights2.multiply(hiddenLayerA);	
		outputLayerZ = outputLayerZ.add(outputLayerB);					//outputLayerZ = outputLayerZ + outputLayerB
		
		matrixBuilder = outputLayerA.copy();	//PrimitiveMatrix is immutable
		int numOutputNodes = (int)outputLayerA.countRows();
		
		//a_2 = sigma(z_2)
		for(int i=0; i<numOutputNodes; i++) {
			matrixBuilder.add(i, 0, sigma(outputLayerZ.get(i, 0)));	
		}
		outputLayerA = matrixBuilder.build();
	}
	
	public void SGD(int miniBatch, int miniBatchSize, int[] shuffledList, double learningRate) {
		//initialize delta arrays
		hiddenLayerDelta = matrixFactory.makeZero(hiddenLayerA.countRows(), 1);
		outputLayerDelta = matrixFactory.makeZero(outputLayerA.countRows(), 1);
		
		for(int input=miniBatch*miniBatchSize; input<(miniBatch+1)*miniBatchSize; input++) {
			loadInputLayer(input);
			calcNetOutput();
			backpropagate(input);
		}
		
		//divide all elements in delta matrices by miniBatchSize to get averages
		hiddenLayerDelta.divide(miniBatchSize);
		outputLayerDelta.divide(miniBatchSize);
		
		//V' = V - learningRate*grad(C(V))
		
		//update b values. dC/db = delta
		outputLayerB = outputLayerB.subtract(outputLayerDelta.multiply(learningRate));	//b = b - learningRate*outDelta
		hiddenLayerB = hiddenLayerB.subtract(hiddenLayerDelta.multiply(learningRate));	//b = b - learningRate*hidDelta
		
		//update weights. dC/dw = (delta^L)* (a^(L-1))^T
		weights2 = weights2.subtract(outputLayerDelta.multiply(hiddenLayerA.transpose()).multiply(learningRate)); 	//w = w - learningRate*outDelta*aHid^T
		weights1 = weights1.subtract(hiddenLayerDelta.multiply(inputLayerA.transpose()).multiply(learningRate));	//w = w - learningRate*hidDelta*aIn^T
	} 
	
	public void backpropagate(int input) {
		//calculate output deltas, adding to outputLayerDelta
		int correctOutput = labels.get(input);				//correct classification for this input
		
		//creating Y vector
		int outputRows = (int)outputLayerA.countRows(); 	//number of elements in output layer
		PrimitiveMatrix outputLayerY = matrixFactory.makeZero(outputRows, 1);
		Builder<PrimitiveMatrix> matrixBuilder = outputLayerY.copy(); 		//mutable copy
		matrixBuilder.add(correctOutput, 0, 1.0); 		//make the one correct element of Y 1
		outputLayerY = matrixBuilder.build();
		
		//delta = (y-a)*a(a-1)			(all operations are element-wise)
		PrimitiveMatrix tmpOutDelta = matrixFactory.makeZero(outputRows, 0);
		tmpOutDelta = outputLayerY.subtract(outputLayerA);	//tmp = (y-a)
		tmpOutDelta = tmpOutDelta.multiplyElements(outputLayerA);	//tmp = (y-a)*a
		tmpOutDelta = tmpOutDelta.multiplyElements(outputLayerA.subtract(1.0));	//tmp = (y-a)*a*(a-1)
		
		//add to output delta vector
		outputLayerDelta = outputLayerDelta.add(tmpOutDelta);
		
		
		//calculate hidden deltas, adding to hiddenLayerDelta
		int hiddenRows = (int)hiddenLayerA.countRows();		//number of elements in hidden layer
		PrimitiveMatrix tmpHiddenDelta = matrixFactory.makeZero(hiddenRows, 1);
		
		//delta^L = (w^(L_1))^T * delta^(L+1) . a(1-a)		(where . is element-wise multiplication)
		tmpHiddenDelta = (weights2.transpose()).multiply(tmpOutDelta);	//w^T*outDelta
		tmpHiddenDelta = tmpHiddenDelta.multiplyElements(hiddenLayerA); //w^T*outDelta . a
		tmpHiddenDelta = tmpHiddenDelta.multiplyElements(hiddenLayerA.negate().add(1.0)); //w^T*outDelta . a . (-a + 1)
		
		//add to hidden delta vector
		hiddenLayerDelta = hiddenLayerDelta.add(tmpHiddenDelta);
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
		labels.limit(numTestImages);		//sets the number of bytes to be held in the buffer
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
			for(int j=0; j<outputLayerA.countRows(); j++) {	//for each output node
				if(outputLayerA.get(j, 0) > highest) {	//find the one with highest activation energy
					highest = outputLayerA.get(j, 0);
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
