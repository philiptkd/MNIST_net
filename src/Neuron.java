import java.util.Random;

public class Neuron {
	NetLayer layer;
	double[] inWeights;
	double bias;
	double output;
	
	public Neuron(int numInputs) {
		inWeights = new double[numInputs];
		Random rand = new Random();
		
		//initialize weights to random double, uniformly distributed with mean 0 and std dev 1
		for(int i=0; i<numInputs; i++) {
			inWeights[i] = rand.nextGaussian();
		}
		
		//initialize bias the same way
		bias = rand.nextGaussian();
	}
}