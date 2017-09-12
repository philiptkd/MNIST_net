import java.io.IOException;

public class Main {
	public static void main(String[] args) throws IOException {
		Net myNet = new Net(784, 15, 10);
		myNet.calcNetOutput();
		
		for(Neuron node : myNet.outputLayer) {
			System.out.println(node.output);
		}
		//for a certain number of times
			//compute the gradient 
			//update point in weight/bias space
		
		//save final weights and biases for testing
		
		//check output of net for each test image against its label
		//compute error rate
	}
}
