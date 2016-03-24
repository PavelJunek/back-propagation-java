/*
 * Copyright (C) 2016 Pavel Junek
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.
 */
package ai;

/**
 * This class represents one layer of the network.
 *
 * The layer is represented as one single class (there is no class Neuron for
 * individual neurons), because all the calculations on a layer can be
 * represented by simple matrix operations. The introduction of Neuron class
 * would bring additional complexity to the programm.
 *
 * The layer contains a two-dimensional array containing the weights of all
 * neurons in one table. Each line in this table contains the vector of input
 * weights of one neuron.
 *
 * This class represents a layer where all neurons use the logistic function,
 * but it can be easily modified for any other activation function by overriding
 * the methods activation() and derivation().
 *
 * @author Pavel Junek
 */
public class Layer {

	/**
	 * The number of inputs of the layer.
	 */
	private final int mNumberOfInputs;

	/**
	 * The number of neurons in the layer.
	 */
	private final int mNumberOfNeurons;

	/**
	 * The matrix of weights of all neurons in the layer. Each line in the
	 * matrix represents the input weights of one neuron.
	 */
	private final double[][] mWeights;

	/**
	 * An array of inputs of the layer. The inputs are set in the feed-forward
	 * step and must be remembered for the updating of weights.
	 */
	private double[] mInputs;

	/**
	 * An array of calculated outputs. The outputs are set in the feed-forward
	 * step and must be remembered for the back-propagation step
	 */
	private double[] mOutputs;

	/**
	 * The errors propagated to the post-synaptic potential. These errors are
	 * set in the back-propagation step and must be remembered for the updating
	 * of weights.
	 */
	private double[] mDeltas;

	/**
	 * Initializes the new instance.
	 *
	 * @param numberOfInputs number if inputs
	 * @param numberOfNeurons number of neurons (outputs)
	 */
	public Layer(int numberOfInputs, int numberOfNeurons) {
		mNumberOfInputs = numberOfInputs;
		mNumberOfNeurons = numberOfNeurons;

		// Create the array of weights
		// Each line in the array represents the weights of one neuron
		// Note that the number of weights of each neuron is bigger than the number of its inputs
		// - we will use the last element for the threshold (the weight of the fictional input of 1)
		mWeights = new double[numberOfNeurons][numberOfInputs + 1];

		// Set the initial weights to random values between -0.5 and +0.5
		for (int n = 0; n < numberOfNeurons; ++n) {
			for (int i = 0; i <= numberOfInputs; ++i) {
				mWeights[n][i] = Math.random() - 0.5;
			}
		}
	}

	/**
	 * Calculates the output of all neurons in the layer.
	 *
	 * @param inputs an array of input values
	 * @return an array of output values
	 */
	public double[] feedForward(double[] inputs) {
		// Remember the inputs, we will need them later for updating the weights
		mInputs = inputs;

		// Calculate the outputs of all neurons and remember them, we will need them later in the back propagation
		mOutputs = new double[mNumberOfNeurons];
		for (int n = 0; n < mNumberOfNeurons; ++n) {

			// Calculate the scalar multiplication of inputs and weights of the n-th neuron
			double sum = 0;
			for (int i = 0; i < mNumberOfInputs; ++i) {
				sum += mWeights[n][i] * inputs[i];
			}

			// Add the threshold (the fictional input of 1)
			sum += mWeights[n][mNumberOfInputs] * 1;

			// Apply the activation function and store the output
			mOutputs[n] = activation(sum);
		}

		return mOutputs;
	}

	/**
	 * Back-propagates the errors.
	 *
	 * @param outputErrors an array of errors on the output of the layer -
	 * differences between real (calculated) outputs and expected (correct)
	 * outputs.
	 * @return an array of errors back-propagated (transformed) to the inputs of
	 * the layer, where they can be used as errors on the outputs of the
	 * previous layer.
	 */
	public double[] backPropagate(double[] outputErrors) {
		// Propagate the error from the output to the post-synaptic potential of each neuron
		// This is done by multiplying the output error by the derivation of the logistic function (which is y*(1-y))
		mDeltas = new double[mNumberOfNeurons];
		for (int n = 0; n < mNumberOfNeurons; ++n) {
			mDeltas[n] = derivation(n) * outputErrors[n];
		}

		// Propagate the error from the post-synaptic potential of each neuron to the inputs
		// For each input, this is done by summing up the post-synaptic errors of all neurons multiplied by the weight of this input in the selected neuron.
		double[] inputErrors = new double[mNumberOfInputs];
		for (int i = 0; i < mNumberOfInputs; ++i) {
			inputErrors[i] = 0;
			for (int n = 0; n < mNumberOfNeurons; ++n) {
				inputErrors[i] += mWeights[n][i] * mDeltas[n];
			}
		}

		return inputErrors;
	}

	/**
	 * Updates the input weights of each neuron.
	 *
	 * @param epsilon the parameter which controls the learning step
	 */
	public void updateWeights(double epsilon) {
		for (int n = 0; n < mNumberOfNeurons; ++n) {
			for (int i = 0; i < mNumberOfInputs; ++i) {
				// Update the weight of the i-th input of n-th neuron, w(n,i) = w(n,i) - e * x(i) * delta(n)
				mWeights[n][i] -= epsilon * mInputs[i] * mDeltas[n];
			}
			// Update the threshold (weight of the fictional input of 1)
			mWeights[n][mNumberOfInputs] -= epsilon * 1 * mDeltas[n];
		}
	}

	/**
	 * Calculates the activation function from the post-synaptic potential using
	 * the logistic function.
	 *
	 * @param pot the post-synaptic potential (scalar multiplication of input
	 * vector * weights vector)
	 * @return the result of the activation function.
	 */
	protected double activation(double pot) {
		return 1 / (1 + Math.exp(-pot));
	}

	/**
	 * Calculates the derivation of the activation function for the n-th neuron.
	 *
	 * This must be done after the feed-forward step, because the derivation
	 * depends on the output value of the neuron.
	 *
	 * @param n the index of the neuron.
	 * @return the derivation of the activation function.
	 */
	protected double derivation(int n) {
		return mOutputs[n] * (1 - mOutputs[n]);
	}
}
