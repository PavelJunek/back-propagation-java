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

import java.util.List;

/**
 * This class represents a network with one hidden layer and an output layer.
 *
 * @author Pavel Junek
 */
public class Network {

	/**
	 * The constant affecting the speed of learning.
	 */
	private final double mEpsilon;

	/**
	 * The hidden layer.
	 */
	private final Layer mHiddenLayer;

	/**
	 * The output layer.
	 */
	private final Layer mOutputLayer;

	/**
	 * Initializes the new instance.
	 *
	 * @param numberOfInputs number of neurons in the input layer (or number of
	 * input values)
	 * @param numberOfHidden number of neurons in the hidden layer
	 * @param numberOfOutputs number of neurons in the output layer (or number
	 * of output values)
	 * @param epsilon constant affecting the speed of learning
	 */
	public Network(int numberOfInputs, int numberOfHidden, int numberOfOutputs, double epsilon) {
		mEpsilon = epsilon;

		mHiddenLayer = new Layer(numberOfInputs, numberOfHidden);
		mOutputLayer = new Layer(numberOfHidden, numberOfOutputs);
	}

	/**
	 * Classifies the given item.
	 *
	 * @param item the item to be classified
	 * @return the index of neuron which gave the largest output value
	 */
	public int classify(Item item) {
		// Get the outputs of all neurons
		double[] outputs = think(item);

		// Find the highest output and store its index
		int indexMax = 0;
		for (int index = 1; index < outputs.length; ++index) {
			if (outputs[index] > outputs[indexMax]) {
				indexMax = index;
			}
		}

		return indexMax;
	}

	/**
	 * Calculates the output for the given input.
	 *
	 * @param item the item to be processed
	 * @return the output of the network
	 */
	public double[] think(Item item) {
		return mOutputLayer.feedForward(
				mHiddenLayer.feedForward(item.x));
	}

	/**
	 * Performs one training step.
	 *
	 * This method first calculates the output of the network, then compares it
	 * with the expected (correct) output, back-propagates the error and updates
	 * the weights.
	 *
	 * @param item the training item to be used
	 */
	public void train(TrainingItem item) {
		double[] outputs = think(item);

		mHiddenLayer.backPropagate(
				mOutputLayer.backPropagate(
						item.getDifferences(outputs)));

		mOutputLayer.updateWeights(mEpsilon);
		mHiddenLayer.updateWeights(mEpsilon);
	}

	/**
	 * Trains the network using all items from the training set.
	 *
	 * @param trainingSet the set of training items
	 */
	public void trainEpoch(List<TrainingItem> trainingSet) {
		for (TrainingItem item : trainingSet) {
			train(item);
		}
	}

	/**
	 * Calculates the total error on the validation set.
	 *
	 * @param validationSet the set of validation items
	 * @return the total error
	 */
	public double getError(List<TrainingItem> validationSet) {
		double error = 0;
		for (TrainingItem item : validationSet) {
			double[] outputs = think(item);
			error += item.getError(outputs);
		}
		return error;
	}

}
