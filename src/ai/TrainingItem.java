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
 * This class represents one line in the training data set.
 *
 * It contains the expected outputs and is able to calculate the differences
 * between real and expected outputs and also the error function.
 *
 * @author Pavel Junek
 */
public class TrainingItem extends Item {

	/**
	 * The array of expected (correct) output values.
	 */
	public final double[] d;

	/**
	 * Initializes the new instance.
	 *
	 * @param x an array of input values
	 * @param d an array of expected output values
	 */
	public TrainingItem(double[] x, double[] d) {
		super(x);

		this.d = d;
	}

	/**
	 * Calculates the differences between real and expected outputs.
	 *
	 * @param Y the real outputs produced by the network
	 * @return an array of differences between real and expected outputs
	 */
	public double[] getDifferences(double[] Y) {
		double[] delta = new double[d.length];
		for (int i = 0; i < d.length; ++i) {
			delta[i] = Y[i] - d[i];
		}
		return delta;
	}

	/**
	 * Calculates the error value from the differences between real and expected
	 * outputs using the formula E = 1/2*sum((Y-d)^2).
	 *
	 * @param Y the real outputs produced by the network
	 * @return the error value
	 */
	public double getError(double[] Y) {
		double sum = 0;
		for (double delta : getDifferences(Y)) {
			sum += delta * delta;
		}
		return sum / 2;
	}

	/**
	 * Creates a new training item from the given array of strings.
	 *
	 * @param line a string containing the values (the line must contain only
	 * numeric values separated by commas)
	 * @param inputSize count of input values to use (the line must contain at
	 * least inputSize numeric values + 1 integer value representing the correct
	 * output for classification)
	 * @param outputSize count of output values (the array of output values is
	 * created from the last integer value of strings)
	 * @return the item created from the given array
	 */
	public static TrainingItem read(String line, int inputSize, int outputSize) {
		String[] parts = line.split(",");
		if (parts.length < inputSize + 1) {
			throw new IllegalArgumentException((inputSize + 1) + " values required, " + parts.length + " values given");
		}

		double[] x = new double[inputSize];
		for (int i = 0; i < inputSize; ++i) {
			x[i] = Double.valueOf(parts[i]);
		}

		int which = Integer.valueOf(parts[inputSize]);
		if (which >= outputSize) {
			throw new IllegalArgumentException("Output must be between 0 and " + (outputSize - 1) + ", " + which + " given");
		}
		double[] d = new double[outputSize];
		for (int i = 0; i < outputSize; ++i) {
			d[i] = i == which ? 1 : 0;
		}

		return new TrainingItem(x, d);
	}
}
